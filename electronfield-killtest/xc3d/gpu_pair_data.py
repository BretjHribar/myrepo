"""CUDA-only preparation for our constrained two-electron pair state.

Reuses saved reference labels and input fields. New Coulomb integrals and XC
terms run on GPU. Molecule/basis metadata and file I/O use the host; CPU integral
entry points are blocked. This is not a new FCI/reference calculation.
"""
from .runtime import configure, torch_setup
ROOT = configure(gpu_only_compatible=True)

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import numpy as np
import torch
from .model import density_features
from .gpu_semilocal import pbe_singlet


def radial_partition(coords, landmarks):
    displacement = coords[:, None]-landmarks[None]
    radius = displacement.square().sum(-1).clamp_min(1e-20).sqrt()
    atomic = torch.softmax(-radius.square()/2, -1)
    center = atomic @ landmarks
    atomic_log_gradient = landmarks[None]-center[:, None]
    core = torch.sigmoid((1.-radius)/.45)
    radial = torch.stack((core, 1-core), -1)
    partition = (atomic[..., None]*radial).flatten(-2)
    unit = displacement/radius[..., None]
    radial_log_gradient = torch.stack((-(1-core)[..., None]*unit/.45,
                                      core[..., None]*unit/.45), -2)
    log_gradient = (atomic_log_gradient[..., None, :]+radial_log_gradient).flatten(1, 2)
    return partition, log_gradient


def cuda_pbe(rho):
    return pbe_singlet(rho)


def cuda_eri(info):
    import cupy as cp
    from pyscf import gto
    from gpu4pyscf.scf.int4c2e import get_int4c2e
    mol = gto.M(atom=info["atom"], basis=info["basis"], charge=info["charge"], spin=info["spin"], verbose=0)
    def forbidden(*args, **kwargs):
        raise RuntimeError("CPU integral generation is forbidden by GPU-only policy")
    # The CUDA integral engine may handle basis bookkeeping on the host, but
    # must not call any CPU molecular-integral evaluator.
    previous = gto.Mole.intor, gto.Mole.intor_symmetric, gto.Mole.intor_asymmetric
    gto.Mole.intor = gto.Mole.intor_symmetric = gto.Mole.intor_asymmetric = forbidden
    try:
        result = get_int4c2e(mol)
        if not isinstance(result, cp.ndarray):
            raise RuntimeError("Integral result is not on CUDA")
        return torch.from_dlpack(result)
    finally:
        gto.Mole.intor, gto.Mole.intor_symmetric, gto.Mole.intor_asymmetric = previous


@torch.no_grad()
def prepare(path, output, device):
    start = time.perf_counter()
    with np.load(path) as data:
        info = json.loads(str(data["metadata"]))
        if info["nelec"] != [1, 1]:
            return None, {"id": info["id"], "nelec": info["nelec"],
                          "reason": "Outside this two-electron singlet mechanism experiment"}
        arrays = {key: torch.from_numpy(data[key]).to(device) for key in
                  ("rho", "coords", "weights", "landmarks", "ao", "orth", "dm", "fock", "energy", "energy_pbe")}
    rho, coords, weights, landmarks, ao, orth, dm, exact, baseline = [arrays[key] for key in
                  ("rho", "coords", "weights", "landmarks", "ao", "orth", "dm", "energy", "energy_pbe")]
    spin_difference = float((rho[:, 0]-rho[:, 1]).abs().max())
    if spin_difference > 1e-7:
        raise RuntimeError("This singlet experiment requires equal spin input fields")
    print(json.dumps({"id": info["id"], "phase": "cuda_integrals"}), flush=True)
    eri_cache = output.with_suffix(".eri.pt")
    eri = torch.load(eri_cache, map_location=device, weights_only=True) if eri_cache.exists() else cuda_eri(info)
    if not eri_cache.exists():
        torch.save(eri.cpu(), eri_cache)
    print(json.dumps({"id": info["id"], "phase": "cuda_xc"}), flush=True)
    xc = (cuda_pbe(rho)*weights).sum(-1)
    exchange = -.5*torch.einsum("bsij,bskl,ikjl->b", dm, dm, eri)
    hf = baseline-xc+exchange
    direct = torch.einsum("bij,ijkl,bkl->b", dm[:, 0], eri, dm[:, 1])
    # New CUDA operator consistency, using saved SCF data rather than rerunning
    # any reference. The trace identity catches basis ordering/XC mistakes.
    charges = torch.tensor([1. if row.strip().split()[0] == "H" else 2.
                            for row in info["atom"].split(";")], device=device)
    nuclear_distance = torch.cdist(landmarks, landmarks).clamp_min(1e-12)
    nuclear = torch.triu(charges[:, None]*charges[None, :]/nuclear_distance, diagonal=1).sum()
    with torch.enable_grad():
        amplitude = torch.ones(len(rho), device=device, requires_grad=True)
        scaled_xc = (pbe_singlet(rho*amplitude[:, None, None, None])*weights).sum(-1)
        xc_euler = torch.autograd.grad(scaled_xc.sum(), amplitude)[0]
    fock_trace = (dm*arrays["fock"]).sum((-1, -2, -3))
    trace_residual = fock_trace-baseline+xc+nuclear-2*direct-xc_euler
    if float(trace_residual.abs().max()) > 2e-5:
        raise RuntimeError(f"New CUDA operator reconstruction is inconsistent: {float(trace_residual.abs().max())} Hartree")
    partition, log_gradient = radial_partition(coords, landmarks)
    weighted = partition*weights[:, None]
    raw_mass = torch.einsum("bsg,ga->bsa", rho[:, :, 0], weighted)
    mass = raw_mass/raw_mass.sum(-1, keepdim=True)
    n, features = density_features(rho)
    total_mass = raw_mass.sum(-2)
    pooled = torch.einsum("bgf,bg,ga->baf", features, n, weighted)/(total_mass[..., None]+1e-12)
    site_centers = landmarks.repeat_interleave(2, 0)
    site_radius = (coords[:, None]-site_centers[None]).square().sum(-1)
    radius_moment = torch.einsum("bg,ga,ga->ba", n, weighted, site_radius)/(total_mass+1e-12)
    radial_label = torch.arange(len(site_centers), device=device).remainder(2).to(n.dtype)
    node_features = torch.cat((pooled, torch.log1p(total_mass)[..., None],
                               torch.log1p(radius_moment)[..., None],
                               radial_label[None, :, None].expand(len(rho), -1, -1)), -1)
    # Positive Fisher localization cost. All neighbours share the same pair
    # probability state rather than receiving independent energy lowerings.
    delta = log_gradient[:, :, None]-log_gradient[:, None, :]
    fisher = partition[:, :, None]*partition[:, None, :]*delta.square().sum(-1)/8
    edges = torch.einsum("bsg,g,gac->bsac", rho[:, :, 0], weights, fisher)
    # Project each density-localized orbital into the saved finite AO basis.
    target = (rho[:, 0, 0, :, None].clamp_min(0)*partition[None]).sqrt()
    rhs = torch.einsum("gi,g,bga->bia", ao[0], weights, target)
    inverse = orth @ orth
    overlap_root = torch.linalg.inv(orth)
    overlap = overlap_root @ overlap_root
    coefficients = inverse @ rhs
    norms = torch.einsum("bia,ij,bja->ba", coefficients, overlap, coefficients).clamp_min(1e-15).sqrt()
    coefficients = coefficients/norms[:, None]
    projected = torch.einsum("gi,bia->bga", ao[0], coefficients)
    normalized_target = target/(raw_mass[:, 0, None, :].clamp_min(1e-15).sqrt())
    projection_error = ((projected-normalized_target).square()*weights[None, :, None]).sum(-2)
    densities = torch.einsum("bia,bja->baij", coefficients, coefficients)
    interaction = torch.einsum("baij,ijkl,bdkl->bad", densities, eri, densities)
    uncorrelated = torch.einsum("ba,bad,bd->b", mass[:, 0], interaction, mass[:, 1])
    interaction = interaction*(direct/uncorrelated)[:, None, None]
    distance = torch.cdist(site_centers, site_centers)
    info = {**info, "mechanism_scope": "Two-electron singlets only; no general-electron coverage claim",
            "numerical_device": "cuda", "cpu_integral_fallback_disabled": True,
            "radial_modes_per_atom": 2, "maximum_orbital_projection_l2_error": float(projection_error.max()),
            "maximum_operator_trace_residual_hartree": float(trace_residual.abs().max()),
            "coulomb_normalization": "Matches actual input opposite-spin Hartree integral; uses no correlated labels",
            "seconds": time.perf_counter()-start}
    payload = {"metadata": info, "node_features": node_features, "mass": mass,
               "edges": edges, "interaction": interaction, "distance": distance,
               "energy_hf_input": hf, "energy_pbe": baseline, "energy": exact,
               "operator_eri": eri, "pbe_xc_energy": xc}
    torch.cuda.synchronize()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({key: value.cpu() if isinstance(value, torch.Tensor) else value for key, value in payload.items()}, output)
    return info, None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/pilot_3d_ccpvdz")
    parser.add_argument("--output", type=Path, default=ROOT / "data/gpu_pair_state")
    args = parser.parse_args()
    device = torch_setup()
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"created_utc": datetime.now(timezone.utc).isoformat(),
              "policy": "All numerical tensor work, ERIs and XC run on CUDA. Host metadata/I/O only.",
              "source": str(args.data), "cases": [], "outside_scope": [], "failures": []}
    for path in sorted(args.data.glob("*.npz")):
        try:
            output = args.output / path.with_suffix(".pt").name
            if output.exists():
                info = torch.load(output, map_location="cpu", weights_only=True)["metadata"]
                outside = None
            else:
                info, outside = prepare(path, output, device)
            if outside:
                report["outside_scope"].append(outside)
            else:
                report["cases"].append(info)
                print(json.dumps({"id": info["id"], "device": "cuda", "seconds": round(info["seconds"], 2)}), flush=True)
        except Exception as exc:
            report["failures"].append({"id": path.stem, "exception": f"{type(exc).__name__}: {exc}"})
            print(json.dumps(report["failures"][-1]), flush=True)
            # No repeated package-repair or silent CPU fallback campaign.
            if not report["cases"]:
                (args.output / "manifest.json").write_text(json.dumps(report, indent=2)+"\n")
                raise
        (args.output / "manifest.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({"prepared": len(report["cases"]), "outside_scope": len(report["outside_scope"]),
                      "failures": len(report["failures"])}), flush=True)


if __name__ == "__main__":
    main()
