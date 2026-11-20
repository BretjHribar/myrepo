"""Our density-assembled collective singlet pair wavefunction, CUDA only.

Constructs rotation-scalar radial, dipolar and inter-region pair channels from
the input occupied orbital. All regions contribute to shared channels, whose
coefficients follow a small variational eigenproblem. This is our mechanism
experiment, not an existing frontier implementation or a general-electron DFA.
No FCI solver is called; cached references are used only to report errors.
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
from .model import rho_from_dm
from .gpu_semilocal import pbe_singlet

KCAL = 627.509474


def symmetric_pair(left, right):
    return (left[:, None]*right[None]+right[:, None]*left[None])/2


def electronic_hamiltonian(dm, ao, eri, fock, weights, xc_function=pbe_singlet):
    density = dm.detach().requires_grad_(True)
    with torch.enable_grad():
        rho = rho_from_dm(density, ao)
        xc = (xc_function(rho)*weights).sum()
        potential = torch.autograd.grad(xc, density)[0]
    potential = (potential+potential.transpose(-1, -2))/2
    coulomb = torch.einsum("ijkl,kl->ij", eri, dm.sum(0))
    spin_one_body = fock-potential.detach()-coulomb
    if float((spin_one_body[0]-spin_one_body[1]).abs().max()) > 2e-5:
        raise RuntimeError("Reconstructed one-body operators disagree between spin channels")
    one_body = spin_one_body.mean(0)
    return (one_body+one_body.T)/2


def collective_basis(dm, ao, orth, coords, weights, landmarks):
    inverse_orth = torch.linalg.inv(orth)
    overlap = inverse_orth @ inverse_orth
    occupation = inverse_orth @ dm[0] @ inverse_orth
    _, orbitals = torch.linalg.eigh((occupation+occupation.T)/2)
    occupied = orth @ orbitals[:, -1]
    phi = ao[0] @ occupied
    displacement = coords[:, None]-landmarks[None]
    radius = displacement.square().sum(-1).clamp_min(1e-20).sqrt()
    partition = torch.softmax(-radius.square()/2, -1)
    atomic_mass = (weights[:, None]*phi.square()[:, None]*partition).sum(0)
    charge_field = phi[:, None]*partition
    radial_field = phi[:, None]*partition*radius/(1+radius)
    vector_field = (phi[:, None, None]*partition[:, :, None]*displacement/(1+radius[..., None]))
    charge = orth @ orth @ (ao[0].T @ (weights[:, None]*charge_field))
    radial = orth @ orth @ (ao[0].T @ (weights[:, None]*radial_field))
    vector = torch.einsum("ij,jak->iak", orth @ orth,
                           torch.einsum("gi,g,gak->iak", ao[0], weights, vector_field))
    # Pauli orthogonalization: every added orbital direction is virtual to c0.
    charge = charge-occupied[:, None]*(occupied @ overlap @ charge)[None]
    radial = radial-occupied[:, None]*(occupied @ overlap @ radial)[None]
    vector = vector-occupied[:, None, None]*torch.einsum("i,ij,jak->ak", occupied, overlap, vector)[None]
    count = len(landmarks)
    channels, labels = [], []
    channels.append(symmetric_pair(occupied, occupied))
    labels.append("reference")
    # Two density moments distinguish inequivalent regions without Z embeddings.
    for power in (0., 1.):
        site_weight = atomic_mass.pow(power)
        for name, scalar in (("charge", charge), ("radial", radial)):
            channels.append(sum(site_weight[a]*symmetric_pair(occupied, scalar[:, a]) for a in range(count)))
            labels.append(f"collective_{name}_single_moment_{power}")
        for name, first, second in (("charge", charge, charge), ("charge_radial", charge, radial), ("radial", radial, radial)):
            channels.append(sum(site_weight[a]*symmetric_pair(first[:, a], second[:, a]) for a in range(count)))
            labels.append(f"collective_{name}_pair_moment_{power}")
        channels.append(sum(site_weight[a]*sum(symmetric_pair(vector[:, a, axis], vector[:, a, axis])
                                                for axis in range(3)) for a in range(count)))
        labels.append(f"collective_angular_pair_moment_{power}")
    distance = torch.cdist(landmarks, landmarks)
    for length in (.75, 2., 5.):
        accumulators = [torch.zeros_like(channels[0]) for _ in range(8)]
        for a in range(count):
            for b in range(a):
                direction = (landmarks[a]-landmarks[b])/distance[a, b]
                weight = torch.exp(-distance[a, b].square()/(2*length**2))
                along_a, along_b = vector[:, a] @ direction, vector[:, b] @ direction
                accumulators[0] = accumulators[0]+weight*symmetric_pair(charge[:, a], charge[:, b])
                accumulators[1] = accumulators[1]+weight*(symmetric_pair(charge[:, a], radial[:, b])
                                                         +symmetric_pair(radial[:, a], charge[:, b]))
                accumulators[2] = accumulators[2]+weight*symmetric_pair(radial[:, a], radial[:, b])
                accumulators[3] = accumulators[3]+weight*sum(
                    symmetric_pair(vector[:, a, axis], vector[:, b, axis]) for axis in range(3))
                accumulators[4] = accumulators[4]+weight*symmetric_pair(along_a, along_b)
                accumulators[5] = accumulators[5]+weight*(symmetric_pair(radial[:, a], along_b)
                                                         -symmetric_pair(radial[:, b], along_a))
                accumulators[6] = accumulators[6]+weight*(symmetric_pair(charge[:, a], along_b)
                                                         -symmetric_pair(charge[:, b], along_a))
                accumulators[7] = accumulators[7]+weight*symmetric_pair(occupied, along_a-along_b)
        channels.extend(accumulators)
        labels.extend(f"{name}_length_{length}" for name in
                      ("charge_edges", "charge_radial_edges", "radial_edges", "angular_edges", "longitudinal_edges",
                       "radial_dipole_edges", "charge_dipole_edges", "polarization_single"))
    basis = torch.stack(channels)
    metric_images = overlap @ basis @ overlap
    norm = (basis*metric_images).sum((-1, -2)).clamp_min(0).sqrt()
    active = norm > 1e-10
    basis = basis[active]/norm[active, None, None]
    return basis, overlap, {"proposed_channels": len(channels), "nonzero_channels": int(active.sum()),
                            "labels": labels, "local_regions": count}


@torch.no_grad()
def calculate(info, args, device):
    started = time.perf_counter()
    cached = torch.load(args.operators / f"{info['id']}.pt", map_location=device, weights_only=True)
    eri = cached["operator_eri"]
    with np.load(args.data / f"{info['id']}.npz") as data:
        fields = {name: torch.from_numpy(data[name]).to(device) for name in
                  ("dm", "ao", "orth", "coords", "weights", "landmarks", "fock", "energy")}
    dm, ao, orth = fields["dm"][0], fields["ao"], fields["orth"]
    one_body = electronic_hamiltonian(dm, ao, eri, fields["fock"][0], fields["weights"])
    basis, overlap, construction = collective_basis(dm, ao, orth, fields["coords"], fields["weights"], fields["landmarks"])
    metric_images = overlap @ basis @ overlap
    h_images = one_body @ basis @ overlap+overlap @ basis @ one_body
    h_images = h_images+torch.einsum("ikjl,akl->aij", eri, basis)
    metric = torch.einsum("aij,bij->ab", basis, metric_images)
    hamiltonian = torch.einsum("aij,bij->ab", basis, h_images)
    values, vectors = torch.linalg.eigh((metric+metric.T)/2)
    retained = values > values.max()*1e-9
    transformation = vectors[:, retained]/values[retained].sqrt()
    projected = transformation.T @ ((hamiltonian+hamiltonian.T)/2) @ transformation
    energies, coefficients = torch.linalg.eigh((projected+projected.T)/2)
    pair = torch.einsum("a,aij->ij", transformation @ coefficients[:, 0], basis)
    normalized = float((pair*(overlap @ pair @ overlap)).sum())
    nuclear_charges = torch.tensor([1. if row.strip().split()[0] == "H" else 2.
                                    for row in info["atom"].split(";")], device=device)
    nuclear_distances = torch.cdist(fields["landmarks"], fields["landmarks"]).clamp_min(1e-12)
    nuclear = torch.triu(nuclear_charges[:, None]*nuclear_charges[None]/nuclear_distances, diagonal=1).sum()
    total = energies[0]+nuclear
    reference = fields["energy"][0]
    error = float((total-reference)*KCAL)
    # Compare scope against the complete AO singlet space, not just raw size.
    full_dimension = dm.shape[-1]*(dm.shape[-1]+1)//2
    torch.cuda.synchronize()
    return {"id": info["id"], "family": info["family"], "split": info["split"],
            "energy_hartree": float(total), "error_kcal": error,
            "reference_hartree": float(reference), "input_hf_error_kcal": float((cached["energy_hf_input"][0]-reference)*KCAL),
            "pbe_input_error_kcal": float((cached["energy_pbe"][0]-reference)*KCAL),
            "retained_channels": int(retained.sum()), "full_ao_singlet_dimension": full_dimension,
            "wavefunction_norm": normalized, "variational_bound_satisfied": error > -1e-3,
            "metric_min_retained": float(values[retained].min()), "construction": construction,
            "device": device, "seconds": time.perf_counter()-started}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/pilot_3d_ccpvdz")
    parser.add_argument("--operators", type=Path, default=ROOT / "data/gpu_pair_state")
    parser.add_argument("--output", type=Path, default=Path("research/charge_angular_geminal_gpu_20261116.json"))
    args = parser.parse_args()
    device = torch_setup()
    manifest = json.loads((args.operators / "manifest.json").read_text())
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "device": device,
              "scope": "Our collective rotation-scalar pair trial space. Two-electron, finite-basis mechanism experiment; not GMTKN55 or a universal functional.",
              "training": "None: reference energies used only for reporting",
              "priority": "Accuracy first; no speed optimization", "cases": [], "failures": [],
              "outside_scope": manifest["outside_scope"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for info in manifest["cases"]:
        try:
            row = calculate(info, args, device)
            report["cases"].append(row)
            print(json.dumps({key: row[key] for key in ("id", "error_kcal", "retained_channels", "full_ao_singlet_dimension", "seconds")}), flush=True)
        except Exception as exc:
            failure = {"id": info["id"], "exception": f"{type(exc).__name__}: {exc}"}
            report["failures"].append(failure)
            print(json.dumps(failure), flush=True)
        args.output.write_text(json.dumps(report, indent=2)+"\n")
    report["summary"] = {}
    for split in ("train", "val", "test"):
        rows = [row for row in report["cases"] if row["split"] == split]
        if rows:
            errors = torch.tensor([row["error_kcal"] for row in rows], device=device)
            report["summary"][split] = {"cases": len(rows), "mae_kcal": float(errors.abs().mean()),
                                        "maximum_error_kcal": float(errors.abs().max()),
                                        "variational_violations": sum(not row["variational_bound_satisfied"] for row in rows)}
    report["peak_cuda_bytes"] = torch.cuda.max_memory_allocated()
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({"summary": report["summary"], "failures": report["failures"]}), flush=True)


if __name__ == "__main__":
    main()
