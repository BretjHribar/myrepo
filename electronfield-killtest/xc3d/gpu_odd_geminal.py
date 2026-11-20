"""A number-conserving three-electron extension of our collective pair state.

Antisymmetrizes the shared singlet geminal with a freely moving unpaired
electron. Uses the same density-assembled channels, no fitted energy offsets.
All contractions, integrals and eigensolves run on CUDA; no FCI solver call.
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
from .gpu_pair_data import cuda_eri
from .gpu_geminal import collective_basis, electronic_hamiltonian
from .gpu_semilocal import pbe_spin

KCAL = 627.509474


@torch.no_grad()
def calculate(path, output_cache, device):
    started = time.perf_counter()
    with np.load(path) as data:
        info = json.loads(str(data["metadata"]))
        if info["nelec"] != [2, 1]:
            raise ValueError("This experiment is the two-up, one-down extension")
        fields = {name: torch.from_numpy(data[name]).to(device) for name in
                  ("dm", "ao", "orth", "coords", "weights", "landmarks", "fock", "energy", "energy_pbe")}
    dm, ao, orth = fields["dm"][0], fields["ao"], fields["orth"]
    eri_cache = output_cache / f"{info['id']}.eri.pt"
    if eri_cache.exists():
        eri = torch.load(eri_cache, map_location=device, weights_only=True)
    else:
        eri = cuda_eri(info)
        output_cache.mkdir(parents=True, exist_ok=True)
        torch.save(eri.cpu(), eri_cache)
    one_body = electronic_hamiltonian(dm, ao, eri, fields["fock"][0], fields["weights"], xc_function=pbe_spin)
    paired_dm = torch.stack((dm[1], dm[1]))
    pair_basis, overlap, construction = collective_basis(paired_dm, ao, orth, fields["coords"], fields["weights"], fields["landmarks"])
    inverse_orth = torch.linalg.inv(orth)
    pair = inverse_orth @ pair_basis @ inverse_orth
    nao = len(orth)
    identity = torch.eye(nao, dtype=orth.dtype, device=device)
    # Wedge the geminal with every unpaired-orbital direction. C_ijk is
    # antisymmetric in the two up-spin indices; its inner product includes 1/2.
    tensor = (torch.einsum("ui,ajk->auijk", identity, pair)
              -torch.einsum("uj,aik->auijk", identity, pair)).flatten(0, 1)
    metric = .5*torch.einsum("aijk,bijk->ab", tensor, tensor)
    values, vectors = torch.linalg.eigh((metric+metric.T)/2)
    retained = values > values.max()*1e-9
    transformation = vectors[:, retained]/values[retained].sqrt()
    # Orthogonalizing the trial space before H action avoids work on its null modes.
    tensor = torch.einsum("ar,aijk->rijk", transformation, tensor)
    h = orth.T @ one_body @ orth
    integrals = torch.einsum("ia,ijkl->ajkl", orth, eri)
    integrals = torch.einsum("jb,ajkl->abkl", orth, integrals)
    integrals = torch.einsum("kc,abkl->abcl", orth, integrals)
    integrals = torch.einsum("ld,abcl->abcd", orth, integrals)
    acted = (torch.einsum("il,aljk->aijk", h, tensor)
             +torch.einsum("jl,ailk->aijk", h, tensor)
             +torch.einsum("kl,aijl->aijk", h, tensor))
    acted = acted+torch.einsum("iljm,almk->aijk", integrals, tensor)
    acted = acted+torch.einsum("ilkm,aljm->aijk", integrals, tensor)
    acted = acted+torch.einsum("jlkm,ailm->aijk", integrals, tensor)
    projected = .5*torch.einsum("aijk,bijk->ab", tensor, acted)
    energies, coefficients = torch.linalg.eigh((projected+projected.T)/2)
    seed = torch.einsum("a,aijk->ijk", coefficients[:, 0], tensor)
    sector_block = torch.einsum("ab,aijk->bijk", coefficients[:, :8], tensor)
    distances = torch.cdist(fields["landmarks"], fields["landmarks"]).clamp_min(1e-12)
    nuclear = torch.triu(1/distances, diagonal=1).sum()  # These two new cases contain H only.
    total = energies[0]+nuclear
    exact = fields["energy"][0]
    error = float((total-exact)*KCAL)
    from .gpu_pair_data import radial_partition
    partition, _ = radial_partition(fields["coords"], fields["landmarks"])
    local_overlap = ao[0].T[None] @ ((fields["weights"][:, None]*partition).T[:, :, None]*ao[0][None])
    regions = orth.T @ local_overlap @ orth
    cache = {"metadata": info, "seed": seed, "sector_block": sector_block,
             "h": h, "eri": integrals, "regions": regions, "landmarks": fields["landmarks"],
             "nuclear": nuclear, "reference": exact, "initial_energy": total}
    torch.save({key: value.cpu() if isinstance(value, torch.Tensor) else value for key, value in cache.items()},
               output_cache / f"{info['id']}.flow_seed.pt")
    torch.cuda.synchronize()
    return {"id": info["id"], "family": info["family"], "nelec": info["nelec"],
            "energy_hartree": float(total), "reference_hartree": float(exact), "error_kcal": error,
            "pbe_input_error_kcal": float((fields["energy_pbe"][0]-exact)*KCAL),
            "retained_trial_states": int(retained.sum()), "proposed_trial_states": len(metric),
            "full_fixed_spin_dimension": nao*(nao-1)//2*nao,
            "variational_bound_satisfied": error > -1e-3, "construction": construction,
            "device": device, "seconds": time.perf_counter()-started}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/pilot_3d_ccpvdz")
    parser.add_argument("--output", type=Path, default=Path("research/odd_collective_geminal_gpu_20261116.json"))
    args = parser.parse_args()
    device = torch_setup()
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "device": device,
              "scope": "First three-electron extension of our variational collective pair construction; finite-basis mechanism evidence only",
              "training": "None", "cases": [], "failures": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for path in sorted(args.data.glob("H3_neutral_*.npz")):
        try:
            row = calculate(path, ROOT / "data/gpu_odd_operators", device)
            report["cases"].append(row)
            print(json.dumps(row), flush=True)
        except Exception as exc:
            failure = {"id": path.stem, "exception": f"{type(exc).__name__}: {exc}"}
            report["failures"].append(failure)
            print(json.dumps(failure), flush=True)
        args.output.write_text(json.dumps(report, indent=2)+"\n")
    if report["cases"]:
        errors = torch.tensor([row["error_kcal"] for row in report["cases"]], device=device)
        report["summary"] = {"cases": len(errors), "mae_kcal": float(errors.abs().mean()),
                             "maximum_error_kcal": float(errors.abs().max())}
    report["peak_cuda_bytes"] = torch.cuda.max_memory_allocated()
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({"summary": report.get("summary"), "failures": report["failures"]}), flush=True)


if __name__ == "__main__":
    main()
