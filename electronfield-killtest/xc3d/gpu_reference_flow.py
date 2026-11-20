"""Reference-complete version of our connected correlation state.

Keeps the spin-projected input determinant inside the physical trial block,
so correlation never relies on the compact pair basis to reconstruct its own
reference. No reference energies enter optimization. All tensor work is CUDA.
"""
from .runtime import configure, torch_setup
ROOT = configure(gpu_only_compatible=True)

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import numpy as np
import torch
from .gpu_orbital_flow import OrbitalSpinChargeFlow
from .gpu_sector_flow import optimize
from .gpu_connected_flow import h3_action


def augment(source, device):
    data = torch.load(source, map_location=device, weights_only=True)
    name = data["metadata"]["id"]
    output = source.with_name(name+".reference_seed.pt")
    if output.exists():
        return output
    with np.load(ROOT / "data/pilot_3d_ccpvdz" / f"{name}.npz") as saved:
        dm = torch.from_numpy(saved["dm"][0]).to(device)
        orth = torch.from_numpy(saved["orth"]).to(device)
    root = torch.linalg.inv(orth)
    occupations, orbitals = torch.linalg.eigh(root @ dm @ root)
    up, down = orbitals[0, :, -2:], orbitals[1, :, -1]
    determinant = (up[:, 0, None]*up[None, :, 1]-up[:, 1, None]*up[None, :, 0])[:, :, None]*down[None, None]
    # P_{S=1/2}=(15/4-S^2)/3 in the three-electron Ms=1/2 sector.
    projected = (2*determinant+determinant.transpose(1, 2)-determinant.permute(2, 0, 1))/3
    projected = projected/projected.square().sum().sqrt()
    spin_error = (projected-projected.transpose(1, 2)+projected.permute(2, 0, 1)).abs().max()
    if float(spin_error) > 1e-10:
        raise RuntimeError("Projected input determinant is not a doublet")
    reference_energy = (projected*h3_action(projected, data["h"], data["eri"])).sum()+data["nuclear"]
    data["sector_block"] = torch.cat((projected[None], data["sector_block"]))
    data["sector_block"] = data["sector_block"]/data["sector_block"].square().sum((1, 2, 3), keepdim=True).sqrt()
    data["input_reference_energy"] = reference_energy
    data["input_reference_kind"] = "Spin-projected input determinant; explicitly retained"
    torch.save({key: value.cpu() if isinstance(value, torch.Tensor) else value for key, value in data.items()}, output)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--output", type=Path, default=Path("research/reference_complete_flow_gpu_20261116.json"))
    args = parser.parse_args()
    device = torch_setup()
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "device": device,
              "scope": "Reference-complete particle/spin-conserving correlation state; three-electron development tests only",
              "training_targets": "Hamiltonian expectation only, no reference-energy loss",
              "validity_condition": "Trial space contains spin-projected mean-field input and must not raise its best energy",
              "steps": args.steps, "cases": [], "failures": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for source in sorted((ROOT / "data/gpu_odd_operators").glob("*.flow_seed.pt")):
        try:
            path = augment(source, device)
            row = optimize(path, args, device, model_class=OrbitalSpinChargeFlow)
            data = torch.load(path, map_location=device, weights_only=True)
            row["input_reference_error_kcal"] = float((data["input_reference_energy"]-data["reference"])*627.509474)
            row["reference_not_degraded"] = row["energy_hartree"] <= float(data["input_reference_energy"])+1e-7
            if not row["reference_not_degraded"]:
                raise RuntimeError("Variational reference coverage failed")
            report["cases"].append(row)
        except Exception as exc:
            failure = {"id": source.stem, "exception": f"{type(exc).__name__}: {exc}"}
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
