"""Diagnose a live SCF failure: the learned kinetic-energy coefficient."""
from .runtime import configure, torch_setup
ROOT = configure()

import argparse
import json
from pathlib import Path
import numpy as np
import torch
from .ensemble_screened import EnsembleScreenedXC
from .train_broad import Input


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=ROOT / "data/ensemble_tzvp_191")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    device = torch_setup()
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model = EnsembleScreenedXC(operator=checkpoint["operator"]).to(device).float().requires_grad_(False)
    model.load_compatible(checkpoint["model"])
    manifest = json.loads((args.data / "manifest.json").read_text())
    selected = sorted([row for row in manifest["molecules"] if row["split"] == "val"], key=lambda row: row["id"])[:4]
    report = {"scope": "Changed model's kinetic coefficient on inputs from the failing CPU SCF screen",
              "criterion": "T=integral(tau); 1+delta_E/delta_tau must not become strongly negative", "cases": []}
    for info in selected:
        case = Input(args.data / f"mol_{info['id']}.npz", device, model)
        case.tau.requires_grad_(True)
        energy = case.correction(model)
        slope = torch.autograd.grad(energy, case.tau)[0].detach().cpu().numpy()
        weights = case.template.weights64.cpu().numpy()
        density = case.rho[:, 0].detach().cpu().numpy()
        valid = (density > 1e-8) & (weights[None] > 1e-16)
        coefficient = 1+slope/np.maximum(weights[None], 1e-300)
        values = coefficient[valid]
        mass = weights[None]*density
        report["cases"].append({"id": info["id"], "effective_kinetic_coefficient_min": float(values.min()),
                                "quantiles_01_10_50_90": np.quantile(values, [.01, .1, .5, .9]).tolist(),
                                "electron_mass_fraction_negative_coefficient": float(mass[(coefficient < 0) & valid].sum()/mass.sum())})
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
