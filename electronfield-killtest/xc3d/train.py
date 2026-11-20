"""Train our collective XC prototype on the finite-basis 3D pilot.

First-stage energies use fixed PBE input densities. Field/response losses use
one differentiable orbital update, not an entire converged SCF trajectory.
Neither metric is competition performance. Fresh-SCF evaluation is separate.
"""
from .runtime import configure, torch_setup
ROOT = configure()

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import numpy as np
import torch
from .model import CollectiveXC, OccupiedProjector, SpatialTemplate, rho_from_dm

KCAL = 627.509474


class Case:
    def __init__(self, path, device):
        with np.load(path) as data:
            self.metadata = json.loads(str(data["metadata"]))
            for key in ("rho", "rho_ref", "ao", "orth", "dm", "fock", "energy", "energy_pbe"):
                setattr(self, key, torch.tensor(data[key], device=device))
            self.template = SpatialTemplate(*[
                torch.tensor(data[key], device=device) for key in ("coords", "weights", "landmarks")
            ])
        self.occupations = torch.zeros(2, self.dm.shape[-1], device=device)
        for spin, count in enumerate(self.metadata["nelec"]):
            self.occupations[spin, :count] = 1
        self.id = self.metadata["id"]
        self.split = self.metadata["split"]
        self.family = self.metadata["family"]


def prediction(model, case):
    return case.energy_pbe + model(case.rho, case.template)


def orbital_update(model, case, all_states):
    limit = len(case.dm) if all_states else 1
    dm = case.dm[:limit].detach().requires_grad_(True)
    rho = rho_from_dm(dm, case.ao)
    energy = model(rho, case.template)
    potential = torch.autograd.grad(energy.sum(), dm, create_graph=True)[0]
    fock = case.fock[:limit] + potential
    orth = case.orth
    transformed = orth.T @ fock @ orth
    projector = OccupiedProjector.apply(transformed, case.occupations)
    new_dm = orth @ projector @ orth.T
    return rho_from_dm(new_dm, case.ao)


def field_loss(model, case, response=False):
    new_rho = orbital_update(model, case, response)
    n = new_rho[:, :, 0].sum(1)
    reference = case.rho_ref[:, :, 0].sum(1)
    weights = case.template.weights
    number = sum(case.metadata["nelec"])
    density = (weights * (n[0] - reference[0]).square() / (reference[0] + 1e-5)).sum() / number
    response_loss = density.new_zeros(())
    if response and len(n) == 9:
        predicted_response = (n[[1, 3, 5, 7]] - n[[2, 4, 6, 8]]) / n.new_tensor([.004, .008, .004, .008])[:, None]
        exact_response = (reference[[1, 3, 5, 7]] - reference[[2, 4, 6, 8]]) / n.new_tensor([.004, .008, .004, .008])[:, None]
        numerator = (weights * (predicted_response - exact_response).square()).sum(-1)
        denominator = (weights * exact_response.square()).sum(-1) + 1e-4
        response_loss = (numerator / denominator).mean()
    return density, response_loss


def energy_objective(model, cases):
    predictions = [prediction(model, case) for case in cases]
    errors = torch.stack([(predicted - case.energy).mean() for predicted, case in zip(predictions, cases)])
    loss = (errors / .03).square().mean()
    # Relative curves are separate from absolute molecular offsets.
    curve_terms = []
    for family in sorted({case.family for case in cases}):
        positions = [i for i, case in enumerate(cases) if case.family == family]
        if len(positions) > 1:
            curve = errors[positions]
            curve_terms.append(((curve[1:] - curve[0]) / .03).square().mean())
    if curve_terms:
        loss = loss + torch.stack(curve_terms).mean()
    # Small energy changes from perturbation are not replaced by an anchor mean.
    field_energy = torch.stack([
        (((predicted - predicted[0]) -
          (case.energy - case.energy[0])) / .002).square().mean() for predicted, case in zip(predictions, cases)
    ]).mean()
    return loss + .1 * field_energy


@torch.no_grad()
def metrics(model, cases):
    output = {}
    for split in ("train", "val", "test"):
        selected = [case for case in cases if case.split == split]
        if not selected:
            continue
        errors = [float((prediction(model, case) - case.energy)[0]) for case in selected]
        base = [float((case.energy_pbe - case.energy)[0]) for case in selected]
        curves, base_curves = [], []
        for family in sorted({case.family for case in selected}):
            positions = [i for i, case in enumerate(selected) if case.family == family]
            for position in positions[1:]:
                curves.append(errors[position] - errors[positions[0]])
                base_curves.append(base[position] - base[positions[0]])
        output[split] = {
            "frozen_energy_mae_kcal": float(np.abs(errors).mean() * KCAL),
            "pbe_energy_mae_kcal": float(np.abs(base).mean() * KCAL),
            "frozen_curve_difference_mae_kcal": float(np.abs(curves).mean() * KCAL) if curves else None,
            "pbe_curve_difference_mae_kcal": float(np.abs(base_curves).mean() * KCAL) if base_curves else None,
            "by_case": [{"id": case.id, "error_hartree": error, "pbe_error_hartree": baseline}
                        for case, error, baseline in zip(selected, errors, base)],
        }
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/pilot_3d_ccpvdz")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", choices=("energy", "density", "response"), default="energy")
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--seed", type=int, default=20261116)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--field-weight", type=float, default=2.)
    args = parser.parse_args()
    device = torch_setup(args.seed)
    cases = [Case(path, device) for path in sorted(args.data.glob("*.npz"))]
    train = [case for case in cases if case.split == "train"]
    if not train:
        raise RuntimeError("No training references")
    model = CollectiveXC(args.width, nonlocal_state=not args.local).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=.003, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=.00015)
    initial = metrics(model, cases)
    history = []
    start = time.perf_counter()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(args.epochs):
        optimizer.zero_grad(set_to_none=True)
        energy_loss = energy_objective(model, train)
        energy_loss.backward()
        density_value = response_value = 0.
        if args.arm != "energy":
            for case in train:
                density, response = field_loss(model, case, args.arm == "response")
                extra = args.field_weight * (density + response) / len(train)
                extra.backward()
                density_value += float(density.detach()) / len(train)
                response_value += float(response.detach()) / len(train)
        if not all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
            raise RuntimeError(f"Nonfinite new-model gradients at epoch {epoch}")
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
        optimizer.step()
        scheduler.step()
        if (epoch + 1) % 25 == 0 or epoch == 0 or epoch + 1 == args.epochs:
            result = {"epoch": epoch + 1, "energy_loss": float(energy_loss.detach()),
                      "density_loss": density_value, "response_loss": response_value,
                      "elapsed_seconds": round(time.perf_counter() - start, 2),
                      "metrics": metrics(model, cases)}
            history.append(result)
            print(json.dumps({key: value for key, value in result.items() if key != "metrics"} |
                             {"energy_mae": {split: row["frozen_energy_mae_kcal"] for split, row in result["metrics"].items()}}), flush=True)
    report = {"created_utc": datetime.now(timezone.utc).isoformat(),
              "status": "Finite-basis pilot; frozen-density and one-step metrics, not competition performance",
              "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
              "device": device, "parameters": sum(p.numel() for p in model.parameters()),
              "initial": initial, "history": history, "final": metrics(model, cases),
              "runtime_seconds": time.perf_counter() - start,
              "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else None}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    torch.save({"model": model.state_dict(), "width": args.width, "nonlocal_state": not args.local,
                "arm": args.arm, "seed": args.seed}, args.output.with_suffix(".pt"))


if __name__ == "__main__":
    main()
