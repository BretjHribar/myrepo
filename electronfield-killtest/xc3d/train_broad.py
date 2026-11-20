"""Fit our screened-state XC on atomization differences, including atom fields.

Frozen TZVP inputs are a research experiment, not self-consistent challenge
scores. Formula families stay in their predetermined splits. Test is reported
only after the predeclared training schedule.
"""
from .runtime import configure, torch_setup
ROOT = configure()

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import numpy as np
import torch
from .model import SpatialTemplate
from .screened_state import ScreenedStateXC
from .ensemble_screened import EnsembleScreenedXC
from .exact_constraints import ConstraintShape, OrbitalConstraint

KCAL = 627.509474


class Input:
    def __init__(self, path, device, model):
        with np.load(path) as data:
            self.metadata = json.loads(str(data["metadata"]))
            self.rho = torch.tensor(data["rho"], dtype=torch.float32, device=device)
            self.tau = torch.tensor(data["tau"], dtype=torch.float32, device=device)
            self.template = SpatialTemplate(*[
                torch.tensor(data[key], dtype=torch.float32, device=device)
                for key in ("coords", "weights", "landmarks")
            ])
            self.template.weights64 = torch.tensor(data["weights"], dtype=torch.float64, device=device)
            self.quantum = ({"charge": torch.tensor(data["quantum_charge"], dtype=torch.float32, device=device),
                             "coherence": torch.tensor(data["quantum_coherence"], dtype=torch.float32, device=device)}
                            if "quantum_charge" in data else None)
            if self.quantum is not None and "hartree_matrix" in data:
                for key in ("hartree_matrix", "pbe_density"):
                    self.quantum[key] = torch.tensor(data[key], dtype=torch.float64, device=device)
        self.prepared = model.prepare(self.template)

    def correction(self, model):
        return model(self.rho, self.tau, self.template, self.prepared, quantum=self.quantum)


def predicted_tae(model, case, atomic):
    return (case.metadata["tae_pbe"] +
            sum(atomic[symbol] * count for symbol, count in Counter(case.metadata["symbols"]).items()) -
            case.correction(model))


@torch.no_grad()
def measure(model, atoms, cases, include_test=False):
    atomic = {symbol: atom.correction(model) for symbol, atom in atoms.items()}
    result = {}
    for split in (("train", "val", "test") if include_test else ("train", "val")):
        selected = [case for case in cases if case.metadata["split"] == split]
        if not selected:
            continue
        errors = [(float(predicted_tae(model, case, atomic)) - case.metadata["tae@w1-f12"]) * KCAL for case in selected]
        base = [(case.metadata["tae_pbe"] - case.metadata["tae@w1-f12"]) * KCAL for case in selected]
        result[split] = {"cases": len(selected), "tae_mae_kcal": float(np.abs(errors).mean()),
                         "tae_rmse_kcal": float(np.sqrt(np.mean(np.square(errors)))),
                         "pbe_input_tae_mae_kcal": float(np.abs(base).mean()),
                         "errors": [{"id": case.metadata["id"], "error_kcal": error}
                                    for case, error in zip(selected, errors)]}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/broad_tzvp_192")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20261116)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--constraints", type=Path, default=None)
    parser.add_argument("--constraint-weight", type=float, default=.3)
    parser.add_argument("--operator", choices=("diffusive", "coherent"), default="diffusive")
    parser.add_argument("--quantum", action="store_true")
    parser.add_argument("--architecture", choices=("screened_state", "ensemble_screened"), default="screened_state")
    args = parser.parse_args()
    device = torch_setup(args.seed)
    generator = np.random.default_rng(args.seed)
    if args.architecture == "ensemble_screened":
        if args.constraints:
            parser.error("Ensemble cancellation is structural; this arm does not use a competing identity loss")
        args.quantum = True
    model_class = EnsembleScreenedXC if args.architecture == "ensemble_screened" else ScreenedStateXC
    model = model_class(operator=args.operator, quantum=args.quantum).to(device).float()
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_compatible(checkpoint["model"])
    atoms = {path.stem.removeprefix("atom_"): Input(path, device, model) for path in sorted(args.data.glob("atom_*.npz"))}
    manifest = json.loads((args.data / "manifest.json").read_text())
    cases = [Input(args.data / f"mol_{row['id']}.npz", device, model)
             for row in sorted(manifest["molecules"], key=lambda row: row["id"])]
    training = [case for case in cases if case.metadata["split"] == "train"]
    constraint_class = OrbitalConstraint if args.quantum else ConstraintShape
    constraints = [constraint_class(path, device, model) for path in sorted(args.constraints.glob("*.npz"))] if args.constraints else []
    if not training or not atoms:
        raise RuntimeError("Broad input dataset is not ready")
    initial = measure(model, atoms, cases)
    optimizer = torch.optim.Adam(model.parameters(), lr=.0015, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.steps, eta_min=.0001)
    start = time.perf_counter()
    history = []
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        batch = [training[index] for index in generator.choice(len(training), size=min(args.batch_size, len(training)), replace=False)]
        needed_atoms = {symbol for case in batch for symbol in case.metadata["symbols"]}
        atomic = {symbol: atoms[symbol].correction(model) for symbol in needed_atoms}
        errors = torch.stack([predicted_tae(model, case, atomic) - case.metadata["tae@w1-f12"] for case in batch])
        loss = torch.nn.functional.huber_loss(errors / .03, torch.zeros_like(errors), delta=2.)
        constraint_energy = constraint_derivative = loss.new_zeros(())
        if constraints:
            shape = constraints[int(generator.integers(len(constraints)))]
            constraint_energy, constraint_derivative = shape.loss(model, generator)
            loss = loss + args.constraint_weight * (constraint_energy + constraint_derivative)
        loss.backward()
        if not bool(torch.isfinite(loss)) or not all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
            raise RuntimeError(f"Nonfinite new screened-state gradient at step {step}")
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
        optimizer.step()
        scheduler.step()
        if step == 0 or (step + 1) % 100 == 0 or step + 1 == args.steps:
            current = measure(model, atoms, cases)
            row = {"step": step + 1, "loss": float(loss.detach()), "elapsed_seconds": time.perf_counter() - start,
                   "constraint_energy_loss": float(constraint_energy.detach()),
                   "constraint_derivative_loss": float(constraint_derivative.detach()),
                   "metrics": current}
            history.append(row)
            print(json.dumps({"step": row["step"], "loss": row["loss"],
                              "tae_mae": {key: value["tae_mae_kcal"] for key, value in current.items()},
                              "elapsed_seconds": round(row["elapsed_seconds"], 2)}), flush=True)
    report = {"created_utc": datetime.now(timezone.utc).isoformat(),
              "status": "Own screened-state functional, frozen-input atomization experiment; not GMTKN55 or competition performance",
              "source": json.loads((args.data / "manifest.json").read_text())["source"],
              "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
              "initial": initial, "history": history, "final": measure(model, atoms, cases, include_test=True),
              "constraint_density_shapes": len(constraints),
              "runtime_seconds": time.perf_counter() - start,
              "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else None}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    torch.save({"model": model.state_dict(), "architecture": args.architecture, "width": model.width,
                "channels": model.channels, "operator": model.operator, "quantum": model.quantum_enabled,
                "version": 3}, args.output.with_suffix(".pt"))
    print(json.dumps({"final": {key: {name: value for name, value in row.items() if name != "errors"}
                                for key, row in report["final"].items()}}), flush=True)


if __name__ == "__main__":
    main()
