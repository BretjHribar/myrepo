"""First mechanism experiment for our Pauli-protected pair-state functional."""
from .runtime import configure, torch_setup
ROOT = configure()

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import numpy as np
import torch
from pyscf import dft, gto, lib
from .pair_state import PairStateXC
from .collective_pair import CollectivePairXC
from .conserved_hole import ConservedHoleXC
from .train import Case, KCAL


class PairCase(Case):
    def __init__(self, path, device, model):
        super().__init__(path, device)
        info = self.metadata
        mol = gto.M(atom=info["atom"], basis=info["basis"], charge=info["charge"], spin=info["spin"], verbose=0)
        self.eri = torch.tensor(mol.intor("int2e", aosym="s1"), device=device)
        ni = dft.numint.NumInt()
        energies = []
        for rho in self.rho.cpu().numpy():
            eps = ni.eval_xc_eff("pbe", rho, deriv=0, spin=1)[0]
            energies.append(np.dot(self.template.weights.cpu().numpy(), eps * rho[:, 0].sum(0)))
        self.pbe_xc = torch.tensor(energies, device=device)
        self.prepared = model.prepare(self.ao, self.template)


def prediction(model, case):
    return case.energy_pbe - case.pbe_xc + model(case.dm, case.ao, case.template, case.eri, case.prepared)


@torch.no_grad()
def metrics(model, cases):
    result = {}
    for split in ("train", "val", "test"):
        selected = [case for case in cases if case.split == split]
        errors = [(float((prediction(model, case) - case.energy)[0]), case) for case in selected]
        if errors:
            curve_errors = []
            for family in sorted({case.family for case in selected}):
                group = [error for error, case in errors if case.family == family]
                curve_errors.extend(error - group[0] for error in group[1:])
            result[split] = {"frozen_energy_mae_kcal": float(np.mean([abs(error) for error, _ in errors]) * KCAL),
                             "frozen_curve_difference_mae_kcal": float(np.mean(np.abs(curve_errors)) * KCAL),
                             "by_case": [{"id": case.id, "error_hartree": error} for error, case in errors]}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--seed", type=int, default=20261116)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--architecture", choices=("pair_state", "collective_pair", "conserved_hole"), default="pair_state")
    parser.add_argument("--flat-spin", action="store_true")
    parser.add_argument("--data", type=Path, default=ROOT / "data/pilot_3d_ccpvdz")
    args = parser.parse_args()
    device = torch_setup(args.seed)
    lib.num_threads(1)
    model_class = {"pair_state": PairStateXC, "collective_pair": CollectivePairXC,
                   "conserved_hole": ConservedHoleXC}[args.architecture]
    model = model_class().to(device)
    cases = [PairCase(path, device, model) for path in sorted(args.data.glob("*.npz"))]
    train = [case for case in cases if case.split == "train"]
    spin_constraints = []
    if args.flat_spin:
        for case in train:
            if sum(case.metadata["nelec"]) != 1:
                continue
            with np.load(args.data / f"{case.id}.npz") as data:
                dm_ref = torch.tensor(data["dm_ref"][0].sum(0), device=device)
            for fraction in (.25, .5, .75):
                mixed = torch.stack((fraction * dm_ref, (1-fraction) * dm_ref))
                mol = gto.M(atom=case.metadata["atom"], basis=case.metadata["basis"],
                            charge=case.metadata["charge"], spin=case.metadata["spin"], verbose=0)
                hcore = torch.tensor(mol.intor("int1e_kin") + mol.intor("int1e_nuc"), device=device)
                hartree = .5 * torch.einsum("ij,kl,ijkl->", dm_ref, dm_ref, case.eri)
                rest = (dm_ref * hcore).sum() + hartree + mol.energy_nuc()
                spin_constraints.append((case, mixed, rest, case.energy[0]))
    initial = metrics(model, cases)
    optimizer = torch.optim.Adam(model.parameters(), lr=.005)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=.0002)
    history = []
    start = time.perf_counter()
    for epoch in range(args.epochs):
        optimizer.zero_grad(set_to_none=True)
        errors = torch.stack([(prediction(model, case) - case.energy).mean() for case in train])
        loss = (errors / .03).square().mean()
        for family in sorted({case.family for case in train}):
            indexes = [i for i, case in enumerate(train) if case.family == family]
            if len(indexes) > 1:
                curve = errors[indexes]
                loss = loss + ((curve[1:] - curve[0]) / .03).square().mean()
        if spin_constraints:
            terms = [((rest + model(dm, case.ao, case.template, case.eri, case.prepared) - target) / .03).square()
                     for case, dm, rest, target in spin_constraints]
            loss = loss + torch.stack(terms).mean()
        loss.backward()
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("Nonfinite pair-state loss")
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
        optimizer.step()
        scheduler.step()
        if epoch == 0 or (epoch + 1) % 50 == 0 or epoch + 1 == args.epochs:
            row = {"epoch": epoch + 1, "loss": float(loss.detach()), "metrics": metrics(model, cases),
                   "elapsed_seconds": time.perf_counter() - start}
            history.append(row)
            print(json.dumps({"epoch": row["epoch"], "loss": row["loss"],
                              "energy_mae": {key: value["frozen_energy_mae_kcal"] for key, value in row["metrics"].items()},
                              "elapsed_seconds": round(row["elapsed_seconds"], 2)}), flush=True)
    report = {"created_utc": datetime.now(timezone.utc).isoformat(),
              "status": "Our second candidate; finite-basis frozen-input mechanism experiment, not competition performance",
              "seed": args.seed, "epochs": args.epochs, "architecture": args.architecture,
              "flat_spin_constraints": len(spin_constraints), "initial": initial,
              "history": history, "final": metrics(model, cases),
              "runtime_seconds": time.perf_counter() - start,
              "limitation": "Explicit AO exchange in this prototype; large-system speed target not established"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    torch.save({"model": model.state_dict(), "width": model.width, "architecture": args.architecture}, args.output.with_suffix(".pt"))


if __name__ == "__main__":
    main()
