"""Out-of-benchmark test: fit on GMTKN55 reactions, predict W1-F12 atomization energies of molecules
from the public MSR-ACC/TAE25 file that the fit has never seen.

The fit sees no TAE25 label. Labels are read only to score. Errors are plain (unweighted) kcal/mol on
total atomization energies, which is a different quantity from WTMAD-2.
"""
import argparse
import json
from pathlib import Path

from .common import BULK, pin_one_core

pin_one_core()
import torch
from . import fitlib as fl
from .fit import LADDER


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="tae25")
    ap.add_argument("--scf-tag", default="pbe_qzvp_g3")
    ap.add_argument("--models", default="K:,N:")
    ap.add_argument("--lam", type=float, default=1e-6)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    train_rows, _ = fl.structure_table(args.scf_tag)
    ext_rows, ext_problems = fl.structure_table(args.scf_tag, dataset=args.dataset)
    labels = json.loads((BULK / args.dataset / "labels.json").read_text())
    mols = [m for m in labels if m in ext_rows and all(f"atom/{a}" in ext_rows for a in labels[m]["atoms"])]
    print(f"external molecules with every ingredient: {len(mols)} (problems: {len(ext_problems)})")
    if not mols:
        return
    ref = torch.tensor([labels[m]["tae_hartree"] for m in mols], dtype=torch.float64, device=fl.DEV) * fl.KCAL
    elements = sorted({a for m in mols for a in labels[m]["atoms"]})
    counts = torch.tensor([[labels[m]["atoms"].get(a, 0) for a in elements] for m in mols], dtype=torch.float64, device=fl.DEV)
    report = {"n_external": len(mols), "models": []}

    def tae(base, keys, theta):
        e_mol, f_mol = fl.structure_energy_terms(ext_rows, mols, keys, base)
        e_atom, f_atom = fl.structure_energy_terms(ext_rows, [f"atom/{a}" for a in elements], keys, base)
        em = e_mol + (f_mol @ theta if len(keys) else 0.0)
        ea = e_atom + (f_atom @ theta if len(keys) else 0.0)
        return (counts @ ea - em) * fl.KCAL

    for label, base, blocks in LADDER:
        keys = [k for b in blocks for k in fl.FEATURE_SETS[b]]
        if keys and not any(label.startswith(x) for x in args.models.split(",")):
            continue
        if keys:
            prob = fl.Problem(train_rows, keys, base)
            theta = fl.solve(prob.A, prob.y, prob.w, args.lam)[0]
            n_train = len(prob.y)
        else:
            theta, n_train = torch.zeros(0, dtype=torch.float64, device=fl.DEV), 0
        err = tae(base, keys, theta) - ref
        ok = torch.isfinite(err)
        e = err[ok]
        row = {"label": label, "p": len(keys), "n_train_reactions": n_train, "n_scored": int(ok.sum()),
               "mae": float(e.abs().mean()), "mean_signed": float(e.mean()), "rmse": float((e * e).mean().sqrt()),
               "max_abs": float(e.abs().max()), "p95_abs": float(torch.quantile(e.abs(), 0.95))}
        report["models"].append(row)
        print(f"{label:72s} p={row['p']:3d} trained on {n_train:4d} GMTKN55 reactions | external TAE: MAE {row['mae']:6.2f}  "
              f"mean {row['mean_signed']:+6.2f}  RMSE {row['rmse']:6.2f}  95th pct {row['p95_abs']:6.2f}  max {row['max_abs']:6.2f}  (n={row['n_scored']})")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
