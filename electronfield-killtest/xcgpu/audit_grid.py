"""Numerical-noise audit: how much do reaction energies change when the integration grid is refined?

Fits the model on the production data, then evaluates the same fitted parameters on ingredients
recomputed with a finer grid for a set of high-weight reactions. Reported per reaction and per
subset, for the PBE input, the unfitted exact-exchange + RPA energy and the fitted model.
"""
import argparse
import json

from .common import pin_one_core

pin_one_core()
import torch
from . import fitlib as fl
from .fit import LADDER


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fine-tag", default="pbe_qzvp_g5")
    ap.add_argument("--model", default="N:")
    ap.add_argument("--lam", type=float, default=1e-6)
    args = ap.parse_args()
    rows, _ = fl.structure_table("pbe_qzvp_g3")
    fine, problems = fl.structure_table(args.fine_tag)
    label, base, blocks = next(m for m in LADDER if m[0].startswith(args.model))
    keys = [k for b in blocks for k in fl.FEATURE_SETS[b]]
    prob = fl.Problem(rows, keys, base)
    theta = fl.solve(prob.A, prob.y, prob.w, args.lam)[0]
    s = prob.suite
    print(f"model {label}; fine-grid structures {len(fine)}; problems {len(problems)}")
    out = {}
    for r in s.reactions:
        ids = r["structure_ids"]
        if not all(i in fine and i in rows for i in ids):
            continue
        coef = torch.tensor(r["coefficients"], dtype=torch.float64, device=fl.DEV)
        vals = {}
        for tag, table in (("coarse", rows), ("fine", fine)):
            e0, f = fl.structure_energy_terms(table, ids, keys, base)
            pbe = torch.tensor([table[i]["e_pbe"] for i in ids], dtype=torch.float64, device=fl.DEV)
            vals[tag] = (float(coef @ pbe) * fl.KCAL, float(coef @ e0) * fl.KCAL, float(coef @ (e0 + f @ theta)) * fl.KCAL)
        d = [vals["fine"][k] - vals["coarse"][k] for k in range(3)]
        out.setdefault(r["subset"], []).append((r["id"], float(r["reference_kcal_per_mol_text"]), d))
    print(f"{'subset':10s} {'n':>3s}  mean |change| on refining the grid, kcal/mol:  PBE    EXX+RPA   fitted model   (max for fitted)   weight")
    w = {name: float(s.protocol["normalization_kcal_per_mol"]) / float(meta["average_kcal_per_mol_text"]) for name, meta in s.subsets.items()}
    for name, items in out.items():
        m = [sum(abs(x[2][k]) for x in items) / len(items) for k in range(3)]
        print(f"{name:10s} {len(items):3d}  {'':45s} {m[0]:6.3f}  {m[1]:7.3f}  {m[2]:9.3f}        {max(abs(x[2][2]) for x in items):6.3f}        {w[name]:6.1f}"
              f"   -> weighted {m[2]*w[name]:.2f}")
    worst = sorted((x for items in out.values() for x in items), key=lambda x: -abs(x[2][2]))[:8]
    for rid, ref, d in worst:
        print(f"   {rid:14s} ref {ref:7.2f}  change: PBE {d[0]:+.3f}  EXX+RPA {d[1]:+.3f}  fitted {d[2]:+.3f}")


if __name__ == "__main__":
    main()
