"""How does the strength of the regularisation trade fit quality against transfer?

For one model, scan lambda and report: in-sample error, random-fold and whole-subset-held-out
cross-validation on GMTKN55, and the error on the unseen external atomization energies when the fit
uses GMTKN55 only. Larger lambda pulls every coefficient toward zero, i.e. toward the unfitted
exact-exchange + RPA energy.
"""
import argparse
import json

from .common import BULK, pin_one_core

pin_one_core()
import torch
from . import fitlib as fl
from .fit import LADDER


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="N:")
    ap.add_argument("--lams", default="1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1")
    ap.add_argument("--dataset", default="tae25")
    args = ap.parse_args()
    rows, _ = fl.structure_table()
    ext, _ = fl.structure_table(dataset=args.dataset)
    labels = json.loads((BULK / args.dataset / "labels.json").read_text())
    label, base, blocks = next(m for m in LADDER if m[0].startswith(args.model))
    keys = [k for b in blocks for k in fl.FEATURE_SETS[b]]
    prob = fl.Problem(rows, keys, base)
    mols = sorted(m for m in labels if m in ext and all(f"atom/{a}" in ext for a in labels[m]["atoms"]))
    elements = sorted({a for m in mols for a in labels[m]["atoms"]})
    counts = torch.tensor([[labels[m]["atoms"].get(a, 0) for a in elements] for m in mols], dtype=torch.float64, device=fl.DEV)
    ref = torch.tensor([labels[m]["tae_hartree"] for m in mols], dtype=torch.float64, device=fl.DEV) * fl.KCAL
    e_mol, f_mol = fl.structure_energy_terms(ext, mols, keys, base)
    e_atom, f_atom = fl.structure_energy_terms(ext, [f"atom/{a}" for a in elements], keys, base)
    a_ext = (counts @ f_atom - f_mol) * fl.KCAL
    y_ext = ref - (counts @ e_atom - e_mol) * fl.KCAL
    ok = torch.isfinite(a_ext).all(1) & torch.isfinite(y_ext)
    print(f"{label}: {len(prob.y)} GMTKN55 reactions, p={len(keys)}; external molecules {int(ok.sum())}")
    print(f"{'lambda':>8s} {'in-sample':>10s} {'CV random':>10s} {'CV subset out':>14s} {'external MAE':>13s} {'external mean':>14s} {'|theta| scaled':>15s}")
    for lam in (float(x) for x in args.lams.split(",")):
        r = fl.fit_and_validate(prob, lam)
        theta = torch.tensor(r["theta"], dtype=torch.float64, device=fl.DEV)
        e = (a_ext @ theta - y_ext)[ok]
        scale = torch.sqrt((prob.w[:, None] * prob.A * prob.A).mean(0))
        print(f"{lam:8.0e} {r['in_sample']:10.3f} {r['cv_random']:10.3f} {r['cv_leave_subset_out']:14.3f} {float(e.abs().mean()):13.2f} "
              f"{float(e.mean()):+14.2f} {float((theta * scale).norm()):15.2f}")


if __name__ == "__main__":
    main()
