"""Diagnostic: is the remaining error a conflict between reaction types or missing information?

For each sufficiently large subset, fit the feature set on that subset alone (cross-validated inside
the subset) and compare with the error the global fit leaves on it. A subset whose own fit is much
better than the global fit points to a model that cannot adapt across chemistry; a subset whose own
fit is no better points to missing information or noise in the ingredients.
"""
import argparse
import json

from .common import pin_one_core

pin_one_core()
import torch
from . import fitlib as fl


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--blocks", default="poly_tot,poly_cross,x_semi,c_semi,pt2,singles")
    ap.add_argument("--base", default="exx_rpa")
    ap.add_argument("--lam", type=float, default=1e-5)
    ap.add_argument("--min-n", type=int, default=20)
    args = ap.parse_args()
    rows, _ = fl.structure_table()
    keys = [k for b in args.blocks.split(",") for k in fl.FEATURE_SETS[b]]
    prob = fl.Problem(rows, keys, args.base)
    glob = fl.fit_and_validate(prob, args.lam)
    print(f"global: n={len(prob.y)} p={len(keys)} in-sample {glob['in_sample']:.3f} CV-random {glob['cv_random']:.3f} CV-LSO {glob['cv_leave_subset_out']:.3f}")
    print(f"{'subset':10s} {'n':>4s} {'p':>3s} | global fit: {'in':>6s} {'CV':>6s} {'LSO':>6s} | own fit: {'in':>6s} {'5-fold CV':>9s}   (MAD kcal/mol)")
    for i, name in enumerate(prob.subset_names):
        m = prob.subset_index == i
        n = int(m.sum())
        if n < args.min_n:
            continue
        A, y, w = prob.A[m], prob.y[m], prob.w[m]
        own = {}
        for lam in (1e-5, 1e-3, 1e-1):
            g = torch.Generator(device="cpu").manual_seed(0)
            fold = (torch.randperm(n, generator=g) % 5).to(fl.DEV)
            masks = torch.stack([torch.ones(n, dtype=torch.bool, device=fl.DEV)] + [fold != k for k in range(5)])
            th = fl.solve(A, y, w, lam, masks)
            err = th @ A.T - y[None, :]
            cv = torch.zeros_like(y)
            for k in range(5):
                held = fold == k
                cv[held] = err[k + 1][held]
            own[lam] = (float(err[0].abs().mean()), float(cv.abs().mean()))
        best = min(own.values(), key=lambda v: v[1])
        print(f"{name:10s} {n:4d} {len(keys):3d} |             {float(glob['in_sample_err'][m].abs().mean()):6.2f} "
              f"{float(glob['cv_random_err'][m].abs().mean()):6.2f} {float(glob['cv_lso_err'][m].abs().mean()):6.2f} |"
              f"          {best[0]:6.2f} {best[1]:9.2f}")


if __name__ == "__main__":
    main()
