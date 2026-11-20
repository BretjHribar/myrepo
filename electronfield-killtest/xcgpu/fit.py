"""Run the fitting ladder on whatever structures are complete and print a labelled comparison.

Every row carries its run card: what drives the energy (base and feature blocks), how many of the
1,505 reactions it covers, and whether the number is in-sample or cross-validated.
"""
import argparse
import json
import time
from pathlib import Path

from .common import BULK, pin_one_core

pin_one_core()
import torch
from . import fitlib as fl

LADDER = (
    ("PBE (input baseline, nothing fitted)", "pbe", []),
    ("EXX only (no correlation, nothing fitted)", "exx", []),
    ("EXX + RPA @PBE (nothing fitted)", "exx_rpa", []),
    ("A: RPA rescaled", "exx_rpa", ["rpa_scale"]),
    ("C: RPA + fixed semilocal components + shifted PT2 + singles", "exx_rpa", ["rpa_scale", "x_semi", "c_semi", "pt2", "singles"]),
    ("H: fixed semilocal + PT2 + singles, no spectrum (double-hybrid-like control)", "exx", ["x_semi", "c_semi", "pt2", "singles"]),
    ("K: fixed semilocal, PT2, singles, 5+5 spectral", "exx_rpa", ["poly_tot", "poly_cross", "x_semi", "c_semi", "pt2", "singles"]),
    ("L: K with frequency-band spectral terms instead", "exx_rpa", ["band_tot", "band_cross", "x_semi", "c_semi", "pt2", "singles"]),
    ("M: regime-resolved semilocal + PT2 + singles + 5+5 spectral", "exx_rpa", ["poly_tot", "poly_cross", "gx", "gcss", "gcos", "pt2", "singles"]),
    ("N: M + fixed semilocal components", "exx_rpa", ["poly_tot", "poly_cross", "gx", "gcss", "gcos", "x_semi", "c_semi", "pt2", "singles"]),
    ("O: N with frequency-band spectral terms", "exx_rpa", ["band_tot", "band_cross", "gx", "gcss", "gcos", "x_semi", "c_semi", "pt2", "singles"]),
    ("P: regime-resolved semilocal only on EXX (no PT2, no spectrum)", "exx", ["gx", "gcss", "gcos"]),
)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scf-tag", default="pbe_qzvp_g3")
    ap.add_argument("--lams", default="1e-5,1e-4,1e-3,1e-2,1e-1")
    ap.add_argument("--output", type=Path)
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--only", default="", help="comma-separated label prefixes to run (default: whole ladder)")
    args = ap.parse_args()
    t0 = time.time()
    rows, problems = fl.structure_table(args.scf_tag, refresh=args.refresh)
    print(f"structures with all passes complete: {len(rows)}; with problems: {len(problems)}  (table {time.time()-t0:.0f}s)")
    for sid, why in list(problems.items())[:20]:
        print("   problem:", sid, why)
    lams = [float(x) for x in args.lams.split(",")]
    report = {"scf_tag": args.scf_tag, "n_structures": len(rows), "problems": problems, "models": []}
    first = True
    only = [x for x in args.only.split(",") if x]
    for label, base, blocks in LADDER:
        if only and not any(label.startswith(x) for x in only) and blocks:
            continue
        keys = [k for b in blocks for k in fl.FEATURE_SETS[b]]
        prob = fl.Problem(rows, keys, base)
        if first:
            print(f"reactions usable: {len(prob.y)}/{prob.n_total}; complete subsets: {len(prob.complete_subsets)}/55")
            print("complete subsets:", " ".join(prob.complete_subsets))
            print("NOTE: every number below is a weighted mean absolute error over the usable reactions only. "
                  "It equals WTMAD-2 only at 1505/1505.")
            first = False
        if not keys:
            err = -prob.y
            val = prob.wtmad2(err)
            print(f"{label:70s} n={len(prob.y):4d} p= 0  error {val:7.3f}  (no parameters)")
            report["models"].append({"label": label, "base": base, "blocks": blocks, "n": len(prob.y), "p": 0, "error": val,
                                     "by_subset": prob.by_subset(err)})
            continue
        best = None
        for lam in lams:
            r = fl.fit_and_validate(prob, lam)
            if best is None or r["cv_random"] < best["cv_random"]:
                best = r
        print(f"{label:70s} n={best['n_reactions']:4d} p={best['n_features']:2d}  in-sample {best['in_sample']:6.3f}  "
              f"CV-random {best['cv_random']:6.3f} (+-{best['cv_random_spread']:.3f})  CV-leave-subset-out {best['cv_leave_subset_out']:6.3f}  "
              f"lam {best['lambda']:g}")
        report["models"].append({"label": label, "base": base, "blocks": blocks, "n": best["n_reactions"],
                                 "p": best["n_features"], "lambda": best["lambda"], "in_sample": best["in_sample"],
                                 "cv_random": best["cv_random"], "cv_random_spread": best["cv_random_spread"],
                                 "cv_leave_subset_out": best["cv_leave_subset_out"], "theta": dict(zip(map(str, keys), best["theta"])),
                                 "by_subset_in_sample": prob.by_subset(best["in_sample_err"]),
                                 "by_subset_cv_random": prob.by_subset(best["cv_random_err"]),
                                 "by_subset_cv_lso": prob.by_subset(best["cv_lso_err"]),
                                 "reactions": [{"id": i, "w": float(w), "ref": float(r), "in": float(a), "cv": float(b), "lso": float(c)}
                                               for i, w, r, a, b, c in zip(prob.rid, prob.w.tolist(), prob.ref.tolist(),
                                                                           best["in_sample_err"].tolist(), best["cv_random_err"].tolist(),
                                                                           best["cv_lso_err"].tolist())]})
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
