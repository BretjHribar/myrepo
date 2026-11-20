"""Paired-seed summary of the rerun outputs in ./out. Usage: python3 analyze.py"""
import json, glob
from pathlib import Path
import numpy as np
from scipy import stats

K = 627.509474
OUT = Path(__file__).resolve().parent / "out"

def load(prefix):
    d = {}
    for f in sorted(glob.glob(str(OUT / f"{prefix}_*.json"))):
        r = json.load(open(f)); d[r["seed"]] = r["log"]
    return d

def table(tagE, tagF, label, step="1000"):
    E, F = load(tagE), load(tagF)
    seeds = sorted(set(E) & set(F))
    print(f"\n== {label}: seeds={len(seeds)} step={step}")
    for split in ("train", "test_id", "test_ood"):
        for m in ("e_mae", "dens", "v_rmse"):
            a = np.array([E[s][step][split][m] for s in seeds]); b = np.array([F[s][step][split][m] for s in seeds])
            d = b - a
            p = stats.ttest_rel(b, a).pvalue if len(seeds) > 2 else float("nan")
            extra = f" ({a.mean()*K:.2f} -> {b.mean()*K:.2f} kcal)" if m == "e_mae" else ""
            print(f"{split:8s} {m:7s} E={a.mean():.5f} F={b.mean():.5f} change={100*(b.mean()/a.mean()-1):+6.1f}% "
                  f"wins={int((d<0).sum())}/{len(seeds)} p={p:.3f} ratio_by_seed={np.round(b/a,2).tolist()}{extra}")
    unE = sum(E[s][step][sp]["unconv"] for s in seeds for sp in E[s][step]); unF = sum(F[s][step][sp]["unconv"] for s in seeds for sp in F[s][step])
    print("unconverged SCF cases: E", unE, "F", unF)

if __name__ == "__main__":
    table("a_E", "a_F5", "28 training systems, converged, field weight 5")
    table("c_E", "c_F5", "4 training systems, converged, field weight 5")
    table("a_E", "b_F02", "28 training systems, converged, published field weight 0.2")
    for step in ("130", "500"):
        table("a_E", "a_F5", "28 training systems, field weight 5", step)
