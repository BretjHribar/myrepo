"""Per-reaction errors of an unfitted base model for one subset (GPU arithmetic via fitlib)."""
import sys

from .common import pin_one_core

pin_one_core()
from . import fitlib as fl


def main():
    subset = sys.argv[1]
    rows, _ = fl.structure_table()
    probs = {b: fl.Problem(rows, [], b) for b in ("pbe", "exx", "exx_rpa")}
    p = probs["pbe"]
    s = p.suite
    idx = [i for i, n in enumerate(p.subset) if n == subset]
    lookup = {r["id"]: r for r in s.reactions}
    print(f"{'reaction':14s} {'ref':>8s} {'PBE err':>8s} {'EXX err':>8s} {'EXX+RPA err':>11s}  structures")
    for i in idx:
        rid = p.rid[i]
        row = lookup[rid]
        print(f"{rid:14s} {float(p.ref[i]):8.2f} {float(-probs['pbe'].y[i]):8.2f} {float(-probs['exx'].y[i]):8.2f} "
              f"{float(-probs['exx_rpa'].y[i]):11.2f}  " + " ".join(f"{c:+d}*{x.split('/')[1]}" for c, x in zip(row['coefficients'], row['structure_ids'])))


if __name__ == "__main__":
    main()
