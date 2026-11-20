"""Per-set sanity table for the outside (ACCDB) reactions: unfitted PBE and exact-exchange + RPA errors.

A set whose plain-PBE error is absurd points at a data or state problem (units, multiplicity, wrong
electronic state), not at the fitted model.
"""
import json
import sys

from .common import BULK, pin_one_core

pin_one_core()
from . import fitlib as fl


def main():
    tables = {"gmtkn55": fl.structure_table()[0], "accdb": fl.structure_table(dataset="accdb")[0]}
    rx = json.loads((BULK / "accdb/reactions.json").read_text())
    sets = {}
    for r in rx:
        if not r["complete"] or r["overlaps_benchmark"]:
            continue
        ids = [(a["dataset"], a["id"]) for a in r["structures"]]
        if not all(i in tables[d] for d, i in ids):
            continue
        vals = {}
        for base in ("pbe", "exx_rpa"):
            vals[base] = sum(c * fl.base_energy(tables[d][i], base) for c, (d, i) in zip(r["coefficients"], ids)) * fl.KCAL
        sets.setdefault(r["database"] + "/" + r["subset"], []).append((r, vals))
    print(f"{'set':30s} {'n':>4s} {'mean |ref|':>10s} {'PBE MAE':>9s} {'EXX+RPA MAE':>12s} {'PBE max':>9s}   worst reaction (ref, PBE value)")
    for name, items in sorted(sets.items(), key=lambda kv: -max(abs(v["pbe"] - r["reference_kcal"]) for r, v in kv[1])):
        refs = [r["reference_kcal"] for r, _ in items]
        ep = [v["pbe"] - r["reference_kcal"] for r, v in items]
        er = [v["exx_rpa"] - r["reference_kcal"] for r, v in items]
        k = max(range(len(items)), key=lambda i: abs(ep[i]))
        print(f"{name:30s} {len(items):4d} {sum(map(abs, refs))/len(refs):10.2f} {sum(map(abs, ep))/len(ep):9.2f} {sum(map(abs, er))/len(er):12.2f} "
              f"{max(map(abs, ep)):9.2f}   {items[k][0]['id'].split('/')[-1]} ({refs[k]:.2f}, {items[k][1]['pbe']:.2f})")


if __name__ == "__main__":
    main()
