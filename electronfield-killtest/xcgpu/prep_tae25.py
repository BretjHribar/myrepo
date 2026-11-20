"""Prepare an external training/validation set from the pinned public MSR-ACC/TAE25 file.

Writes BULK/<name>/structures.json (geometry, charge, spin only: what the runners may see) and
BULK/<name>/labels.json (W1-F12 total atomization energies, read only by the fitting code).
The sample is the first N molecules in SHA-1 order, i.e. a fixed pseudo-random draw that does not
look at energies, composition or size. Bookkeeping only; no numerics.
"""
import argparse
import json

from .common import BULK, RESEARCH, build_mol, pin_one_core

ATOM_UNPAIRED = {"H": 1, "Li": 1, "Be": 0, "B": 1, "C": 2, "N": 3, "O": 2, "F": 1,
                 "Na": 1, "Mg": 0, "Al": 1, "Si": 2, "P": 3, "S": 2, "Cl": 1}


def main():
    pin_one_core()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--name", default="tae25")
    args = ap.parse_args()
    import pyarrow.parquet as pq
    cols = ["sha1", "name", "symbols", "geometry", "molecular_charge", "molecular_multiplicity", "tae@w1-f12",
            "atom-count:non-h", "tae:frac[(T)]@w1-f12"]
    table = pq.read_table(RESEARCH / "data/msr-acc-tae25/msr-acc-tae25-all.parquet", columns=cols).to_pylist()
    table.sort(key=lambda r: r["sha1"])
    picked = table[:args.n]
    rows, labels = [], {}
    for sym, unpaired in ATOM_UNPAIRED.items():
        atoms = [[sym, 0.0, 0.0, 0.0]]
        mol = build_mol([(sym, (0.0, 0.0, 0.0))], 0, unpaired, unit="Bohr")
        rows.append({"id": f"atom/{sym}", "subset": "atom", "charge": 0, "unpaired_electrons": unpaired,
                     "atoms": atoms, "unit": "Bohr", "nao": int(mol.nao), "formula": sym})
    for r in picked:
        if r["molecular_charge"] != 0 or r["molecular_multiplicity"] != 1:
            raise ValueError("unexpected charge/multiplicity in the public file")
        syms = list(r["symbols"])
        geom = [list(map(float, xyz)) for xyz in r["geometry"]]
        if any(s not in ATOM_UNPAIRED for s in syms):
            raise ValueError(f"unexpected element in {r['sha1']}")
        atoms = [[s, *xyz] for s, xyz in zip(syms, geom)]
        mol = build_mol([(a[0], tuple(a[1:])) for a in atoms], 0, 0, unit="Bohr")
        sid = "mol/" + r["sha1"]
        rows.append({"id": sid, "subset": "tae25", "charge": 0, "unpaired_electrons": 0, "atoms": atoms,
                     "unit": "Bohr", "nao": int(mol.nao), "formula": r["name"]})
        counts = {}
        for s in syms:
            counts[s] = counts.get(s, 0) + 1
        labels[sid] = {"tae_hartree": float(r["tae@w1-f12"]), "atoms": counts, "formula": r["name"],
                       "non_h": int(r["atom-count:non-h"]), "frac_t": r["tae:frac[(T)]@w1-f12"]}
    out = BULK / args.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "structures.json").write_text(json.dumps(rows))
    (out / "labels.json").write_text(json.dumps(labels))
    naos = sorted(r["nao"] for r in rows[len(ATOM_UNPAIRED):])
    print(json.dumps({"molecules": len(picked), "atoms": len(ATOM_UNPAIRED), "nao_median": naos[len(naos) // 2],
                      "nao_max": naos[-1], "distinct_formulas": len({r["name"] for r in picked}),
                      "written": str(out)}))


if __name__ == "__main__":
    main()
