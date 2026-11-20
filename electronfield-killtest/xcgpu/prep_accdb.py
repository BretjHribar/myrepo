"""Prepare the public ACCDB collection as a training set that excludes the GMTKN55 test benchmark.

Reads the reaction lists (DatasetEval_kcal.csv: name, coefficient, geometry, ..., reference) and the
xyz geometries (line 2: charge and multiplicity). Writes, under BULK/accdb/:
  structures.json  unique geometries to compute, cheapest first (geometry, charge, spin only)
  reactions.json   every reaction with its reference energy and an overlap flag
  aliases.json     geometry name -> structure id (duplicates inside ACCDB and copies of GMTKN55 structures)
A reaction is flagged as overlapping the benchmark when its subset shares a name with a GMTKN55
subset, when ACCDB files it under its GMTKN55 database, or when every one of its structures is a
(near-)copy of a GMTKN55 structure. Overlapping reactions are never used for training.
Bookkeeping only; no numerics beyond distance fingerprints for de-duplication.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np

from .common import BASIS, BULK, ECP_ELEMENTS, GMTKN_DATA, build_mol, gmtkn_structures, pin_one_core, read_xyz

SRC = BULK / "external" / "ACCDB"


def fingerprint(symbols, coords, charge, mult):
    """Geometry identity up to rotation, translation and atom order, at 0.01 Angstrom resolution."""
    x = np.asarray(coords, dtype=float)
    d = np.sqrt(((x[:, None, :] - x[None, :, :]) ** 2).sum(-1))
    iu = np.triu_indices(len(x), 1)
    dist = np.sort(np.round(d[iu], 2))
    key = ",".join(sorted(symbols)) + f"|{charge}|{mult}|" + ",".join(f"{v:.2f}" for v in dist)
    return hashlib.sha1(key.encode()).hexdigest()


def read_accdb_xyz(path):
    """xyz with 'charge multiplicity' on line 2. Tolerates commas, atomic numbers as element labels,
    and a header atom count larger than the atoms actually listed (seen for a few single atoms)."""
    from pyscf.data import elements
    lines = [l for l in path.read_text().splitlines()]
    charge, mult = (int(float(v)) for v in lines[1].replace(",", " ").split()[:2])
    syms, xyz = [], []
    for line in lines[2:]:
        p = line.replace(",", " ").split()
        if len(p) < 4:
            continue
        tok = p[0]
        if tok.isdigit():
            s = elements.ELEMENTS[int(tok)]
        else:
            s = re.sub(r"[^A-Za-z]", "", tok)
            s = s[0].upper() + s[1:].lower()
        syms.append(s)
        xyz.append([float(v) for v in p[1:4]])
    if not syms:
        raise ValueError("no atoms")
    return syms, xyz, charge, mult


def norm(name):
    return re.sub(r"[^a-z0-9]", "", name.lower())


def main():
    pin_one_core()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--databases", default="MGCDB84,Minnesota,W4-17,MetalsEE")
    args = ap.parse_args()
    out = BULK / "accdb"
    out.mkdir(exist_ok=True)

    # basis size per element, from one single-atom molecule each
    nao_el = {}

    def nao_of(syms):
        total = 0
        for s in syms:
            if s not in nao_el:
                nao_el[s] = int(build_mol([(s, (0.0, 0.0, 0.0))], 0, 0 if build_parity(s) else 1).nao)
            total += nao_el[s]
        return total

    def build_parity(s):
        from pyscf.data import elements
        z = elements.charge(s)
        core = {True: 0}.get(False, 0)
        if s in ECP_ELEMENTS:
            core = 28 if z <= 54 else (46 if z <= 56 else 60)
            if 57 <= z <= 71:
                core = 28
        return (z - core) % 2 == 0

    # benchmark fingerprints
    g_rows, _ = gmtkn_structures()
    g_print = {}
    for r in g_rows:
        atoms = read_xyz(GMTKN_DATA / r["coordinate_path"])
        fp = fingerprint([a[0] for a in atoms], [a[1] for a in atoms], r["charge"], r["unpaired_electrons"] + 1)
        g_print.setdefault(fp, r["id"])
    g_subsets = {norm(r["subset"]) for r in g_rows} | {"bh76rc"}
    g_natoms = {r["id"]: r["atom_count"] for r in g_rows}

    reactions, problems = [], []
    geoms_needed = {}
    for db in args.databases.split(","):
        for csv in sorted((SRC / "Databases" / db).rglob("DatasetEval*_kcal.csv")):
            if "-RE" in csv.name:      # derived all-pairs reaction-energy sets: skipped
                continue
            for line in csv.read_text().splitlines():
                parts = [p.strip() for p in line.split(",") if p.strip() != ""]
                if len(parts) < 4:
                    continue
                name, body, ref = parts[0], parts[1:-1], parts[-1]
                try:
                    coefs = [float(body[i]) for i in range(0, len(body), 2)]
                    geoms = [body[i] for i in range(1, len(body), 2)]
                    ref = float(ref)
                except (ValueError, IndexError):
                    problems.append((db, name, "unparsed reaction line"))
                    continue
                subset = name.rsplit("_", 1)[0]
                reactions.append({"id": f"{db}/{name}", "database": db, "subset": subset, "coefficients": coefs,
                                  "geometries": geoms, "reference_kcal": ref})
                for g in geoms:
                    geoms_needed[g] = None

    # geometries -> unique structures
    structures, aliases, by_print = {}, {}, {}
    index = {norm(f.stem): f for f in (SRC / "Geometries").glob("*.xyz")}
    for g in sorted(geoms_needed):
        path = SRC / "Geometries" / (g + ".xyz")
        if not path.exists():
            path = index.get(norm(g))          # the lists and the file names differ in punctuation
        if path is None or not path.exists():
            problems.append(("geometry", g, "file missing"))
            continue
        try:
            syms, xyz, charge, mult = read_accdb_xyz(path)
            fp = fingerprint(syms, xyz, charge, mult)
        except Exception as exc:
            problems.append(("geometry", g, repr(exc)[:80]))
            continue
        if fp in g_print:
            aliases[g] = {"dataset": "gmtkn55", "id": g_print[fp]}
            continue
        if fp in by_print:
            aliases[g] = {"dataset": "accdb", "id": by_print[fp]}
            continue
        sid = "accdb/" + g
        by_print[fp] = sid
        aliases[g] = {"dataset": "accdb", "id": sid}
        try:
            nao = nao_of(syms)
        except Exception as exc:
            problems.append(("geometry", g, "no basis: " + repr(exc)[:60]))
            del by_print[fp]
            aliases.pop(g)
            continue
        structures[sid] = {"id": sid, "subset": "accdb", "charge": charge, "unpaired_electrons": mult - 1,
                           "atoms": [[s, *c] for s, c in zip(syms, xyz)], "unit": "Angstrom", "nao": nao,
                           "elements": sorted(set(syms))}

    # overlap flags and usability
    for r in reactions:
        al = [aliases.get(g) for g in r["geometries"]]
        r["structures"] = al
        r["complete"] = all(a is not None for a in al)
        all_copy = r["complete"] and all(a["dataset"] == "gmtkn55" for a in al)
        same_name = norm(r["subset"]) in g_subsets or norm(re.sub(r"\d+$", "", r["subset"])) in {re.sub(r"\d+$", "", s) for s in g_subsets}
        r["overlaps_benchmark"] = bool(all_copy or same_name or r["database"] == "GMTKN")
        r["overlap_reason"] = "all structures are benchmark copies" if all_copy else ("subset name matches a benchmark subset" if same_name else None)
        r["shares_benchmark_structure"] = bool(r["complete"] and any(a["dataset"] == "gmtkn55" and g_natoms.get(a["id"], 0) > 4 for a in al))
    # keep only structures needed by usable, non-overlapping reactions
    needed = set()
    for r in reactions:
        if r["complete"] and not r["overlaps_benchmark"]:
            needed.update(a["id"] for a in r["structures"] if a["dataset"] == "accdb")
    rows = sorted((s for s in structures.values() if s["id"] in needed), key=lambda s: (s["nao"], s["id"]))
    (out / "structures.json").write_text(json.dumps(rows))
    (out / "reactions.json").write_text(json.dumps(reactions))
    (out / "aliases.json").write_text(json.dumps(aliases))
    (out / "problems.json").write_text(json.dumps(problems, indent=1))

    usable = [r for r in reactions if r["complete"] and not r["overlaps_benchmark"]]
    by_subset = {}
    for r in usable:
        by_subset.setdefault(r["database"] + "/" + r["subset"], []).append(r)
    naos = [s["nao"] for s in rows]
    summary = {
        "reactions_total": len(reactions), "reactions_overlapping_benchmark": sum(r["overlaps_benchmark"] for r in reactions),
        "reactions_incomplete": sum(not r["complete"] for r in reactions), "reactions_usable_for_training": len(usable),
        "reactions_usable_strict_no_shared_larger_structure": sum(not r["shares_benchmark_structure"] for r in usable),
        "subsets_usable": len(by_subset), "structures_to_compute": len(rows),
        "structures_reused_from_benchmark": sum(1 for a in aliases.values() if a["dataset"] == "gmtkn55"),
        "problems": len(problems),
        "nao_bins": {f"{lo}-{hi}": sum(lo <= n < hi for n in naos) for lo, hi in ((0, 200), (200, 400), (400, 600), (600, 1000), (1000, 1500), (1500, 10000))},
        "elements": sorted({e for s in rows for e in s["elements"]}),
    }
    (out / "summary.json").write_text(json.dumps({"summary": summary, "subset_counts": {k: len(v) for k, v in sorted(by_subset.items())}}, indent=1))
    print(json.dumps(summary, indent=1))
    print("largest usable subsets:", sorted(((len(v), k) for k, v in by_subset.items()), reverse=True)[:25])
    print("first problems:", problems[:8])


if __name__ == "__main__":
    main()
