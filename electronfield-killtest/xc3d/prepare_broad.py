"""Create our own frozen electronic inputs for public atomization labels.

This computes input densities, not frontier-functional replication. Data use
is pinned by source.json. Initial TZVP/level-1 inputs are a research screen,
not the challenge's QZVP evaluation protocol.
"""
from .runtime import configure
ROOT = configure()

import argparse
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import pyarrow.parquet as pq
from pyscf import dft, gto, lib

SPINS = {"H": 1, "He": 0, "Li": 1, "Be": 0, "B": 1, "C": 2, "N": 3,
         "O": 2, "F": 1, "Ne": 0, "Na": 1, "Mg": 0, "Al": 1, "Si": 2,
         "P": 3, "S": 2, "Cl": 1, "Ar": 0}
COLUMNS = ["sha1", "symbols", "geometry", "molecular_charge", "molecular_multiplicity",
           "atom-count:all", "atom-count:non-h", "tae@w1-f12"]


def choose(source, limit, max_heavy=3, max_atoms=10):
    rows = pq.read_table(source, columns=COLUMNS).to_pylist()
    buckets = defaultdict(list)
    for row in rows:
        if row["atom-count:non-h"] > max_heavy or row["atom-count:all"] > max_atoms:
            continue
        if row["tae@w1-f12"] is None or not np.isfinite(row["tae@w1-f12"]):
            continue
        composition = tuple(sorted(Counter(row["symbols"]).items()))
        family = json.dumps(composition, separators=(",", ":"))
        split_number = int(hashlib.sha256(family.encode()).hexdigest()[:8], 16) % 10
        row["split"] = "test" if split_number == 0 else "val" if split_number == 1 else "train"
        row["family"] = family
        key = (tuple(sorted(set(row["symbols"]) - {"H"})), row["atom-count:non-h"])
        buckets[key].append(row)
    # Hash-order buckets so a prefix cannot select only alphabetically early elements.
    ordered = sorted(buckets.items(), key=lambda item: hashlib.sha256(repr(item[0]).encode()).hexdigest())
    queues = [deque(sorted(bucket, key=lambda row: row["sha1"])) for _, bucket in ordered]
    selected = []
    while queues and len(selected) < limit:
        for queue in queues:
            if queue and len(selected) < limit:
                selected.append(queue.popleft())
        queues = [queue for queue in queues if queue]
    return selected


def solve(atom, unit, spin, charge, basis, grid_level, max_seconds):
    started = time.perf_counter()
    mol = gto.M(atom=atom, unit=unit, spin=spin, charge=charge, basis=basis,
                verbose=0, max_memory=1800)
    mf = (dft.UKS(mol) if spin else dft.RKS(mol)).density_fit()
    mf.xc = "pbe"
    mf.grids.level = grid_level
    mf.conv_tol = 1e-9
    mf.max_cycle = 80
    mf.damp = .2

    def callback(env):
        if time.perf_counter() - started > max_seconds:
            raise TimeoutError(f"Input-density calculation exceeded {max_seconds}s")

    mf.callback = callback
    energy = mf.kernel()
    if not mf.converged:
        raise RuntimeError("Input density did not converge within its bounded attempt")
    dm = np.asarray(mf.make_rdm1())
    if dm.ndim == 2:
        dm = np.stack((dm / 2, dm / 2))
    coords, weights = mf.grids.coords, mf.grids.weights
    ao = dft.numint.eval_ao(mol, coords, deriv=1)
    rho = np.stack([dft.numint.eval_rho(mol, ao, d, xctype="GGA", hermi=1) for d in dm])
    tau = np.stack([
        .5 * np.einsum("ij,dgi,dgj->g", d, ao[1:4], ao[1:4], optimize=True) for d in dm
    ])
    return {"rho": rho.astype(np.float32), "tau": tau.astype(np.float32),
            "coords": coords.astype(np.float32), "weights": weights,
            "landmarks": mol.atom_coords().astype(np.float32),
            "energy_pbe": float(energy), "seconds": time.perf_counter() - started,
            "electrons_integrated": float(np.dot(weights, rho[:, 0].sum(0))),
            "nelectron": mol.nelectron, "scf_cycles": mf.cycles}


def save(path, metadata, fields):
    metadata = {**metadata, **{key: value for key, value in fields.items() if np.isscalar(value)}}
    arrays = {key: value for key, value in fields.items() if not np.isscalar(value)}
    np.savez_compressed(path, metadata=json.dumps(metadata), **arrays)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=192)
    parser.add_argument("--max-heavy", type=int, default=3)
    parser.add_argument("--max-atoms", type=int, default=10)
    parser.add_argument("--basis", default="def2-tzvp")
    parser.add_argument("--grid-level", type=int, default=1)
    parser.add_argument("--molecule-seconds", type=float, default=30)
    parser.add_argument("--output", type=Path, default=ROOT / "data/broad_tzvp_192")
    args = parser.parse_args()
    lib.num_threads(1)
    source_dir = ROOT / "data/msr-acc-tae25"
    chosen = choose(source_dir / "msr-acc-tae25-all.parquet", args.limit, args.max_heavy, args.max_atoms)
    args.output.mkdir(parents=True, exist_ok=True)
    source_record = json.loads((source_dir / "source.json").read_text())
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "source": source_record,
                "basis": args.basis, "grid_level": args.grid_level,
                "geometry_unit": "Bohr (QCSchema)", "energy_unit": "Hartree",
                "geometry_unit_source": "https://github.com/MolSSI/QCSchema/blob/master/qcschema/data/v2/qc_schema_input.schema",
                "purpose": "Our input features and public energy labels; not a frontier-method reproduction",
                "split_policy": "Entire molecular compositions share a fixed SHA256-derived split",
                "selected": chosen, "atoms": {}, "molecules": [], "failures": []}
    manifest_path = args.output / "manifest.json"

    def record():
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    record()
    elements = sorted({symbol for row in chosen for symbol in row["symbols"]})
    for symbol in elements:
        path = args.output / f"atom_{symbol}.npz"
        if path.exists():
            with np.load(path) as cache:
                info = json.loads(str(cache["metadata"]))
        else:
            fields = solve(f"{symbol} 0 0 0", "Bohr", SPINS[symbol], 0, args.basis, args.grid_level, args.molecule_seconds)
            info = save(path, {"id": f"atom_{symbol}", "symbols": [symbol], "spin": SPINS[symbol],
                               "basis": args.basis, "grid_level": args.grid_level}, fields)
        manifest["atoms"][symbol] = info
        record()
        print(json.dumps({"atom": symbol, "seconds": round(info["seconds"], 2)}), flush=True)
    for row in chosen:
        path = args.output / f"mol_{row['sha1']}.npz"
        try:
            if path.exists():
                with np.load(path) as cache:
                    info = json.loads(str(cache["metadata"]))
            else:
                atom = [(symbol, xyz) for symbol, xyz in zip(row["symbols"], row["geometry"])]
                fields = solve(atom, "Bohr", int(row["molecular_multiplicity"]) - 1,
                               int(row["molecular_charge"]), args.basis, args.grid_level, args.molecule_seconds)
                atom_energy = sum(manifest["atoms"][symbol]["energy_pbe"] for symbol in row["symbols"])
                info = save(path, {**row, "id": row["sha1"], "basis": args.basis,
                                   "grid_level": args.grid_level, "tae_pbe": atom_energy - fields["energy_pbe"]}, fields)
            manifest["molecules"].append(info)
            print(json.dumps({"molecule": row["sha1"], "formula": row["family"], "split": row["split"],
                              "seconds": round(info["seconds"], 2), "done": len(manifest["molecules"])}), flush=True)
        except Exception as exc:
            manifest["failures"].append({"id": row["sha1"], "error": f"{type(exc).__name__}: {exc}"})
            print(json.dumps(manifest["failures"][-1]), flush=True)
        record()
    print(json.dumps({"completed": len(manifest["molecules"]), "failures": len(manifest["failures"]),
                      "split_counts": dict(Counter(row["split"] for row in manifest["molecules"]))}), flush=True)


if __name__ == "__main__":
    main()
