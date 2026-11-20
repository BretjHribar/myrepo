"""Prepare formula-balanced reaction contrasts from public reference structures.

Uses the highest-TAE available isomer of each requested formula. Actual selected
structures are preserved in the manifest; reaction labels describe stoichiometry,
not a claimed transition path. This is a development panel, with fitting overlap
explicitly recorded, not a new blind benchmark.
"""
from .runtime import configure
ROOT = configure()

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import numpy as np
import pyarrow.parquet as pq
from pyscf import lib
from .prepare_broad import COLUMNS, SPINS, save, solve

REACTIONS = [
    ("Al_hydride_hydrolysis", {"AlH3": -1, "H2O": -1, "AlH3O": 1, "H2": 1}),
    ("B_hydride_hydrolysis", {"BH3": -1, "H2O": -1, "BH3O": 1, "H2": 1}),
    ("Si_hydride_hydrolysis", {"SiH4": -1, "H2O": -1, "SiH4O": 1, "H2": 1}),
    ("Al_hydride_amination", {"AlH3": -1, "NH3": -1, "AlH4N": 1, "H2": 1}),
    ("B_hydride_amination", {"BH3": -1, "NH3": -1, "BH4N": 1, "H2": 1}),
    ("Si_hydride_amination", {"SiH4": -1, "NH3": -1, "SiH5N": 1, "H2": 1}),
    ("Al_hydroxide_condensation", {"AlH3O": -2, "Al2H4O": 1, "H2O": 1}),
    ("B_hydroxide_condensation", {"BH3O": -2, "B2H4O": 1, "H2O": 1}),
    ("Si_hydroxide_condensation", {"SiH4O": -2, "Si2H6O": 1, "H2O": 1}),
    ("Al_dimer_dissociation", {"Al2H6": -1, "AlH3": 2}),
    ("B_dimer_dissociation", {"B2H6": -1, "BH3": 2}),
    ("methane_water_rearrangement", {"CH4": -1, "H2O": -1, "CH4O": 1, "H2": 1}),
    ("methane_ammonia_rearrangement", {"CH4": -1, "NH3": -1, "CH5N": 1, "H2": 1}),
    ("ethylene_hydrogenation", {"C2H4": -1, "H2": -1, "C2H6": 1}),
    ("acetylene_hydrogenation", {"C2H2": -1, "H2": -1, "C2H4": 1}),
    ("ammonia_formation", {"N2": -1, "H2": -3, "NH3": 2}),
    ("carbon_monoxide_hydrogenation", {"CO": -1, "H2": -3, "CH4": 1, "H2O": 1}),
    ("carbon_dioxide_hydrogenation", {"CO2": -1, "H2": -4, "CH4": 1, "H2O": 2}),
]


def composition(formula):
    return dict((element, int(count) if count else 1) for element, count in re.findall(r"([A-Z][a-z]?)(\d*)", formula))


def key(counts):
    return tuple(sorted(counts.items()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data/reaction_panel_tzvp")
    args = parser.parse_args()
    lib.num_threads(1)
    public = ROOT / "data/msr-acc-tae25"
    requested = {formula for _, reaction in REACTIONS for formula in reaction}
    wanted = {key(composition(formula)): formula for formula in requested}
    selected = {}
    for row in pq.read_table(public / "msr-acc-tae25-all.parquet", columns=COLUMNS).to_pylist():
        formula = wanted.get(key(Counter(row["symbols"])))
        if formula is None or row["tae@w1-f12"] is None:
            continue
        if formula not in selected or row["tae@w1-f12"] > selected[formula]["tae@w1-f12"]:
            selected[formula] = row
    broad_dir = ROOT / "data/broad_tzvp_192"
    broad_manifest = json.loads((broad_dir / "manifest.json").read_text())
    training_ids = {row["id"] for row in broad_manifest["molecules"] if row["split"] == "train"}
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(),
                "source": json.loads((public / "source.json").read_text()),
                "basis": "def2-tzvp", "grid_level": 1, "atoms": {}, "molecules": [],
                "reactions": [], "missing_formulas": sorted(requested - selected.keys()), "failures": [],
                "evaluation_status": "Development reaction contrasts; some constituent molecules may overlap fitting data"}

    def record():
        (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    for symbol in sorted({symbol for row in selected.values() for symbol in row["symbols"]}):
        source = broad_dir / f"atom_{symbol}.npz"
        with np.load(source) as data:
            metadata = json.loads(str(data["metadata"]))
            fields = {name: data[name] for name in data.files if name != "metadata"}
            np.savez_compressed(args.output / source.name, metadata=json.dumps(metadata), **fields)
        manifest["atoms"][symbol] = metadata
    for formula, row in sorted(selected.items()):
        path = args.output / f"mol_{row['sha1']}.npz"
        metadata = {**row, "id": row["sha1"], "formula_label": formula,
                    "family": json.dumps(sorted(Counter(row["symbols"]).items())),
                    "split": "test", "seen_in_fitting": row["sha1"] in training_ids,
                    "basis": "def2-tzvp", "grid_level": 1}
        try:
            cached = broad_dir / path.name
            if path.exists() or cached.exists():
                source = path if path.exists() else cached
                with np.load(source) as data:
                    previous = json.loads(str(data["metadata"]))
                    arrays = {name: data[name] for name in data.files if name != "metadata"}
                metadata = {**previous, **metadata}
                np.savez_compressed(path, metadata=json.dumps(metadata), **arrays)
            else:
                fields = solve(list(zip(row["symbols"], row["geometry"])), "Bohr",
                               int(row["molecular_multiplicity"]) - 1, int(row["molecular_charge"]),
                               "def2-tzvp", 1, 30)
                atom_energy = sum(manifest["atoms"][symbol]["energy_pbe"] for symbol in row["symbols"])
                metadata["tae_pbe"] = atom_energy - fields["energy_pbe"]
                metadata = save(path, metadata, fields)
            manifest["molecules"].append(metadata)
            print(json.dumps({"formula": formula, "seen_in_fitting": metadata["seen_in_fitting"]}), flush=True)
        except Exception as exc:
            manifest["failures"].append({"formula": formula, "error": f"{type(exc).__name__}: {exc}"})
        record()
    available = {row["formula_label"]: row for row in manifest["molecules"]}
    for name, stoichiometry in REACTIONS:
        if not all(formula in available for formula in stoichiometry):
            continue
        balance = Counter()
        for formula, coefficient in stoichiometry.items():
            for symbol, count in composition(formula).items():
                balance[symbol] += coefficient * count
        if any(balance.values()):
            raise RuntimeError(f"Unbalanced reaction definition: {name}")
        exact = -sum(coefficient * available[formula]["tae@w1-f12"] for formula, coefficient in stoichiometry.items())
        manifest["reactions"].append({"name": name, "stoichiometry": stoichiometry,
                                      "reference_hartree": exact,
                                      "constituents_seen_in_fitting": [formula for formula in stoichiometry if available[formula]["seen_in_fitting"]]})
    record()
    print(json.dumps({"molecules": len(available), "reactions": len(manifest["reactions"]),
                      "missing": manifest["missing_formulas"], "failures": manifest["failures"]}), flush=True)


if __name__ == "__main__":
    main()
