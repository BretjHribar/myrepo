"""Freeze unseen composition contrasts without reading reference-energy columns.

Host work here is metadata selection and Parquet/JSON I/O, not a numerical
chemistry workload. No labels, electronic calculations, or model evaluations.
"""
from .runtime import configure
ROOT = configure(gpu_only_compatible=True)

from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq


def family(symbols):
    return json.dumps(sorted(Counter(symbols).items()), separators=(",", ":"))


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def main():
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    destination = ROOT / "data/locked_composition_confirmation.json"
    if destination.exists():
        raise RuntimeError("Confirmation panel is already locked; do not adaptively replace it")
    exclusions = set()
    for directory in ("broad_tzvp_192", "reaction_panel_tzvp"):
        manifest = json.loads((ROOT / "data" / directory / "manifest.json").read_text())
        for row in manifest.get("selected", [])+manifest.get("molecules", []):
            exclusions.add(family(row["symbols"]))
    exclusions.update((family(["H"]), family(["He"]), family(["H", "H"]),
                       family(["He", "H"]), family(["H"]*3), family(["H"]*4)))
    columns = ["sha1", "symbols", "geometry", "molecular_charge", "molecular_multiplicity",
               "atom-count:all", "atom-count:non-h"]
    public = ROOT / "data/msr-acc-tae25"
    grouped = defaultdict(list)
    for row in pq.read_table(public / "msr-acc-tae25-all.parquet", columns=columns, use_threads=False).to_pylist():
        if row["atom-count:non-h"] > 3 or row["atom-count:all"] > 10:
            continue
        composition = family(row["symbols"])
        if composition in exclusions:
            continue
        grouped[composition].append(row)
    categories = defaultdict(list)
    for composition, rows in grouped.items():
        unique = {row["sha1"]: row for row in rows}
        if len(unique) < 2:
            continue
        candidates = sorted(unique.values(), key=lambda row: digest("confirmation-20261116:"+row["sha1"]))
        element_group = tuple(sorted(set(candidates[0]["symbols"])-{"H"}))
        categories[element_group].append((composition, candidates[:2]))
    queues = [deque(sorted(rows, key=lambda item: digest(item[0])))
              for key, rows in sorted(categories.items(), key=lambda item: digest(repr(item[0])))]
    selected = []
    while queues and len(selected) < 16:
        for queue in queues:
            if queue and len(selected) < 16:
                composition, rows = queue.popleft()
                selected.append({"family": composition, "structures": rows,
                                 "contrast": {rows[0]["sha1"]: -1, rows[1]["sha1"]: 1}})
        queues = [queue for queue in queues if queue]
    report = {"created_utc": datetime.now(timezone.utc).isoformat(),
              "purpose": "Untouched out-of-sample same-composition energy contrasts, disjoint from every inspected composition",
              "families": selected, "excluded_compositions": sorted(exclusions),
              "reference_energy_columns_read": False, "selection_columns": columns,
              "geometry_units": "Bohr, inherited QCSchema", "source": json.loads((public / "source.json").read_text()),
              "selection_policy": "16 hash-ordered element-stratified compositions; two SHA-hash-selected structures each",
              "missing_label_policy": "Record missing labels or failed calculations; do not replace cases after observing predictions",
              "unblinding_policy": "Freeze a general-electron candidate and prediction protocol before reading reference labels",
              "limitations": "Neutral closed-shell public structures; not barriers, ions, radicals, or the hidden suite"}
    destination.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({"locked_families": len(selected), "structures": 2*len(selected),
                      "reference_energy_columns_read": False, "path": str(destination)}), flush=True)


if __name__ == "__main__":
    main()
