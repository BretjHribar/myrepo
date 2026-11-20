"""Byte-integrity and integer metadata audit; no scientific calculation."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[2]
FOLDER = ROOT / ".research/data/gmtkn55_official_v1"
EXPECTED = set(("W4-11 G21EA G21IP DIPCS10 PA26 SIE4x4 ALKBDE10 YBDE18 AL2X6 HEAVYSB11 NBPRC ALK8 RC21 G2RC BH76RC FH51 TAUT15 DC13 "
                "MB16-43 DARC RSE43 BSR36 CDIE20 ISO34 ISOL24 C60ISO PArel BH76 BHPERI BHDIV10 INV24 BHROT27 PX13 WCPT18 "
                "RG18 ADIM6 S22 S66 HEAVY28 WATER27 CARBHB12 PNICO23 HAL59 AHB21 CHB6 IL16 IDISP ICONF ACONF Amino20x4 PCONF21 MCONF SCONF UPU23 BUT14DIOL").split())


def main():
    output = ROOT / "electronfield-killtest/research/gmtkn55_data_audit_20261117.json"
    if output.exists(): raise RuntimeError("This pinned byte/metadata audit already has a report")
    manifest = json.loads((FOLDER / "manifest.json").read_text())
    failures = []
    if set(manifest["evaluation_sets"]) != EXPECTED or len(manifest["evaluation_sets"]) != 55:
        failures.append("Evaluation-set coverage differs from the complete original 55-set benchmark")
    files = {row["path"]: row for row in manifest["files"]}
    for name, row in files.items():
        relative = PurePosixPath(name)
        path = (FOLDER / "data" / Path(*relative.parts)).resolve()
        if relative.is_absolute() or ".." in relative.parts or not path.is_relative_to((FOLDER / "data").resolve()):
            failures.append(f"Unsafe manifest path: {name}"); continue
        parts = relative.parts
        allowed = len(parts) == 1 and name in {"LICENSE", "README.md", ".bib", ".list.dirs", ".res"}
        allowed |= len(parts) == 2 and parts[0] in EXPECTED and parts[1] in {".bib", ".res", ".resRC", ".average", ".numen"}
        allowed |= len(parts) == 3 and parts[0] in EXPECTED and parts[2] in {"coord", "struc.xyz", ".CHRG", ".UHF"}
        if not allowed: failures.append(f"Non-data file in selected dataset: {name}"); continue
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        git_hash = hashlib.sha1(b"blob "+str(len(data)).encode()+b"\0"+data).hexdigest()
        if len(data) != row["bytes"] or digest != row["sha256"] or git_hash != row["git_blob_sha1"]:
            failures.append(f"Pinned data hash differs: {name}")
    counts, charges, spin_counts = Counter(), Counter(), Counter()
    seen = set()
    for structure in manifest["structures"]:
        if structure["id"] in seen: failures.append(f"Duplicate structure ID: {structure['id']}")
        seen.add(structure["id"])
        if structure["coordinate_path"] not in files: failures.append(f"Unpinned coordinate: {structure['id']}")
        if structure["unknown_element_tokens"] or not structure["integer_sector_metadata_consistent"]:
            failures.append(f"Unresolved charge/spin metadata: {structure['id']}")
        counts[structure["subset"]] += 1
        charges[structure["charge"]] += 1
        spin_counts[structure["unpaired_electrons"]] += 1
    definition_map = {}
    for subset in sorted(EXPECTED):
        definition = "BH76/.resRC" if subset == "BH76RC" else f"{subset}/.res"
        if definition not in files or files.get(definition, {}).get("bytes", 0) == 0:
            failures.append(f"Missing reaction definition: {subset}")
        definition_map[subset] = definition
        geometry_subset = "BH76" if subset == "BH76RC" else subset
        if counts[geometry_subset] == 0: failures.append(f"Missing structure group: {subset}")
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "scope": "Source bytes, complete set coverage and integer geometry/charge/spin metadata only",
              "commit_sha": manifest["commit_sha"], "license": manifest["license"], "passed_data_audit": not failures,
              "failures": failures, "verified_file_count": len(files), "structure_count": len(seen), "evaluation_set_count": len(EXPECTED),
              "structure_counts_by_subset": dict(sorted(counts.items())), "reaction_definition_map": definition_map,
              "charge_counts": dict(sorted(charges.items())), "reported_unpaired_electron_counts": dict(sorted(spin_counts.items())),
              "minimum_physical_electron_count": min(row["physical_electrons"] for row in manifest["structures"]),
              "maximum_physical_electron_count": max(row["physical_electrons"] for row in manifest["structures"]),
              "maximum_atom_count": max(row["atom_count"] for row in manifest["structures"]),
              "coordinate_consistency_between_source_formats_validated": False,
              "reference_energy_values_parsed": False, "benchmark_weights_numerically_validated": False,
              "official_suite_model_scores_available": False, "benchmark_suitable_for_accuracy_claims_now": False,
              "wtmad2_measured": False, "goal_accuracy_achieved": False,
              "next_requirement": "On CUDA, verify coordinate-format agreement and evaluate complete frozen predictions and correctly versioned reaction weights before any accuracy claim; no filtered subset stands in for the full suite"}
    output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({key: report[key] for key in ("passed_data_audit", "verified_file_count", "structure_count", "evaluation_set_count",
        "charge_counts", "reported_unpaired_electron_counts", "minimum_physical_electron_count", "maximum_physical_electron_count", "maximum_atom_count", "failures", "wtmad2_measured")}), flush=True)
    if failures: raise RuntimeError("The pinned complete-suite data audit failed")


if __name__ == "__main__": main()
