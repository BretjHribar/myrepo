"""Acquire pinned public benchmark DATA without method code or calculations.

Host-only orchestration, byte hashing and integer structure bookkeeping.
Coordinate and reference energy strings are never numerically evaluated.
Use the one-core resource guard. No numerical library is imported.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import threading
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
SHA = "8d485b37a1ca8837e395042671ca5ba4e0714691"
REPOSITORY = "grimme-lab/GMTKN55"
DESTINATION = ROOT / ".research/data/gmtkn55_official_v1"
MAXIMUM_TRANSFER_BYTES = 64_000_000
MAXIMUM_FILE_BYTES = 16_000_000
MAXIMUM_STORED_BYTES = 128_000_000
ELEMENTS = ("H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr "
            "Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu "
            "Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og").split()
CHARGE = {element.lower(): number+1 for number, element in enumerate(ELEMENTS)}
LOCK = threading.Lock()
TRANSFERRED = 0


def write_json(value, path):
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, indent=2)+"\n", encoding="utf-8")
    temporary.replace(path)


def download(url, maximum=MAXIMUM_FILE_BYTES):
    global TRANSFERRED
    request = urllib.request.Request(url, headers={"User-Agent": "own-xc-research-data-acquisition", "Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(request, timeout=45) as response:
        length = response.headers.get("Content-Length")
        if length is not None and int(length) > maximum: raise RuntimeError("Remote file exceeds its explicit size bound")
        pieces, size = [], 0
        while True:
            part = response.read(min(65536, maximum-size+1))
            if not part: break
            size += len(part)
            with LOCK:
                TRANSFERRED += len(part)
                if TRANSFERRED > MAXIMUM_TRANSFER_BYTES: raise RuntimeError("Download transfer budget exhausted")
            if size > maximum: raise RuntimeError("Remote stream exceeds its explicit size bound")
            pieces.append(part)
        return b"".join(pieces)


def git_blob_hash(data):
    return hashlib.sha1(b"blob "+str(len(data)).encode()+b"\0"+data).hexdigest()


def raw_url(path):
    return f"https://raw.githubusercontent.com/{REPOSITORY}/{SHA}/{path}"


def destination(path):
    relative = PurePosixPath(path)
    if relative.is_absolute() or ".." in relative.parts or "\\" in path: raise ValueError("Unsafe source path")
    target = (DESTINATION / "data" / Path(*relative.parts)).resolve()
    if not target.is_relative_to((DESTINATION / "data").resolve()): raise ValueError("Source path escaped data root")
    return target


def acquire(entry):
    target = destination(entry["path"])
    if target.exists():
        data = target.read_bytes()
    else:
        data = download(raw_url(entry["path"]), min(MAXIMUM_FILE_BYTES, entry["size"]+1))
        if len(data) != entry["size"] or git_blob_hash(data) != entry["sha"]:
            raise RuntimeError(f"Pinned Git blob verification failed: {entry['path']}")
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix+".part")
        temporary.write_bytes(data); temporary.replace(target)
    if len(data) != entry["size"] or git_blob_hash(data) != entry["sha"]:
        raise RuntimeError(f"Existing pinned file changed: {entry['path']}")
    return {"path": entry["path"], "bytes": len(data), "git_blob_sha1": entry["sha"], "sha256": hashlib.sha256(data).hexdigest()}


def structure_record(subset, name):
    base = DESTINATION / "data" / subset / name
    xyz = base / "struc.xyz"
    if xyz.exists():
        lines = xyz.read_text().splitlines()
        count = int(lines[0]); atoms = [line.split()[0] for line in lines[2:count+2]]
        if len(atoms) != count: raise RuntimeError(f"Invalid atom-count metadata: {subset}/{name}")
        coordinate_path, units = f"{subset}/{name}/struc.xyz", "angstrom"
    else:
        coordinate_path, units = f"{subset}/{name}/coord", "bohr"
        lines = (base / "coord").read_text().splitlines()
        atoms, active = [], False
        for line in lines:
            if line.strip().startswith("$coord"):
                active = True
                if "angs" in line.lower(): units = "angstrom"
                continue
            if active and line.strip().startswith("$"): break
            if active and line.strip(): atoms.append(line.split()[3])
        count = len(atoms)
    charge_file, spin_file = base / ".CHRG", base / ".UHF"
    charge = int(charge_file.read_text().strip()) if charge_file.exists() else 0
    spin = int(spin_file.read_text().strip()) if spin_file.exists() else 0
    unknown = sorted({atom for atom in atoms if atom.lower() not in CHARGE})
    electrons = None if unknown else sum(CHARGE[atom.lower()] for atom in atoms)-charge
    return {"id": f"{subset}/{name}", "subset": subset, "coordinate_path": coordinate_path, "coordinate_units": units,
            "atom_count": count, "elements": sorted(set(atom.capitalize() for atom in atoms)),
            "charge": charge, "unpaired_electrons": spin, "physical_electrons": electrons,
            "unknown_element_tokens": unknown, "charge_source_present": charge_file.exists(), "spin_source_present": spin_file.exists(),
            "integer_sector_metadata_consistent": electrons is not None and electrons >= abs(spin) and (electrons-spin) % 2 == 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    DESTINATION.mkdir(parents=True, exist_ok=True)
    report_path = ROOT / "electronfield-killtest/research/gmtkn55_data_acquisition_20261117.json"
    if report_path.exists():
        previous = json.loads(report_path.read_text())
        if previous.get("status") == "completed" or not args.resume: raise RuntimeError("Do not repeat a terminal acquisition; resume an interrupted one explicitly")
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "status": "running", "repository": REPOSITORY,
              "commit_sha": SHA, "source_version": "v1; original benchmark with the documented MB16-43 average correction",
              "purpose": "Own-model public benchmark data only; no method execution, training or numerical evaluation",
              "failures": [], "coordinate_arithmetic_on_cpu": False, "reference_energy_values_parsed": False,
              "published_method_code_or_results_acquired": False, "new_environments_or_dependencies_installed": False}
    write_json(report, report_path)
    try:
        tree_path = DESTINATION / "pinned_tree.json"
        if tree_path.exists(): tree = json.loads(tree_path.read_text())
        else:
            tree = json.loads(download(f"https://api.github.com/repos/{REPOSITORY}/git/trees/{SHA}?recursive=1"))
            if tree.get("truncated"): raise RuntimeError("GitHub returned an incomplete recursive tree; no partial benchmark is accepted")
            write_json(tree, tree_path)
        if tree.get("truncated"): raise RuntimeError("Pinned source tree is truncated")
        entries = {entry["path"]: entry for entry in tree["tree"] if entry["type"] == "blob"}
        listing_entry = entries[".list.dirs"]
        listing = acquire(listing_entry)
        subsets = (DESTINATION / "data/.list.dirs").read_text().split()
        selected = []
        for path, entry in entries.items():
            parts = PurePosixPath(path).parts
            allowed = (len(parts) == 1 and path in ("LICENSE", "README.md", ".bib", ".list.dirs", ".res"))
            allowed |= (len(parts) == 2 and parts[0] in subsets and parts[1] in (".bib", ".res", ".resRC", ".average", ".numen"))
            allowed |= (len(parts) == 3 and parts[0] in subsets and parts[2] in ("coord", "struc.xyz", ".CHRG", ".UHF"))
            if allowed:
                if entry.get("mode") not in ("100644", "100755"): raise RuntimeError("Data allowlist contains a non-regular file")
                selected.append(entry)
        required = sum(entry["size"] for entry in selected)+tree_path.stat().st_size
        if required > MAXIMUM_STORED_BYTES: raise RuntimeError("Pinned selected data exceeds the declared storage bound")
        report.update({"selected_file_count": len(selected), "predicted_data_and_tree_bytes": required,
                       "structure_subset_count": len(subsets), "source_tree_sha": tree["sha"]})
        write_json(report, report_path)
        print(json.dumps({"phase": "pinned_data_allowlist", "files": len(selected), "storage_bound_bytes": required,
                          "structure_subsets": len(subsets)}), flush=True)
        records = []
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(acquire, entry): entry["path"] for entry in selected}
            for future in as_completed(futures):
                records.append(future.result())
                if len(records) % 250 == 0: print(json.dumps({"phase": "verified_data_files", "count": len(records), "total": len(selected)}), flush=True)
        geometry_keys = sorted({tuple(PurePosixPath(record["path"]).parts[:2]) for record in records
                                if len(PurePosixPath(record["path"]).parts) == 3 and PurePosixPath(record["path"]).name in ("coord", "struc.xyz")})
        structures = [structure_record(*key) for key in geometry_keys]
        reaction_files = [record["path"] for record in records if PurePosixPath(record["path"]).name in (".res", ".resRC")]
        evaluation_sets = list(subsets)
        if "BH76/.resRC" in reaction_files: evaluation_sets.append("BH76RC")
        manifest = {"source_repository": f"https://github.com/{REPOSITORY}", "commit_sha": SHA, "branch_at_acquisition": "v1",
                    "license": "CC-BY-4.0 for dataset; upstream LICENSE retained", "citation_doi": "10.1039/C7CP04913G",
                    "data_unchanged_from_pinned_source": True, "files": sorted(records, key=lambda record: record["path"]),
                    "structures": structures, "evaluation_sets": sorted(evaluation_sets), "reaction_definition_files": sorted(reaction_files),
                    "reference_energy_values_parsed": False, "numerical_evaluation_performed": False,
                    "metric_policy": "Keep the original full-set version explicit. Recompute reference weights on CUDA before scoring; do not silently substitute v2, filter cases or reuse the uncorrected 56.84 normalization.",
                    "locked_confirmation_panel_opened": False}
        manifest_path = DESTINATION / "manifest.json"; write_json(manifest, manifest_path)
        report.update({"status": "completed", "passed_data_integrity": True, "manifest": str(manifest_path.relative_to(ROOT)),
                       "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(), "structure_count": len(structures),
                       "evaluation_set_count": len(evaluation_sets), "unknown_element_records": [row["id"] for row in structures if row["unknown_element_tokens"]],
                       "inconsistent_sector_metadata_records": [row["id"] for row in structures if not row["integer_sector_metadata_consistent"]],
                       "transferred_bytes_this_invocation": TRANSFERRED, "wtmad2_measured": False, "goal_accuracy_achieved": False})
        write_json(report, report_path)
        print(json.dumps({key: value for key, value in report.items() if key not in ("failures",)}), flush=True)
    except Exception as exc:
        report["status"] = "interrupted"; report["failures"].append({"error": f"{type(exc).__name__}: {exc}"})
        write_json(report, report_path); print(json.dumps(report["failures"][-1]), flush=True)
        raise


if __name__ == "__main__": main()
