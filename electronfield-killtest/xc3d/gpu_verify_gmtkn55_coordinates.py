"""Full pinned input coordinate validation; all scientific arithmetic on CUDA."""
from .runtime import configure, torch_setup
ROOT = configure(gpu_only_compatible=True)

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import torch


def read_xyz(path):
    lines = path.read_text().splitlines()
    count = int(lines[0])
    parts = [line.split() for line in lines[2:count+2]]
    if len(parts) != count or any(len(row) < 4 for row in parts): raise ValueError("Invalid XYZ atom metadata")
    # Parsing decimal source strings is file I/O. Units, distances and
    # numerical comparisons below are evaluated only on CUDA.
    return [row[0].lower() for row in parts], [[float(value) for value in row[1:4]] for row in parts], "angstrom"


def read_coord(path):
    atoms, coordinates, active, units = [], [], False, "bohr"
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith("$coord"):
            if active: raise ValueError("Multiple coordinate headers")
            active = True
            if "angs" in stripped.lower(): units = "angstrom"
            continue
        if active and stripped.startswith("$"): break
        if active and stripped:
            parts = stripped.split()
            if len(parts) < 4: raise ValueError("Invalid TURBOMOLE atom metadata")
            coordinates.append([float(value) for value in parts[:3]]); atoms.append(parts[3].lower())
    if not active or not atoms: raise ValueError("Missing/nonphysical TURBOMOLE coordinates")
    return atoms, coordinates, units


@torch.no_grad()
def main():
    output = Path("research/gmtkn55_coordinate_checks_gpu_20261117.json")
    if output.exists(): raise RuntimeError("Do not repeat this complete coordinate check")
    protocol = json.loads(Path("research/gmtkn55_coordinate_protocol_20261117.json").read_text())
    device = torch_setup(protocol["seed"])
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "device": device, "protocol": protocol,
              "status": "running", "results": [], "failures": []}
    maximum_error = torch.zeros((), device=device)
    try:
        audit = json.loads(Path("research/gmtkn55_data_audit_20261117.json").read_text())
        if not audit["passed_data_audit"]: raise RuntimeError("Pinned source integrity/complete-set audit must pass first")
        manifest_path = ROOT / protocol["manifest"]
        manifest = json.loads(manifest_path.read_text())
        if manifest["commit_sha"] != protocol["source_commit"] or len(manifest["structures"]) != protocol["expected_structures"]:
            raise RuntimeError("Pinned input version/coverage differs from the frozen coordinate protocol")
        folder = manifest_path.parent / "data"
        pinned = {row["path"]: row for row in manifest["files"]}
        angstrom = maximum_error.new_tensor(float(protocol["angstrom_per_bohr"]))
        for structure in manifest["structures"]:
            paths = [f"{structure['id']}/coord", f"{structure['id']}/struc.xyz"]
            if any(path not in pinned for path in paths): raise RuntimeError(f"Missing source coordinate format: {structure['id']}")
            for path in paths:
                if hashlib.sha256((folder / path).read_bytes()).hexdigest() != pinned[path]["sha256"]:
                    raise RuntimeError(f"Pinned coordinate source changed: {path}")
            atoms_a, values_a, units_a = read_coord(folder / paths[0])
            atoms_b, values_b, units_b = read_xyz(folder / paths[1])
            if atoms_a != atoms_b or len(atoms_a) != structure["atom_count"]:
                raise RuntimeError(f"Atom metadata differs between source formats: {structure['id']}")
            first = torch.tensor(values_a, device=device)
            second = torch.tensor(values_b, device=device)
            if units_a == "angstrom": first = first/angstrom
            if units_b == "angstrom": second = second/angstrom
            finite = bool(torch.isfinite(first).all() & torch.isfinite(second).all())
            if not finite: raise RuntimeError(f"Nonfinite coordinate source: {structure['id']}")
            distances_a = torch.cdist(first, first, compute_mode="donot_use_mm_for_euclid_dist")
            distances_b = torch.cdist(second, second, compute_mode="donot_use_mm_for_euclid_dist")
            error = (distances_a-distances_b).abs().max()
            maximum_error = torch.maximum(maximum_error, error)
            duplicate = False
            if len(first) > 1:
                mask = ~torch.eye(len(first), dtype=torch.bool, device=device)
                duplicate = bool((distances_a[mask] < protocol["duplicate_nucleus_tolerance_bohr"]).any()
                                 | (distances_b[mask] < protocol["duplicate_nucleus_tolerance_bohr"]).any())
            valid = not duplicate and bool(error < protocol["maximum_pair_distance_difference_bohr"])
            row = {"id": structure["id"], "passed": valid, "maximum_pair_distance_difference_bohr": float(error),
                   "duplicate_nucleus_detected": duplicate, "atom_count": len(first), "physical_electrons": structure["physical_electrons"]}
            report["results"].append(row)
            if not valid: report["failures"].append({"id": structure["id"], "error": "Source geometries/units differ or duplicate nuclei detected"})
            if len(report["results"]) % 250 == 0:
                print(json.dumps({"phase": "coordinate_source_progress", "structures_checked": len(report["results"]),
                                  "failures": len(report["failures"])}), flush=True)
        report["result"] = {"all_source_geometries_passed": not report["failures"], "structure_count": len(report["results"]),
                            "maximum_pair_distance_difference_bohr": float(maximum_error), "cpu_coordinate_arithmetic": False,
                            "reference_energy_values_read": False, "model_energy_evaluated": False,
                            "benchmark_weights_numerically_validated": False, "wtmad2_measured": False, "goal_accuracy_achieved": False}
        print(json.dumps(report["result"]), flush=True)
    except Exception as exc:
        report["failures"].append({"error": f"{type(exc).__name__}: {exc}"}); print(json.dumps(report["failures"][-1]), flush=True)
    report["status"] = "completed"
    report["passed"] = not report["failures"] and len(report["results"]) == protocol["expected_structures"]
    report["peak_cuda_bytes"] = torch.cuda.max_memory_allocated()
    output.write_text(json.dumps(report, indent=2)+"\n")
    if not report["passed"]: raise RuntimeError("The complete benchmark coordinate validation failed; do not silently alter or omit cases")


if __name__ == "__main__": main()
