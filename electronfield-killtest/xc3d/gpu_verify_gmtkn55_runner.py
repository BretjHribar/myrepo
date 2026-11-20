"""Validate the complete public benchmark parser/scorer, without a model score."""
from .runtime import configure, torch_setup
ROOT = configure(gpu_only_compatible=True)

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile
import torch
from .gmtkn55 import GMTKN55Suite, GROUPS


@torch.no_grad()
def main():
    destination = Path("research/gmtkn55_runner_checks_gpu_20261117.json")
    if destination.exists():
        raise RuntimeError("Do not repeat this completed full-suite pipeline validation")
    protocol_path = Path("research/gmtkn55_scoring_protocol_20261117.json")
    protocol = json.loads(protocol_path.read_text())
    coordinate_path = Path("research/gmtkn55_coordinate_checks_gpu_20261117.json")
    coordinates = json.loads(coordinate_path.read_text())
    if not coordinates["passed"]:
        raise RuntimeError("Complete coordinate validation must pass first")
    device = torch_setup(17112068)
    suite = GMTKN55Suite(ROOT, protocol)
    tensors = suite.tensors(device)
    checks = []
    rows = {row["id"]: row for row in suite.reactions}
    known = {
        "G21IP/001": (["G21IP/h"], [-1], "314.9"),
        "G21IP/002": (["G21IP/li+", "G21IP/li"], [1, -1], "123.3"),
        "ADIM6/001": (["ADIM6/AM2", "ADIM6/AD2"], [2, -1], "1.34"),
        "BH76RC/001": (["BH76/h", "BH76/n2o", "BH76/oh", "BH76/n2"], [-1, -1, 1, 1], "-64.91"),
        "S22/001": (["S22/01", "S22/01a"], [-1, 2], "3.133"),
    }
    for name, (ids, coefficients, reference) in known.items():
        row = rows[name]
        passed = row["structure_ids"] == ids and row["coefficients"] == coefficients and row["reference_kcal_per_mol_text"] == reference
        checks.append({"name": f"literal_source_reaction_{name}", "passed": passed})
    # Complete independent scalar contractions use the source's individual
    # integer terms. This catches matrix indexing, sign, coefficient and
    # Hartree-to-kcal errors across all 1,505 reactions.
    generator = torch.Generator(device=device).manual_seed(17112068)
    synthetic = 1000.*torch.randn(len(suite.required_ids), generator=generator, device=device)
    direct = []
    for row in suite.reactions:
        terms = synthetic[torch.tensor([suite.required_index[name] for name in row["structure_ids"]], device=device)]
        coefficient = terms.new_tensor(row["coefficients"])
        direct.append((terms*coefficient).sum()*tensors["hartree_to_kcal"])
    direct = torch.stack(direct)
    assembly_error = ((tensors["matrix"] @ synthetic)*tensors["hartree_to_kcal"]-direct).abs().max()
    checks.append({"name": "all_reaction_contractions_and_units", "passed": bool(assembly_error < 1e-7),
                   "maximum_difference_kcal_per_mol": float(assembly_error)})
    full = suite.score(synthetic, torch.ones_like(synthetic, dtype=torch.bool), tensors)
    numerator = direct.new_zeros(())
    mean_rows = []
    for name, metadata in suite.subsets.items():
        section = slice(metadata["start"], metadata["stop"])
        reference = tensors["references"][section]
        error = direct[section]-reference
        published_mean = reference.new_tensor(float(metadata["average_kcal_per_mol_text"]))
        weight = reference.new_tensor(float(protocol["normalization_kcal_per_mol"]))/published_mean
        numerator += len(reference)*weight*error.abs().mean()
        mean_rows.append({"subset": name, "count": len(reference), "pinned_average_kcal_per_mol": float(published_mean),
                          "reference_derived_mean_absolute_kcal_per_mol": float(reference.abs().mean()),
                          "derived_minus_pinned_mean_kcal_per_mol": float(reference.abs().mean()-published_mean)})
    expected = numerator/len(suite.reactions)
    weight_error = (expected-expected.new_tensor(full["wtmad2_kcal_per_mol"])).abs()
    checks.append({"name": "independent_complete_weighted_aggregation", "passed": bool(weight_error < 1e-7),
                   "difference_kcal_per_mol": float(weight_error)})
    acceptance = torch.ones_like(synthetic, dtype=torch.bool)
    acceptance[suite.required_index["G21IP/h"]] = False
    partial = suite.score(synthetic, acceptance, tensors)
    expected_missing = {row["id"] for row in suite.reactions if "G21IP/h" in row["structure_ids"]}
    actual_missing = {row["id"] for row in partial["reaction_results"] if not row["scored"]}
    checks.append({"name": "missing_structure_never_reports_full_score", "passed":
                   actual_missing == expected_missing and partial["wtmad2_kcal_per_mol"] is None and not partial["public_accuracy_threshold_met"]})
    nonfinite = synthetic.clone(); nonfinite[suite.required_index["G21IP/h"]] = float("nan")
    invalid = suite.score(nonfinite, torch.ones_like(acceptance), tensors)
    checks.append({"name": "nonfinite_prediction_is_not_silently_scored", "passed": invalid["wtmad2_kcal_per_mol"] is None and invalid["measured_reactions"] == partial["measured_reactions"]})
    absent = suite.score(synthetic, torch.zeros_like(acceptance), tensors)
    checks.append({"name": "empty_predictions_do_not_become_accuracy_estimate", "passed": absent["measured_reactions"] == 0 and absent["full_suite_point_estimate_kcal_per_mol"] is None})
    candidate = {"id": "HARNESS_ONLY_NOT_A_MODEL", "basis": "synthetic", "protocol_sha256": "0"*64, "source_sha256": "0"*64}
    template = {"schema": "gmtkn55_total_energies_v1", "benchmark_commit": protocol["source_commit"],
                "manifest_sha256": suite.manifest_sha256, "units": "hartree", "energy_kind": protocol["prediction_energy_kind"],
                "candidate": candidate, "benchmark_reference_used_for_prediction": False, "predictions": []}
    cases = {
        "duplicate_id": {"predictions": [{"id": "G21IP/h", "status": "uncomputed", "reason": "test"}]*2},
        "wrong_units": {"units": "kcal_per_mol"},
        "electronic_only_energy": {"energy_kind": "electronic_hartree"},
        "reference_leakage": {"benchmark_reference_used_for_prediction": True},
        "wrong_version": {"benchmark_commit": "wrong"},
        "unconverged_energy": {"predictions": [{"id": "G21IP/h", "status": "accepted", "total_energy_hartree": -0.5, "converged": False, "physical_checks_passed": True}]},
        "unphysical_energy": {"predictions": [{"id": "G21IP/h", "status": "accepted", "total_energy_hartree": -0.5, "converged": True, "physical_checks_passed": False}]},
    }
    with tempfile.TemporaryDirectory(prefix="gmtkn55_contract_", dir=ROOT/"data") as temporary:
        for name, mutation in cases.items():
            path = Path(temporary)/f"{name}.json"
            path.write_text(json.dumps({**template, **mutation}))
            rejected = False
            try: suite.read_predictions(path, device)
            except ValueError: rejected = True
            checks.append({"name": f"contract_rejects_{name}", "passed": rejected})
    folder = ROOT/"data/gmtkn55_pipeline"; folder.mkdir(exist_ok=True)
    inputs = {"benchmark_commit": protocol["source_commit"], "manifest_sha256": suite.manifest_sha256,
              "structure_inventory": len(suite.structures), "structures_required_by_all_reactions": len(suite.required_ids),
              "structures": [suite.structures[name] for name in suite.required_ids],
              "reference_energies_included": False, "source_geometry_folder": "data/gmtkn55_official_v1/data"}
    (folder/"prediction_inputs.json").write_text(json.dumps(inputs, indent=2)+"\n")
    (folder/"prediction_contract_template.json").write_text(json.dumps(template, indent=2)+"\n")
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "device": device, "protocol": protocol,
              "protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
              "source_metadata_sha256": suite.source_hashes, "manifest_sha256": suite.manifest_sha256,
              "coordinate_check_sha256": hashlib.sha256(coordinate_path.read_bytes()).hexdigest(),
              "passed": all(row["passed"] for row in checks), "checks": checks,
              "evaluation_sets": len(suite.subsets), "reactions": len(suite.reactions),
              "required_structures": len(suite.required_ids), "inventory_structures": len(suite.structures),
              "unreferenced_upstream_structures": sorted(set(suite.structures)-set(suite.required_ids)),
              "subset_reference_means": mean_rows,
              "group_reaction_counts": {name: sum(suite.subsets[subset]["count"] for subset in names) for name, names in GROUPS.items()},
              "harness_only_synthetic_inputs": True, "actual_model_predictions": 0, "wtmad2_measured": False,
              "initial_model_accuracy_estimate_kcal_per_mol": None, "goal_accuracy_achieved": False,
              "organizer_private_tests_passed": False, "cpu_scientific_arithmetic": False,
              "peak_cuda_bytes": torch.cuda.max_memory_allocated()}
    destination.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({key: report[key] for key in ("passed", "evaluation_sets", "reactions", "required_structures", "actual_model_predictions", "wtmad2_measured", "peak_cuda_bytes")}), flush=True)
    if not report["passed"]:
        raise RuntimeError("Full benchmark pipeline validation failed; retain the failure record")


if __name__ == "__main__": main()
