"""Consolidate existing reports without rerunning any solve or reference."""
from .runtime import configure, torch_setup
ROOT = configure(gpu_only_compatible=True)

from datetime import datetime, timezone
import json
from pathlib import Path
import torch


def read(name):
    return json.loads(Path("research", name).read_text())


def main():
    device = torch_setup()
    original = read("charge_spin_accuracy_gpu_20261117.json")
    continued = read("charge_spin_continuation_gpu_20261117.json")
    completion = read("stationary_panel_completion_gpu_20261117.json")
    peroxide = read("peroxide_accuracy_gpu_20261117.json")
    peroxide_done = read("peroxide_completion_gpu_20261117.json")
    larger = read("larger_charge_spin_accuracy_gpu_20261117.json")
    latest = {row["id"]: {**row, "latest_source": "charge_spin_accuracy_gpu_20261117.json"} for row in original["results"]}
    for row, source in [(continued, "charge_spin_continuation_gpu_20261117.json")]+[
            (row, "stationary_panel_completion_gpu_20261117.json") for row in completion["results"]]:
        latest[row["id"]] = {**latest[row["id"]], **row, "latest_source": source}
    latest[peroxide["case"]["id"]] = {**peroxide, "id": peroxide["case"]["id"], "electrons": peroxide["case"]["electrons"],
                                        "basis": peroxide["case"]["basis"], "latest_source": "peroxide_accuracy_gpu_20261117.json"}
    latest[peroxide_done["id"]].update({**peroxide_done, "latest_source": "peroxide_completion_gpu_20261117.json"})
    for row in larger["results"]:
        latest[row["id"]] = {**row, "latest_source": "larger_charge_spin_accuracy_gpu_20261117.json"}
    keys = ("id", "electrons", "basis", "energy_error_kcal", "total_energy_hartree", "reference_energy_hartree",
            "full_residual", "projected_residual", "physical_validity_passed", "projected_dimension", "full_fixed_sz_dimension",
            "latest_source")
    rows = [{key: value[key] for key in keys if key in value} for value in latest.values()]
    contrasts = {}
    for label, first, second in (("H4_geometries_ccpvdz", "H4_parallel_far", "H4_square_close"),
                                 ("BeH2_stretch_sto3g", "BeH2_neutral_1.33", "BeH2_neutral_2.10"),
                                 ("NH3_geometries_sto3g", "NH3_pyramidal", "NH3_planar"),
                                 ("BeH2_new_stretch_631g", "BeH2_631g_1.65_new", "BeH2_631g_2.75_new")):
        error = torch.tensor([latest[first]["energy_error_kcal"], latest[second]["energy_error_kcal"]], device=device)
        contrasts[label] = float(error[1]-error[0])
    errors = torch.tensor([row["energy_error_kcal"] for row in rows], device=device)
    universal_search = read("universal_search_gpu_20261117.json")
    checks = {}
    for key, filename in (("tensor_export", "spin_tensor_state_checks_gpu_20261117.json"),
                          ("rejected_coulomb_gram_frame", "intrinsic_density_certificate_gpu_20261117.json"),
                          ("spatial_density_frame", "spatial_density_certificate_gpu_20261117.json"),
                          ("spatial_density_derivative", "spatial_derivative_gpu_20261117.json")):
        record = read(filename)
        checks[key] = {"source": filename, "failures": record["failures"], "results": [
            {name: value for name, value in row.items() if name not in ("kinetic_history", "sectors")}
            for row in record["results"]]}
    checks["singlet_projection"] = read("singlet_projection_checks_gpu_20261117.json")
    auxiliary = read("auxiliary_stationary_gpu_20261117.json")
    derivative = read("auxiliary_derivative_gpu_20261117.json")
    auxiliary_keys = ("id", "prediction_valid", "self_consistent", "physical_total_energy_hartree",
                      "energy_error_kcal", "reference_energy_hartree", "number_error",
                      "minimum_physical_occupation", "maximum_physical_occupation", "kinetic_converged",
                      "kinetic_gap_hartree", "ks_commutator_norm", "global_check", "xc_potential_accepted",
                      "new_reference_generated", "oracle_density_used_for_prediction",
                      "post_interruption_development_completion")
    auxiliary_rows = [{**{key: row[key] for key in auxiliary_keys if key in row},
                       "final_stationarity": row["history"][-1]} for row in auxiliary["results"]]
    auxiliary_points_passed = (not auxiliary["failures"] and len(auxiliary_rows) == 2
                               and all(row["prediction_valid"] and row["xc_potential_accepted"] for row in auxiliary_rows)
                               and derivative["passed"])
    failed_ensembles = {}
    for label, filename in (("thermal", "ensemble_spatial_search_gpu_20261117.json"),
                             ("positive_primal", "positive_ensemble_spatial_gpu_20261117.json")):
        failed = read(filename)
        failed_ensembles[label] = {"source": filename, "protocol_closed": True,
                                   "failures": failed["failures"], "results": [
            {key: row[key] for key in ("id", "converged", "status", "density_residual",
                                       "universal_energy_minus_saved_oracle_kcal", "xc_energy_available",
                                       "xc_potential_available") if key in row} for row in failed["results"]]}
    for key, filename in (("density_links", "density_link_checks_gpu_20261117.json"),
                          ("positive_ensemble_primitive", "positive_ensemble_checks_gpu_20261117.json"),
                          ("density_resolvent", "density_resolvent_checks_gpu_20261117.json"),
                          ("response_rank_before_repair", "response_rank_before_repair_gpu_20261117.json"),
                          ("response_rank_after_repair", "response_rank_after_repair_gpu_20261117.json"),
                          ("auxiliary_spin_failure", "auxiliary_spin_failure_gpu_20261117.json"),
                          ("auxiliary_spin_guard", "auxiliary_spin_guard_checks_gpu_20261117.json")):
        checks[key] = read(filename)
    for key, filename in (("purified_tensor_operator", "purified_operator_checks_gpu_20261117.json"),
                          ("tensor_response_initial_test_cap_failure", "spin_response_checks_gpu_20261117.json"),
                          ("tensor_response_test_capacity_completion", "spin_response_capacity_completion_gpu_20261117.json"),
                          ("tensor_physical_response_metric", "physical_response_metric_checks_gpu_20261117.json"),
                          ("tensor_density", "tensor_density_checks_gpu_20261117.json")):
        record = read(filename)
        checks[key] = {"source": filename, "passed": record["passed"], "failures": record.get("failures", []),
                       "results": [{name: value for name, value in row.items() if name != "diagnostics"} for row in record["results"]]}
    water_initial = read("tensor_water_scope_gpu_20261117.json")
    water_metric = read("tensor_water_physical_metric_gpu_20261117.json")
    metric = water_metric["result"]["response"]
    norm_check = torch.tensor([metric["full_image_difference_squared_norm"], metric["sequential_projection_error_bound"],
                               metric["exact_full_image_squared_norm"]], device=device)
    bound_consistent = bool(norm_check[0].clamp_min(0).sqrt() <= norm_check[1]+1e-8*norm_check[2].sqrt())
    tensor_keys = ("physical_scope_passed", "total_energy_hartree", "input_total_energy_hartree", "lowering_from_input_kcal",
                   "particle_number", "particle_number_variance", "spin_squared", "energy_identity_error_hartree",
                   "hermiticity_error_hartree", "state_bond_dimensions", "full_fixed_sz_dimension",
                   "full_occupation_vector_allocated", "new_reference_generated_or_loaded", "full_stationarity_certified",
                   "chemical_accuracy_measured", "xc_functional_evaluated", "wtmad2_measured", "goal_accuracy_achieved")
    tensor_scope = {"functional_optimizer_integrated": False, "general_chemical_accuracy_measured": False,
                    "physical_product_response_error_measured_without_global_vector": True,
                    "correlated_candidate_full_stationarity_certified": False,
                    "water_original": {"source": "tensor_water_scope_gpu_20261117.json",
                                       **{key: water_initial["result"][key] for key in tensor_keys},
                                       "raw_channel_metric_closed": True},
                    "water_physical_metric": {"source": "tensor_water_physical_metric_gpu_20261117.json",
                                               **{key: water_metric["result"][key] for key in tensor_keys},
                                               "full_image_relative_error_upper_estimate": metric["full_image_relative_error_upper_estimate"],
                                               "original_span_retained_squared_image_norm_fraction": metric["original_zipup_span_retained_image_norm_fraction"],
                                               "physical_span_retained_squared_image_norm_fraction": metric["physical_metric_span_retained_image_norm_fraction"],
                                               "physical_image_projection_bound_consistent": bound_consistent,
                                               "peak_cuda_bytes": water_metric["peak_cuda_bytes"]},
                    "saved_water_density": read("tensor_water_density_gpu_20261117.json")}
    local_checks = read("local_tensor_functional_checks_gpu_20261117.json")
    local_water = read("tensor_water_conditional_gpu_20261117.json")
    local_recovery = read("tensor_section_recovery_gpu_20261117.json")
    checks["local_tensor_conditional_functional"] = local_checks
    checks["proximal_optimizer_checkpoint"] = read("resolvent_checkpoint_checks_gpu_20261117.json")
    checks["completed_tensor_section_recovery"] = local_recovery
    previous_energy = torch.tensor([water_metric["result"]["total_energy_hartree"],
                                    local_water["result"]["original_total_energy_hartree"]], device=device)
    input_energy_identity = float((previous_energy[1]-previous_energy[0]).abs())
    tensor_scope["conditional_optimizer_integrated"] = True
    tensor_scope["full_tensor_universal_optimizer_integrated"] = False
    tensor_scope["water_conditional_functional"] = {
        "source": "tensor_water_conditional_gpu_20261117.json", "status": local_water["status"],
        "passed_at_frozen_context": local_water["passed"], "result": local_water["result"],
        "same_saved_model_state_energy_identity_error_hartree": input_energy_identity,
        "same_saved_model_state_energy_identity_passed": input_energy_identity < 1e-7,
        "completed_recovery_passed": local_recovery["passed"], "peak_cuda_bytes": local_water["peak_cuda_bytes"],
        "scope": "Frozen surrounding tensors and one local positive ensemble; no global tensor stationarity or chemical reference error"}
    for key, filename in (("all_site_physical_forward", "physical_tensor_forward_checks_gpu_20261117.json"),
                          ("failed_inverse_gauge_recovery", "tensor_gauge_recovery_gpu_20261117.json"),
                          ("direct_factor_gauge_recovery", "tensor_gauge_direct_recovery_gpu_20261117.json"),
                          ("complete_two_site_operator", "two_site_tensor_checks_gpu_20261117.json"),
                          ("physical_sector_enrichment", "sector_enrichment_checks_gpu_20261117.json"),
                          ("bounded_spin_complete_split", "bounded_spin_split_checks_gpu_20261117.json"),
                          ("variable_sector_forward", "two_site_forward_checks_gpu_20261117.json")):
        record = read(filename)
        checks[key] = {"source": filename, "passed": record["passed"], "failures": record.get("failures", []),
                       **{name: record[name] for name in ("result", "results", "seed_sector_restriction") if name in record}}
    failed_forward = read("tensor_water_forward_gpu_20261117.json")
    closed_protocol = read("tensor_water_forward_protocol_20261117.json")
    water_sector = read("tensor_water_sector_gpu_20261117.json")
    tensor_scope["water_fixed_seed_forward"] = {
        "source": "tensor_water_forward_gpu_20261117.json", "passed": failed_forward["passed"],
        "protocol_status": closed_protocol["status"], "closure_reason": closed_protocol["closure_reason"],
        "failures": failed_forward["failures"],
        "completed_valid_updates": sum(row.get("phase") == "physical_site_update" for row in failed_forward["history"]),
        "maximum_virtual_electrons_permitted": 2}
    tensor_scope["water_missing_sector_response"] = {
        "source": "tensor_water_sector_gpu_20261117.json", "status": water_sector["status"],
        "passed": water_sector["passed"], "result": water_sector.get("result"), "failures": water_sector["failures"],
        "peak_cuda_bytes": water_sector["peak_cuda_bytes"],
        "scope": "One residual-selected sector completion and local variational update; no chemical reference error or full tensor stationarity"}
    bh_accuracy = read("tensor_bh_accuracy_gpu_20261117.json")
    tensor_scope["bh_larger_basis_accuracy"] = {
        "source": "tensor_bh_accuracy_gpu_20261117.json", "status": bh_accuracy["status"],
        "terminal_test_passed": bh_accuracy.get("passed") if bh_accuracy["status"] == "completed" else None,
        "predictions": bh_accuracy["predictions"], "references": bh_accuracy["references"], "results": bh_accuracy["results"],
        "failures": bh_accuracy["failures"],
        "balanced_stretch_error_kcal_per_mol": bh_accuracy.get("balanced_stretch_error_kcal_per_mol"),
        "reference_seed_correctness_amendment": bh_accuracy.get("reference_seed_correctness_amendment"),
        "xc_functional_evaluated": False, "general_out_of_sample_accuracy_established": False,
        "scope": "Two new finite-basis physical predictions; not a complete-basis, XC-functional or WTMAD-2 result"}
    public_data = read("gmtkn55_data_audit_20261117.json")
    checks["official_full_benchmark_data"] = {
        "source": "gmtkn55_data_audit_20261117.json", "passed_data_audit": public_data["passed_data_audit"],
        "commit_sha": public_data["commit_sha"], "evaluation_set_count": public_data["evaluation_set_count"],
        "structure_count": public_data["structure_count"], "verified_file_count": public_data["verified_file_count"],
        "failures": public_data["failures"], "numerical_benchmark_or_model_validation": False}
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "device": device,
              "scope": "Finite-basis development/application evidence, not full chemistry or challenge accuracy",
              "latest_state_results": rows, "maximum_absolute_same_basis_error_kcal": float(errors.abs().max()),
              "balanced_contrast_errors_kcal": contrasts,
              "original_four_update_panel_passed": original["rejection_panel_passed"],
              "original_close_h4_error_kcal": next(row["energy_error_kcal"] for row in original["results"] if row["id"] == "H4_square_close"),
              "original_peroxide_32_update_result": {key: peroxide[key] for key in ("full_residual", "energy_error_kcal")},
              "peroxide_continuation_is_post_label_development": True,
              "universal_functional_interface": {key: universal_search[key] for key in
                                                 ("status", "converged", "error", "density_residual", "xc_energy_available")
                                                 if key in universal_search},
              "universal_interface_last_accepted_density_step": universal_search.get("history", [])[-1:] or None,
              "unfinished_physical_residuals": [{"id": row["id"], "residual": row["full_residual"]}
                                                 for row in rows if row["full_residual"] > 1e-5],
              "state_case_failures": original["failures"]+completion["failures"]+larger["failures"], "checks": checks,
              "functional_interface_passed": universal_search.get("status") == "completed" and universal_search.get("converged", False),
              "functional_interface_passed_scope": "Earlier cold hard-physical-density inverse; remains failed and closed",
              "closed_hard_density_ensemble_routes": failed_ensembles,
              "auxiliary_functional_interface": {
                  "source": "auxiliary_stationary_gpu_20261117.json", "results": auxiliary_rows,
                  "passed_at_reported_development_points": auxiliary_points_passed,
                  "derivative_check": derivative,
                  "derivative_scope": "Frozen response spaces, one auxiliary direction per geometry; not nuclear forces or full-space path accuracy",
                  "development_completion": True, "general_out_of_sample_accuracy_established": False,
                  "auxiliary_coordinate_is_physical_density": False,
                  "unchanged_conventional_ks_plugin": False, "challenge_eligibility_verified": False,
                  "balanced_stretch_contrast_error_kcal": auxiliary["balanced_stretch_contrast_error_kcal"]},
              "resource_usage": read("resource_accounting_20261117.json"),
              "new_tensor_scope": tensor_scope,
              "unmeasured_tensor_scope_full_residuals": ["H2O_ccpvdz_tensor_scope_new", "H2O_ccpvdz_conditional_center6_state",
                                                       "H2O_ccpvdz_sector_completed_state"],
              "general_functional_accuracy_gate_passed": False,
              "official_full_benchmark_model_score_available": False,
              "no_numerical_solve_repeated": True, "speed_work_authorized": False,
              "goal_accuracy_achieved": False, "wtmad2_measured": False,
              "limitations": ["Same finite-basis references, not CBS/GMTKN55", "Accepted auxiliary functional still uses global Fock vectors",
                              "Some selected spaces approach the full Hilbert space", "Accepted auxiliary targets only even-N singlets; tensor primitives also cover odd spin",
                              "Conditional tensor universal optimization connected, but global tensor stationarity/context independence unproved",
                              "Fixed-seed tensor sweeps closed after metric failure and missing higher-excitation sectors; local sector completion is not a global accuracy certificate",
                              "Frozen-context XC derivative checks do not include surrounding-tensor or nuclear/basis response",
                              "Exact tensor response metric currently requires a product source; the correlated molecular state is unconverged",
                              "Cold hard-density interface and both bounded ensemble solvers failed and remain closed",
                              "Auxiliary functional evidence covers two inspected finite-basis points and frozen-space derivatives",
                              "Regularized KS challenge integration and fixed-continuum metric remain unverified",
                              "No learned/amortized inference or broad reaction-family generalization",
                              "References use the shared physical Hamiltonian backend; not independent integral implementations",
                              "Locked composition-confirmation panel remains withheld"]}
    Path("research/accuracy_status_gpu_20261117.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({"report": "research/accuracy_status_gpu_20261117.json",
                      "state_cases": len(rows), "maximum_absolute_same_basis_error_kcal": report["maximum_absolute_same_basis_error_kcal"],
                      "cold_physical_density_interface_passed": report["functional_interface_passed"],
                      "auxiliary_development_points_passed": auxiliary_points_passed,
                      "tensor_molecular_scope_passed": water_metric["result"]["physical_scope_passed"],
                      "tensor_full_image_projection_bound_consistent": bound_consistent,
                      "tensor_functional_optimizer_integrated": False,
                      "tensor_conditional_functional_passed": local_water["passed"],
                      "tensor_conditional_checkpoint_recovery_passed": local_recovery["passed"],
                      "tensor_missing_sector_physical_test_passed": water_sector["passed"],
                      "tensor_correlated_full_stationarity_certified": False,
                      "goal_accuracy_achieved": False, "wtmad2_measured": False,
                      "speed_work_authorized": False}), flush=True)


if __name__ == "__main__":
    main()
