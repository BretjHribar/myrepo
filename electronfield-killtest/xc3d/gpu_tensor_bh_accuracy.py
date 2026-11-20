"""Larger-basis own-state predictions, then independent singlet references."""
from .runtime import configure, torch_setup
ROOT = configure(gpu_only_compatible=True)

import argparse
from datetime import datetime, timezone
import json
from math import comb
from pathlib import Path
import gc
import torch
from .gpu_chemistry_transfer import cuda_operators
from .gpu_fermion_mpo import FermionMPO
from .gpu_spin_response import product_reference
from .physical_two_site_forward import solve_two_site
from .physical_tensor_forward import payload
from .gpu_tensor_water_conditional import atomic_save
from .gpu_manybody_ops import fixed_sector
from .gpu_six_particle import SixParticleOperator, SixParticleFockAction
from .gpu_fock_blocks import SingletProjector
from .gpu_four_particle import incremental_ritz

KCAL = 627.509474


@torch.no_grad()
def reference(data, protocol, policy, case_index, device, log):
    sites = len(data["h"]); dimension = comb(sites, 3)**2
    # Conservative bounded-work estimate includes Ritz images, the dense
    # antisymmetric particle tensor and its action temporaries, index maps,
    # and singlet links. No such storage belongs to the prediction model.
    estimated = 2*dimension*protocol["reference_maximum_dimension"]*8+8*sites**6*8+36*dimension*8+500_000_000
    if estimated > protocol["reference_maximum_predicted_bytes"]:
        raise RuntimeError(f"Independent reference exceeds its predeclared CUDA allocation estimate: {estimated}")
    free, _ = torch.cuda.mem_get_info()
    if estimated > free: raise RuntimeError("Independent reference exceeds currently free CUDA memory")
    log({"phase": "independent_reference_after_both_predictions", "dimension": dimension, "estimated_cuda_bytes": estimated})
    bits, configurations = fixed_sector(sites, 3, 3, device)
    del configurations
    action = SixParticleFockAction(SixParticleOperator(data["h"], data["eri"], 6), bits)
    singlet = SingletProjector(bits, sites, 3)
    generator = torch.Generator(device=device).manual_seed(policy["seed"]+case_index)
    seed = singlet(torch.randn(len(bits), device=device, generator=generator))
    seed /= seed.norm()
    determinant = (1 << 6)-1
    location = torch.searchsorted(bits, bits.new_tensor(determinant))
    if int(bits[location]) != determinant: raise RuntimeError("Reference determinant is outside its fixed-N sector")
    seed[location] += 1.
    result = incremental_ritz(action, seed, maximum=protocol["reference_maximum_dimension"],
                              tolerance=protocol["reference_residual_tolerance_hartree"], projector=singlet)
    vector = result["state"]
    literal_image = action(vector)
    residual = float((literal_image-result["energy"]*vector).norm())
    spin = float((vector @ singlet.spin_squared(vector)).abs())
    valid = result["converged"] and residual < protocol["reference_residual_tolerance_hartree"] and spin < protocol["reference_spin_tolerance"]
    return {"converged": valid, "electronic_energy_hartree": float(result["energy"]),
            "full_fixed_sz_residual_hartree": residual, "spin_squared": spin, "dimension": len(bits),
            "ritz_dimension": result["dimension"], "prediction_available_before_reference": True,
            "starting_vector_covers_other_spatial_symmetries": True,
            "single_fixed_seed": policy["seed"]+case_index,
            "independent_action": "Antisymmetric six-particle Cartesian tensor; no molecular MPO contraction",
            "integral_backend_shared": True}, {"bits": bits, "state": vector}


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    protocol = json.loads(Path("research/tensor_bh_accuracy_protocol_20261117.json").read_text())
    output = Path("research/tensor_bh_accuracy_gpu_20261117.json")
    if output.exists():
        report = json.loads(output.read_text())
        if not args.resume or report["status"] not in ("running", "interrupted") or report["protocol"] != protocol:
            raise RuntimeError("Do not repeat a terminal accuracy protocol or modify an interrupted one")
    else:
        if args.resume: raise RuntimeError("There is no interrupted BH accuracy protocol")
        report = {"created_utc": datetime.now(timezone.utc).isoformat(), "device": "cuda", "protocol": protocol,
                  "status": "running", "predictions": [], "references": [], "results": [], "history": [], "failures": []}
    device = torch_setup(protocol["seed"])
    reference_policy = json.loads(Path("research/tensor_bh_reference_policy_20261117.json").read_text())
    if report["references"] and report.get("reference_seed_correctness_amendment") != reference_policy:
        raise RuntimeError("Do not change a reference policy after any label is generated")
    if report.get("reference_seed_correctness_amendment", reference_policy) != reference_policy:
        raise RuntimeError("The prespecified reference correction changed")
    report["reference_seed_correctness_amendment"] = reference_policy
    if args.resume:
        report.setdefault("resumptions", []).append({
            "resumed_utc": datetime.now(timezone.utc).isoformat(),
            "previous_status": report["status"], "completed_predictions_retained": len(report["predictions"]),
            "completed_references_retained": len(report["references"]),
            "original_protocol_and_acceptance_thresholds_unchanged": True})
    report["status"] = "running"
    folder = ROOT / "data/tensor_bh_accuracy"; folder.mkdir(parents=True, exist_ok=True)
    def log(row):
        report["history"].append(row); print(json.dumps(row), flush=True)
        output.write_text(json.dumps(report, indent=2)+"\n")
    def inputs(case):
        destination = folder / f"{case['id']}.inputs.pt"
        if destination.exists(): return torch.load(destination, map_location=device, weights_only=True)
        log({"id": case["id"], "phase": "new_cuda_inputs"})
        data = cuda_operators(case, protocol["basis"])
        atomic_save(data, destination)
        return data
    output.write_text(json.dumps(report, indent=2)+"\n")
    try:
        if not json.loads(Path("research/two_site_forward_checks_gpu_20261117.json").read_text())["passed"]:
            raise RuntimeError("The changing-sector forward implementation has not passed its independent gate")
        for case in protocol["cases"]:
            if any(row["id"] == case["id"] for row in report["predictions"]): continue
            try:
                data = inputs(case)
                if not data["reference_status"]["converged"]: raise RuntimeError("The single bounded CUDA RHF preparation failed; no retries")
                sites = len(data["h"])
                log({"id": case["id"], "phase": "literal_prediction_operator", "spatial_orbitals": sites,
                     "input_status": data["reference_status"], "full_fixed_sz_dimension_for_later_reference": comb(sites, 3)**2})
                operator = FermionMPO.molecular(data["h"], data["eri"])
                checkpoint_path = folder / f"{case['id']}.forward.pt"
                checkpoint = None
                if checkpoint_path.exists():
                    if not args.resume: raise RuntimeError("Resume the existing forward checkpoint explicitly")
                    saved = torch.load(checkpoint_path, map_location=device, weights_only=True)
                    if saved["protocol"] != protocol: raise RuntimeError("The saved accuracy protocol changed")
                    checkpoint = saved["run"]
                result = solve_two_site(operator, product_reference(sites, 6, 0, device), protocol,
                    checkpoint=checkpoint,
                    checkpoint_callback=lambda run: atomic_save({"protocol": protocol, "run": run}, checkpoint_path),
                    callback=lambda row: log({"id": case["id"], **row}))
                valid = result["coordinate_stationarity_passed"] and result["physical_checks"]["passed"]
                summary = {"id": case["id"], "prediction_gate_passed": valid, "spatial_orbitals": sites,
                           "electronic_energy_hartree": result["electronic_energy_hartree"],
                           "total_energy_hartree": result["electronic_energy_hartree"]+float(data["nuclear"]),
                           "maximum_common_context_gap_hartree": result["maximum_common_context_gap_hartree"],
                           "physical_checks": result["physical_checks"], "sweeps": len(result["sweep_history"]),
                           "rejected_updates": sum(not row["accepted"] for row in result["history"]),
                           "full_hilbert_residual_certified": False, "global_variational_minimum_certified": False,
                           "correlated_reference_used_for_prediction": False}
                atomic_save({**payload(result["state"]), "density": result["density"], "prediction": summary,
                             "sweep_history": result["sweep_history"], "audit": result["audit"]}, folder / f"{case['id']}.prediction.pt")
                report["predictions"].append(summary); log({"phase": "prediction_completed", **summary})
                del result, operator, data
            except Exception as exc:
                failure = {"id": case["id"], "phase": "prediction", "error": f"{type(exc).__name__}: {exc}"}
                report["failures"].append(failure); log(failure)
                report["predictions"].append({"id": case["id"], "prediction_gate_passed": False, "failure": failure})
            gc.collect(); torch.cuda.empty_cache()
        if not all(row["prediction_gate_passed"] for row in report["predictions"]):
            raise RuntimeError("At least one frozen prediction failed its convergence/domain gate; references are deferred for this failed protocol")
        log({"phase": "both_predictions_frozen_before_reference_generation"})
        for case_index, case in enumerate(protocol["cases"]):
            if any(row["id"] == case["id"] for row in report["references"]): continue
            data = inputs(case)
            record, state = reference(data, protocol, reference_policy, case_index, device, lambda row: log({"id": case["id"], **row}))
            record["id"] = case["id"]
            atomic_save({**state, "report": record}, folder / f"{case['id']}.reference.pt")
            report["references"].append(record); log({"phase": "reference_completed", **record})
            if not record["converged"]: raise RuntimeError("Independent reference exhausted its fixed capacity; do not sweep settings")
            del data, state; gc.collect(); torch.cuda.empty_cache()
        for prediction, reference_record in zip(report["predictions"], report["references"]):
            error = (prediction["electronic_energy_hartree"]-reference_record["electronic_energy_hartree"])*KCAL
            report["results"].append({"id": prediction["id"], "energy_error_kcal_per_mol": error,
                                      "below_one": abs(error) < protocol["accuracy_tolerance_kcal_per_mol"],
                                      "variational_sign_valid": error >= -1e-5})
        contrast = report["results"][1]["energy_error_kcal_per_mol"]-report["results"][0]["energy_error_kcal_per_mol"]
        report["balanced_stretch_error_kcal_per_mol"] = contrast
        report["passed"] = (not report["failures"] and all(row["below_one"] and row["variational_sign_valid"] for row in report["results"])
                            and abs(contrast) < protocol["accuracy_tolerance_kcal_per_mol"])
        log({"phase": "accuracy_test_completed", "passed": report["passed"], "results": report["results"],
             "balanced_stretch_error_kcal_per_mol": contrast})
    except Exception as exc:
        report["failures"].append({"phase": "protocol", "error": f"{type(exc).__name__}: {exc}"}); log(report["failures"][-1])
        report["passed"] = False
    report.update({"status": "completed", "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
                   "goal_accuracy_achieved": False, "wtmad2_measured": False, "speed_work_authorized": False,
                   "general_out_of_sample_accuracy_established": False})
    output.write_text(json.dumps(report, indent=2)+"\n")
    if not report["passed"]: raise RuntimeError("BH larger-basis accuracy protocol failed; preserve all evidence and do not promote the model")


if __name__ == "__main__": main()
