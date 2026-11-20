"""Changed ensemble construction after the pure-state water failure.

Reuses the new water operators and its already generated reference label.
There is no reference-energy loss and no label-driven rank/iteration sweep.
"""
from .runtime import configure, torch_setup
ROOT = configure(gpu_only_compatible=True)

from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time
import torch
from .gpu_coulomb_functional import prepare
from .gpu_ensemble_dual import ensemble_dual, ensemble_completion
from .gpu_manybody_ops import spin_words, dense_operator

KCAL = 627.509474


def main():
    device = torch_setup()
    started = time.perf_counter()
    protocol = json.loads(Path("research/ensemble_coulomb_protocol_20261116.json").read_text())
    base = json.loads(Path("research/coulomb_functional_protocol_20261116.json").read_text())
    case = base["case"]
    data = prepare(case, base["basis"], device)
    target = data["input_target"]
    sites, electrons = len(data["kinetic"]), case["electrons"]
    beta = max(math.log(len(data["universal"])), 2*sites*math.log(2))/protocol["entropy_energy_tolerance_hartree"]
    kinetic_operators = torch.cat((torch.eye(sites, device=device)[None], data["moments"]))
    kinetic_target = torch.cat((target.new_tensor([electrons]), target))
    spectrum = torch.linalg.eigvalsh(data["kinetic"])
    initial = torch.zeros_like(kinetic_target)
    initial[0] = -(spectrum[electrons//2-1]+spectrum[electrons//2])/2
    kinetic = ensemble_dual(data["kinetic"], kinetic_operators, kinetic_target, beta,
                            statistics="onebody", initial=initial)
    print(json.dumps({"phase": "ensemble_kinetic", "residual": kinetic["residual"],
                      "history": kinetic["history"]}), flush=True)
    interacting = ensemble_completion(data["universal"], data["q_matrices"], target, data["seed"], beta,
                                        maximum=protocol["maximum_subspace_dimension"],
                                        tolerance=protocol["stationarity_tolerance"])
    determinant = data["input_determinant"]
    kinetic_input = (determinant*data["kinetic"]).sum()
    hartree = .5*torch.einsum("ij,ijkl,kl->", determinant, data["eri"], determinant)
    xc = interacting["functional"]-kinetic["dual"]-hartree
    total = kinetic_input+hartree+xc+data["external_coefficients"] @ target
    total += data["external_constant"]*electrons+data["nuclear"]
    actual_moments = torch.einsum("ij,aij->a", interacting["density_operator"], data["q_matrices"])
    allowance = (data["external_coefficients"] @ (actual_moments-target)).abs()
    sw, sc = spin_words(sites, device)
    spin_matrix = dense_operator(sw, sc, data["bits"], sites)
    spin = torch.trace(interacting["density_operator"] @ spin_matrix)
    trace = interacting["density_operator"].trace()
    energy_valid = interacting["density_residual"] < 1e-8 and kinetic["residual"] < 1e-8
    potential_valid = energy_valid and interacting["stationarity_residual"] < protocol["stationarity_tolerance"]
    # Reuse the already generated same-operator label, read only after prediction.
    reference_report = json.loads(Path("research/coulomb_functional_water_gpu_20261116.json").read_text())
    reference = reference_report["reference_total_hartree"]
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "device": device,
              "protocol": protocol, "case": case, "basis": base["basis"],
              "scope": "First ensemble Coulomb/kinetic-completed functional, one development water density, complete minimal basis",
              "energy_prediction_valid": energy_valid, "potential_stationarity_valid": potential_valid,
              "interacting_status": interacting["status"], "density_moment_residual": interacting["density_residual"],
              "state_stationarity_residual": interacting["stationarity_residual"], "kinetic_density_number_residual": kinetic["residual"],
              "ensemble_trace": float(trace), "particle_number": float(trace*electrons), "spin_squared": float(spin),
              "beta": beta, "entropy_energy_allowance_hartree": interacting["entropy_allowance"],
              "external_moment_energy_allowance_kcal": float(allowance*KCAL),
              "external_span_error_hartree": data["external_span_error_hartree"],
              "total_energy_hartree": float(total) if energy_valid else None,
              "total_error_kcal": float((total-reference)*KCAL) if energy_valid else None,
              "rejected_raw_total_hartree": None if energy_valid else float(total),
              "kinetic_completion_hartree": float(kinetic_input-kinetic["dual"]),
              "moment_xc_potential_coefficients": (-interacting["potential"]+kinetic["potential"][1:]).tolist(),
              "kinetic_dual_history": kinetic["history"], "interacting_history": interacting["history"],
              "interacting_dual_history": interacting["dual_history"], "reference_energy_fitted": False,
              "reference_recomputed": False, "seconds": time.perf_counter()-started,
              "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
              "limitations": ["One development density, not broad out-of-sample evidence", "Complete minimal basis only",
                              "No model-driven self-consistency or nuclear-frame derivatives",
                              "Entropy regularization has a stated finite energy tolerance",
                              "Dense occupation action; size consistency of global truncation unresolved"]}
    destination = Path("research/ensemble_coulomb_water_gpu_20261116.json")
    destination.write_text(json.dumps(report, indent=2)+"\n")
    torch.save({"density_operator": interacting["density_operator"].cpu(),
                "interacting_potential": interacting["potential"].cpu(), "kinetic_potential": kinetic["potential"].cpu(),
                "report": report}, destination.with_suffix(".pt"))
    print(json.dumps({key: value for key, value in report.items() if not key.endswith("history") and key != "protocol"}), flush=True)


if __name__ == "__main__":
    main()
