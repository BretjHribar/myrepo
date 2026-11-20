"""Direct dual fixed point for the Coulomb-moment functional.

Solves the interacting potential condition lambda=v_ext, then inverts its
moments to a noninteracting ensemble. This is an expensive implicit physical
functional solve, not an amortized predictor or a speed breakthrough.
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
from .gpu_density_completion import complete_molecular
from .gpu_ensemble_dual import ensemble_dual
from .gpu_canonical_density import canonical_lift
from .gpu_manybody_ops import spin_words, dense_operator

KCAL = 627.509474


@torch.no_grad()
def run_case(case, protocol, device):
    started = time.perf_counter()
    data = prepare(case, protocol["basis"], device)
    sites, electrons = len(data["kinetic"]), case["electrons"]
    external = torch.einsum("a,aij->ij", data["external_coefficients"], data["q_matrices"])
    physical = data["universal"]+external
    physical += torch.eye(len(physical), device=device)*(data["external_constant"]*electrons+data["nuclear"])
    interacting = complete_molecular(physical, data["seed"], maximum=protocol["maximum_subspace_dimension"],
                                     tolerance=protocol["stationarity_tolerance"])
    state = interacting["state"]
    target = torch.einsum("i,aij,j->a", state, data["q_matrices"], state)
    beta = 2*sites*math.log(2)/protocol["kinetic_entropy_tolerance_hartree"]
    kinetic_operators = torch.cat((torch.eye(sites, device=device)[None], data["moments"]))
    kinetic_target = torch.cat((target.new_tensor([electrons]), target))
    spectrum = torch.linalg.eigvalsh(data["kinetic"])
    initial = torch.zeros_like(kinetic_target)
    initial[0] = -(spectrum[electrons//2-1]+spectrum[electrons//2])/2
    kinetic = ensemble_dual(data["kinetic"], kinetic_operators, kinetic_target, beta,
                            statistics="onebody", initial=initial)
    determinant = (kinetic["vectors"]*kinetic["population"]) @ kinetic["vectors"].T
    q_output = torch.einsum("ij,aij->a", determinant, data["moments"])
    universal = state @ data["universal"] @ state
    t_input = (determinant*data["kinetic"]).sum()
    hartree = .5*torch.einsum("ij,ijkl,kl->", determinant, data["eri"], determinant)
    xc = universal-kinetic["dual"]-hartree
    total = t_input+hartree+xc+data["external_coefficients"] @ q_output
    total += data["external_constant"]*electrons+data["nuclear"]
    lift = canonical_lift(kinetic["vectors"], kinetic["population"], electrons, data["bits"])
    lifted_q = torch.einsum("b,bi,aij,bj->a", lift["weights"], lift["vectors"], data["q_matrices"], lift["vectors"])
    sw, sc = spin_words(sites, device)
    spin_matrix = dense_operator(sw, sc, data["bits"], sites)
    spin = state @ spin_matrix @ state
    lifted_spin = torch.einsum("b,bi,ij,bj->", lift["weights"], lift["vectors"], spin_matrix, lift["vectors"])
    vxc = torch.einsum("a,aij->ij", kinetic["potential"][1:]-data["external_coefficients"], data["moments"])
    fock = data["h"]+vxc  # J and -J in v_xc cancel exactly in this model.
    ks_residual = (fock @ determinant-determinant @ fock).norm()
    valid = (interacting["converged"] and kinetic["residual"] < 1e-8 and float(spin.abs()) < 1e-7
             and lift["occupation_residual"] < 1e-8 and float((lifted_q-target).norm()) < 1e-8
             and float(ks_residual) < 1e-8)
    # Reference is accessed/generated only after all model predictions stop.
    if "reference_report" in case:
        reference_data = json.loads(Path(case["reference_report"]).read_text())
        exact = reference_data["reference_total_hartree"]
        new_reference = False
    else:
        exact = float(torch.linalg.eigvalsh(physical)[0])
        new_reference = True
    result = {"id": case["id"], "electrons": electrons, "spatial_orbitals": sites, "sector_dimension": len(physical),
              "prediction_valid": valid, "total_energy_hartree": float(total) if valid else None,
              "reference_energy_hartree": exact, "energy_error_kcal": float((total-exact)*KCAL) if valid else None,
              "rejected_raw_total_hartree": None if valid else float(total),
              "interacting_converged": interacting["converged"], "interacting_residual": interacting["residual"],
              "interacting_dimension": interacting["dimension"], "kinetic_density_number_residual": kinetic["residual"],
              "density_moment_self_consistency_residual": float((target-q_output).norm()),
              "ks_commutator_residual": float(ks_residual), "spin_squared": float(spin),
              "canonical_lift_components": len(lift["weights"]), "canonical_lift_spin_squared": float(lifted_spin),
              "canonical_lift_moment_residual": float((lifted_q-target).norm()),
              "canonical_lift_occupation_residual": lift["occupation_residual"],
              "kinetic_completion_hartree": float(t_input-kinetic["dual"]),
              "external_span_error_hartree": data["external_span_error_hartree"],
              "new_reference_generated": new_reference, "reference_energy_fitted": False,
              "input_reference_status": data["reference_status"], "seconds": time.perf_counter()-started,
              "interacting_history": interacting["history"], "kinetic_history": kinetic["history"]}
    torch.save({"state": state.cpu(), "noninteracting_density": determinant.cpu(), "kinetic_potential": kinetic["potential"].cpu(),
                "case": case, "report": result}, Path("research") / f"{case['id']}_dual_fixedpoint.pt")
    print(json.dumps({key: value for key, value in result.items() if not key.endswith("history")}), flush=True)
    return result


def main():
    device = torch_setup()
    protocol = json.loads(Path("research/dual_fixedpoint_protocol_20261116.json").read_text())
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "device": device, "protocol": protocol,
              "results": [], "failures": [], "scope": "Direct implicit physical functional fixed point; dense complete minimal bases, no amortized prediction or speed claim"}
    destination = Path("research/dual_fixedpoint_transfer_gpu_20261116.json")
    for case in protocol["cases"]:
        try:
            report["results"].append(run_case(case, protocol, device))
        except Exception as exc:
            report["failures"].append({"id": case["id"], "error": f"{type(exc).__name__}: {exc}"})
            print(json.dumps(report["failures"][-1]), flush=True)
        destination.write_text(json.dumps(report, indent=2)+"\n")
    report["peak_cuda_bytes"] = torch.cuda.max_memory_allocated()
    destination.write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    main()
