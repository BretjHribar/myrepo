"""Independent operator and scalar/derivative checks for local tensor XC."""
from .runtime import configure, torch_setup
ROOT = configure(gpu_only_compatible=True)

from datetime import datetime, timezone
import json
from pathlib import Path
import torch
from .gpu_purified_spin import PurifiedSpinState
from .gpu_spin_response import SpinChain
from .gpu_tensor_density import density_matrix
from .tensor_auxiliary_model import UniversalTensorOperators
from .gpu_manybody_ops import molecular_words, dense_operator
from .gpu_verify_purified_operator import random_hamiltonian, randomize, exhaustive_purification


def symmetric_frame(sites, device):
    identity = torch.eye(sites, device=device)
    result = [identity/identity.new_tensor(sites).sqrt()]
    for first in range(sites):
        for second in range(first+1, sites):
            matrix = torch.zeros_like(identity); matrix[first, second] = matrix[second, first] = 1
            result.append(matrix/matrix.new_tensor(2).sqrt())
    for count in range(1, sites):
        diagonal = identity.new_zeros(sites)
        diagonal[:count] = 1; diagonal[count] = -count
        result.append(torch.diag(diagonal)/diagonal.new_tensor(count*(count+1)).sqrt())
    return torch.stack(result)


@torch.no_grad()
def main():
    output = Path("research/local_tensor_functional_checks_gpu_20261117.json")
    if output.exists(): raise RuntimeError("This new conditional-functional check already has a report")
    protocol = json.loads(Path("research/local_tensor_functional_protocol_20261117.json").read_text())
    device = torch_setup(protocol["seed"])
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "device": device, "protocol": protocol,
              "results": [], "failures": [], "status": "running"}
    try:
        sites, center = protocol["sites"], protocol["center"]
        _, eri = random_hamiltonian(sites, device)
        raw = torch.randn(sites, sites, device=device)
        kinetic = raw @ raw.T/8+torch.eye(sites, device=device)*.25
        operators = symmetric_frame(sites, device)
        words, coefficients = molecular_words(kinetic, eri, threshold=0.)
        bits = torch.arange(4**sites, device=device)
        dense = dense_operator(words, coefficients, bits, sites)
        density_operators = []
        for field in operators:
            words, coefficients = molecular_words(field, torch.zeros_like(eri), threshold=0.)
            density_operators.append(dense_operator(words, coefficients, bits, sites))
        for case in protocol["state_cases"]:
            state = PurifiedSpinState(sites, case["electrons"], case["twice_spin"], channels=protocol["channels"], ancilla=protocol["ancilla"], device=device)
            randomize(state)
            context = SpinChain.from_state(state).normalized()
            universal = {"kinetic": kinetic, "eri": eri, "operators": operators,
                         "electrons": case["electrons"], "twice_spin": case["twice_spin"]}
            rejects_external = False
            try:
                UniversalTensorOperators({**universal, "h": kinetic})
            except ValueError:
                rejects_external = True
            domain = UniversalTensorOperators(universal)
            section = domain.section(context, center, protocol)
            basis, transform = section.space["basis"], section.space["transform"]
            waves = []
            for column in range(transform.shape[1]):
                cores = list(context.cores); cores[center] = basis.core(transform[:, column:column+1])
                waves.append(exhaustive_purification(cores)[2])
            wave = torch.stack(waves, dim=-1)
            gram = torch.einsum("pae,paf->ef", wave, wave)
            applied = (dense @ wave.flatten(1)).reshape_as(wave)
            expected_h = torch.einsum("pae,paf->ef", wave, applied)
            expected_q = torch.stack([torch.einsum("pae,paf->ef", wave, (field @ wave.flatten(1)).reshape_as(wave)) for field in density_operators[1:]])
            h_error = float((section.matrix-expected_h).abs().max())
            q_error = float((section.reduced-expected_q).abs().max())
            metric_error = float((gram-torch.eye(len(gram), device=device)).norm())
            if max(h_error, q_error, metric_error) > protocol["operator_identity_tolerance"]:
                raise RuntimeError(f"Local literal operator mismatch: H={h_error}, Q={q_error}, G={metric_error}")
            initial_density = density_matrix(context)
            initial_moments = torch.einsum("aij,ij->a", domain.moments, initial_density)
            direction = torch.randn(len(domain.moments), device=device); direction /= direction.norm()
            auxiliary = initial_moments+.05*direction
            solved = section.evaluate(auxiliary)
            if not solved["converged"]: raise RuntimeError(f"Conditional search failed its bounded gap criterion: {solved['gap']}")
            installed, installation = section.install(solved)
            derivative = None
            if not case["twice_spin"]:
                kinetic_point = section.kinetic(auxiliary, initial_density)
                central = section.xc(auxiliary, solved, kinetic_point)
                values = []
                for sign in (-1, 1):
                    changed = auxiliary+sign*protocol["derivative_step"]*direction
                    interacting = section.evaluate(changed, initial=solved["state"])
                    kinetic_changed = section.kinetic(changed, kinetic_point["state"])
                    values.append(section.xc(changed, interacting, kinetic_changed)["energy"])
                finite = (values[1]-values[0])/(2*protocol["derivative_step"])
                analytic = central["gradient"] @ direction
                error = float((finite-analytic).abs())
                derivative = {"finite_difference": float(finite), "analytic": float(analytic),
                              "absolute_difference_hartree": error, "passed": error < protocol["derivative_tolerance_hartree"],
                              "surrounding_context_refitted": False}
                if not derivative["passed"]: raise RuntimeError(f"Conditional XC scalar derivative mismatch: {error}")
            row = {"case": case, "passed": True, "external_key_rejected": rejects_external,
                   "physical_local_dimension": section.space["physical_dimension"],
                   "reduced_coordinate_count": section.space["reduced_coordinate_count"],
                   "literal_hamiltonian_error_hartree": h_error, "density_operator_error": q_error,
                   "independent_overlap_metric_error": metric_error, "input_retention_error": section.space["input_retention_error"],
                   "local_conditional_gap_hartree": solved["gap"], "installation": installation,
                   "xc_derivative": derivative, "global_tensor_stationarity_certified": False,
                   "chemical_accuracy_measured": False}
            report["results"].append(row); print(json.dumps(row), flush=True)
            output.write_text(json.dumps(report, indent=2)+"\n")
    except Exception as exc:
        report["failures"].append({"error": f"{type(exc).__name__}: {exc}"}); print(json.dumps(report["failures"][-1]), flush=True)
    report["status"] = "completed"
    report["passed"] = len(report["results"]) == len(protocol["state_cases"]) and not report["failures"] and all(row["passed"] and row["external_key_rejected"] for row in report["results"])
    report["peak_cuda_bytes"] = torch.cuda.max_memory_allocated()
    output.write_text(json.dumps(report, indent=2)+"\n")
    if not report["passed"]: raise RuntimeError("Local tensor functional is not approved for molecular use")


if __name__ == "__main__": main()
