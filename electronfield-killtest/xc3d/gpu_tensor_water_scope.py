"""New larger-basis tensor scope test, with no global Fock enumeration."""
from .runtime import configure, torch_setup
ROOT = configure(gpu_only_compatible=True)

import argparse
from datetime import datetime, timezone
import json
from math import comb
from pathlib import Path
import torch
from .gpu_chemistry_transfer import cuda_operators
from .gpu_fermion_mpo import FermionMPO
from .gpu_spin_response import SpinChain, closed_shell_reference, apply_physical, overlap, superposition
from .gpu_physical_response_metric import physical_product_response
from .gpu_manybody_ops import spin_words


def number_mpo(sites, device, squared=False):
    words = []
    for first in range(2*sites):
        if not squared:
            words.append(((first//2, first % 2, 1), (first//2, first % 2, 0), (-1, 0, 0), (-1, 0, 0)))
        else:
            for second in range(2*sites):
                words.append(((first//2, first % 2, 1), (first//2, first % 2, 0),
                              (second//2, second % 2, 1), (second//2, second % 2, 0)))
    return FermionMPO(torch.tensor(words, device=device), torch.ones(len(words), device=device), sites)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default="research/tensor_water_scope_protocol_20261117.json")
    parser.add_argument("--output", default="research/tensor_water_scope_gpu_20261117.json")
    args = parser.parse_args()
    protocol = json.loads(Path(args.protocol).read_text())
    if protocol.get("status", "").startswith("closed"):
        raise RuntimeError("This response selection hypothesis is closed; retain its report and use the physical metric protocol")
    output = Path(args.output)
    if output.exists(): raise RuntimeError("This coupled molecular scope test already has a report; do not restart it")
    gate = json.loads(Path("research/spin_response_capacity_completion_gpu_20261117.json").read_text())
    if not gate["passed"]: raise RuntimeError("Spin response physical gate has not passed")
    if protocol.get("response_metric") == "physical":
        metric_gate = json.loads(Path("research/physical_response_metric_checks_gpu_20261117.json").read_text())
        if not metric_gate["passed"]: raise RuntimeError("The physical response metric has not passed its independent gate")
    device = torch_setup(protocol["seed"])
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "device": device, "protocol": protocol,
              "status": "running", "history": [], "failures": []}
    folder = ROOT / "data/tensor_scope"
    folder.mkdir(exist_ok=True, parents=True)
    def log(phase, **values):
        row = {"phase": phase, **values}; report["history"].append(row)
        print(json.dumps(row), flush=True); output.write_text(json.dumps(report, indent=2)+"\n")
    try:
        destination = folder / f"{protocol['case']['id']}.inputs.pt"
        if destination.exists():
            data = torch.load(destination, map_location=device, weights_only=True)
            log("restored_cuda_inputs")
        else:
            log("cuda_inputs")
            data = cuda_operators(protocol["case"], protocol["basis"])
            torch.save({key: value.cpu() if isinstance(value, torch.Tensor) else value for key, value in data.items()}, destination)
        if not data["reference_status"]["converged"]: raise RuntimeError("The bounded RHF input did not converge; no retry campaign")
        number, sites = protocol["case"]["electrons"], len(data["h"])
        density = torch.diag(torch.cat((data["h"].new_full((number//2,), 2.), data["h"].new_zeros(sites-number//2))))
        coulomb = torch.einsum("ijkl,kl->ij", data["eri"], density)
        exchange = torch.einsum("ikjl,kl->ij", data["eri"], density)
        reference_energy = (density*(data["h"]+.5*coulomb-.25*exchange)).sum()
        shifted_h = data["h"]-reference_energy/number*torch.eye(sites, device=device)
        log("building_literal_hamiltonian", spatial_orbitals=sites, electrons=number,
            full_fixed_sz_dimension=comb(sites, number//2)**2, input_reference_status=data["reference_status"])
        mpo = FermionMPO.molecular(shifted_h, data["eri"])
        log("hamiltonian_ready", **mpo.summary())
        reference = closed_shell_reference(sites, number, device)
        shift_error = float(mpo.expectation(reference.cores).abs())
        if shift_error > protocol["operator_and_energy_identity_tolerance_hartree"]:
            raise RuntimeError("Literal fixed-N constant shift disagrees with independent Wick energy")
        log("reference_energy_identity", error_hartree=shift_error, total_input_energy_hartree=float(reference_energy+data["nuclear"]))
        def response_progress(row):
            row = dict(row)
            phase = row.pop("phase", "physical_response")
            log(phase, **row)
        if protocol.get("response_metric") == "physical":
            image, diagnostics = physical_product_response(mpo, reference,
                maximum_bond=protocol["response_maximum_bond"], singular_tolerance=protocol["relative_response_singular_tolerance"],
                frame_null_tolerance=protocol["right_frame_null_tolerance"], maximum_gram_bytes=protocol["maximum_gram_bytes"],
                maximum_frame_bytes=protocol["maximum_raw_response_bytes"], callback=response_progress)
            exact_norm = data["h"].new_tensor(diagnostics["exact_full_image_squared_norm"]).sqrt()
            if diagnostics["right_frame_triangle_error_bound"] > float(exact_norm)*protocol["maximum_relative_right_frame_error_bound"]:
                raise RuntimeError("Numerical right-frame compression failed its prespecified physical error bound")
            previous = torch.load(folder / f"{protocol['case']['id']}.state.pt", map_location=device, weights_only=True)
            old = SpinChain(previous["cores"], previous["labels"], number, 0).normalized()
            old_states = [reference, old]
            old_gram = torch.stack([torch.stack([overlap(first, second) for second in old_states]) for first in old_states])
            old_moments = torch.stack([mpo.contract(reference.cores, bra=first.cores) for first in old_states])
            old_retained = old_moments @ torch.linalg.solve(old_gram, old_moments)/exact_norm.square()
            log("physical_image_assessment", full_image_relative_error_estimate=diagnostics["full_image_relative_error_upper_estimate"],
                right_frame_triangle_error_bound=diagnostics["right_frame_triangle_error_bound"],
                original_zipup_span_retained_image_norm_fraction=float(old_retained))
            diagnostics["original_zipup_span_retained_image_norm_fraction"] = float(old_retained)
        else:
            image, diagnostics = apply_physical(mpo, reference, maximum_bond=protocol["response_maximum_bond"],
                                                relative_tolerance=protocol["relative_response_singular_tolerance"],
                                                maximum_raw_bytes=protocol["maximum_raw_response_bytes"], callback=response_progress)
        image = image.normalized()
        states = [reference, image]
        gram = torch.stack([torch.stack([overlap(first, second) for second in states]) for first in states])
        matrix = torch.stack([torch.stack([mpo.contract(second.cores, bra=first.cores) for second in states]) for first in states])
        hermiticity = float((matrix-matrix.T).norm())
        if hermiticity > protocol["operator_and_energy_identity_tolerance_hartree"]: raise RuntimeError("Molecular physical Hamiltonian lost Hermiticity")
        values, rotation = torch.linalg.eigh((gram+gram.T)/2)
        if float(values.min()) <= protocol["overlap_rank_threshold"]: raise RuntimeError("The response is linearly dependent on the retained input")
        orth = rotation*values.rsqrt()
        energies, coefficients = torch.linalg.eigh(orth.T @ ((matrix+matrix.T)/2) @ orth)
        coefficients = orth @ coefficients[:, 0]
        if protocol.get("response_metric") == "physical":
            moments = torch.stack([mpo.contract(reference.cores, bra=first.cores) for first in states])
            retained = moments @ torch.linalg.solve(gram, moments)/diagnostics["exact_full_image_squared_norm"]
            diagnostics["physical_metric_span_retained_image_norm_fraction"] = float(retained)
        candidate = superposition(states, coefficients).normalized()
        actual = mpo.expectation(candidate.cores)
        energy_identity_error = float((actual-energies[0]).abs())
        log("two_state_variational_prediction", total_energy_hartree=float(actual+reference_energy+data["nuclear"]),
            lowering_from_input_kcal=float(actual*627.509474), energy_reconstruction_error_hartree=energy_identity_error,
            response_capacity_limited=diagnostics["capacity_limited"])
        n = number_mpo(sites, device).expectation(candidate.cores)
        n2 = number_mpo(sites, device, squared=True).expectation(candidate.cores)
        sw, sc = spin_words(sites, device)
        spin = FermionMPO(sw, sc, sites).expectation(candidate.cores)
        variance = n2-2*number*n+number**2
        physical = max(float((n-number).abs()), float(variance.abs()), float(spin.abs())) < protocol["physical_tolerance"]
        valid = physical and energy_identity_error < protocol["operator_and_energy_identity_tolerance_hartree"] and float(actual) <= 1e-8
        result = {"physical_scope_passed": valid, "total_energy_hartree": float(actual+reference_energy+data["nuclear"]),
                  "input_total_energy_hartree": float(reference_energy+data["nuclear"]), "lowering_from_input_kcal": float(actual*627.509474),
                  "particle_number": float(n), "particle_number_variance": float(variance), "spin_squared": float(spin),
                  "energy_identity_error_hartree": energy_identity_error, "hermiticity_error_hartree": hermiticity,
                  "state_bond_dimensions": [value.shape[-1] for value in candidate.cores],
                  "full_fixed_sz_dimension": comb(sites, number//2)**2,
                  "full_occupation_vector_allocated": False, "new_reference_generated_or_loaded": False,
                  "full_stationarity_certified": False, "chemical_accuracy_measured": False,
                  "xc_functional_evaluated": False, "wtmad2_measured": False, "goal_accuracy_achieved": False,
                  "response": diagnostics}
        report["result"] = result
        torch.save({"cores": [value.cpu() for value in candidate.cores], "labels": candidate.labels,
                    "electrons": number, "twice_spin": 0, "result": result}, folder / f"{protocol['case']['id']}{protocol.get('state_suffix', '')}.state.pt")
        log("physical_checks", particle_number=float(n), number_variance=float(variance), spin_squared=float(spin), passed=valid)
    except Exception as exc:
        report["failures"].append({"error": f"{type(exc).__name__}: {exc}"}); log("failure", **report["failures"][-1])
    report["status"] = "completed"
    report["passed"] = not report["failures"] and report.get("result", {}).get("physical_scope_passed", False)
    report["peak_cuda_bytes"] = torch.cuda.max_memory_allocated()
    output.write_text(json.dumps(report, indent=2)+"\n")
    if not report["passed"]: raise RuntimeError("Tensor molecular scope test failed; no accuracy claim is accepted")


if __name__ == "__main__": main()
