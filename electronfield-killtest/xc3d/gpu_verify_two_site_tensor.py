"""Independent full-matrix verification of the two-site physical section."""
from .runtime import configure, torch_setup
ROOT = configure(gpu_only_compatible=True)

from datetime import datetime, timezone
import json
from pathlib import Path
import torch
from .gpu_purified_spin import PurifiedSpinState
from .gpu_spin_response import SpinChain, product_reference
from .gpu_tensor_gauge import canonical_at
from .gpu_two_site_tensor import pair_problem, split_pair, TwoSiteSpinBasis
from .gpu_fermion_mpo import FermionMPO
from .gpu_manybody_ops import molecular_words, dense_operator
from .gpu_four_particle import incremental_ritz
from .gpu_verify_purified_operator import random_hamiltonian, randomize, exhaustive_purification


@torch.no_grad()
def main():
    output = Path("research/two_site_tensor_checks_gpu_20261117.json")
    if output.exists(): raise RuntimeError("This complete two-site operator check already has a report")
    protocol = json.loads(Path("research/two_site_tensor_checks_protocol_20261117.json").read_text())
    device = torch_setup(protocol["seed"])
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "device": device, "protocol": protocol,
              "status": "running", "failures": []}
    try:
        h, eri = random_hamiltonian(protocol["sites"], device)
        operator = FermionMPO.molecular(h, eri)
        model = PurifiedSpinState(protocol["sites"], protocol["electrons"], channels=protocol["channels"], ancilla=1, device=device)
        randomize(model)
        source = SpinChain.from_state(model).normalized()
        state, _ = canonical_at(source, protocol["center"])
        problem = pair_problem(operator, state, protocol["center"])
        basis = problem["basis"]
        columns = []
        for index in range(basis.dimension):
            parameters = h.new_zeros(basis.dimension); parameters[index] = basis.inverse_norm[index]
            column, _ = split_pair(state, protocol["center"], basis, parameters)
            columns.append(exhaustive_purification(column.cores)[2][:, 0])
        wave = torch.stack(columns, dim=1)
        words, coefficients = molecular_words(h, eri, threshold=0.)
        bits = torch.arange(4**protocol["sites"], device=device)
        dense = dense_operator(words, coefficients, bits, protocol["sites"])
        expected = wave.T @ dense @ wave
        matrix = torch.stack([problem["action"](torch.eye(basis.dimension, device=device)[:, index]) for index in range(basis.dimension)], 1)
        difference = float((matrix-expected).abs().max())
        hermiticity = float((matrix-matrix.T).abs().max())
        metric_error = float((wave.T @ wave-torch.eye(basis.dimension, device=device)).norm())
        split, split_report = split_pair(state, protocol["center"], basis, problem["seed"]*basis.inverse_norm)
        original_wave = exhaustive_purification(state.cores)[2]
        split_wave = exhaustive_purification(split.cores)[2]
        split_error = float((original_wave-split_wave).norm())
        # The state/operator construction has finished before the independent
        # matrix eigenvalue is assessed; it never controls a setting or stop.
        result = incremental_ritz(problem["action"], problem["seed"], maximum=protocol["ritz_maximum_dimension"],
                                   tolerance=protocol["ritz_residual_tolerance_hartree"])
        energy_error = float((result["energy"]-torch.linalg.eigvalsh(expected)[0]).abs())
        hf = product_reference(4, 4, 0, device)
        hf_basis = TwoSiteSpinBasis(hf.labels[1], hf.labels[3], device)
        allowed = sorted(set((path[4], path[5]) for path in hf_basis.paths))
        old = sorted(set(label[:2] for label in hf.labels[2]))
        escaped = any(value not in old for value in allowed)
        valid = (max(difference, hermiticity) < protocol["operator_tolerance_hartree"]
                 and max(metric_error, split_error) < protocol["state_tolerance"] and result["converged"]
                 and energy_error < protocol["ritz_energy_tolerance_hartree"] and escaped)
        report["result"] = {"passed": valid, "physical_dimension": basis.dimension,
                            "maximum_operator_difference_hartree": difference, "hermiticity_error_hartree": hermiticity,
                            "independent_overlap_error": metric_error, "exact_split_state_error": split_error,
                            "local_ritz_residual_hartree": result["residual"], "independent_ritz_energy_error_hartree": energy_error,
                            "old_determinant_middle_sectors": old, "complete_middle_sectors": allowed,
                            "new_sectors_not_limited_to_seed": escaped, "merged_operator_blocks": problem["merged_operator_blocks"]}
        print(json.dumps(report["result"]), flush=True)
    except Exception as exc:
        report["failures"].append({"error": f"{type(exc).__name__}: {exc}"}); print(json.dumps(report["failures"][-1]), flush=True)
    report["status"] = "completed"; report["passed"] = not report["failures"] and report.get("result", {}).get("passed", False)
    report["peak_cuda_bytes"] = torch.cuda.max_memory_allocated()
    output.write_text(json.dumps(report, indent=2)+"\n")
    if not report["passed"]: raise RuntimeError("Two-site physical section failed its independent gate")


if __name__ == "__main__": main()
