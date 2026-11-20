"""Independent physical validation of the non-quartic-word operator path."""
from .runtime import configure, torch_setup
ROOT = configure(gpu_only_compatible=True)

from datetime import datetime, timezone
import json
from pathlib import Path
import torch
from .gpu_density_square_operator import OneBodyMPO, DensitySquareOperator, ProductMPO
from .gpu_fermion_mpo import FermionMPO
from .gpu_manybody_ops import molecular_words, dense_operator
from .gpu_verify_purified_operator import diagnostic_matrix
from .gpu_spin_response import product_reference
from .gpu_tensor_gauge import canonical_at
from .gpu_two_site_tensor import pair_problem, split_pair


def local_matrix(problem):
    identity = torch.eye(problem["basis"].dimension, device=problem["seed"].device)
    return torch.stack([problem["action"](column) for column in identity.T], 1)


@torch.no_grad()
def main():
    output = Path("research/density_square_operator_checks_gpu_20261117.json")
    if output.exists(): raise RuntimeError("Do not repeat a terminal operator check")
    protocol = json.loads(Path("research/density_square_operator_protocol_20261117.json").read_text())
    device = torch_setup(protocol["seed"])
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "device": device, "protocol": protocol,
              "status": "running", "checks": [], "failures": []}
    def log(value):
        print(json.dumps(value), flush=True)
        output.write_text(json.dumps(report, indent=2)+"\n")
    def check(name, error, tolerance):
        report["checks"].append({"name": name, "maximum_absolute_error_hartree": float(error), "passed": bool(error < tolerance)})
        log(report["checks"][-1])
    try:
        sites = protocol["small_sites"]
        h = torch.randn(sites, sites, device=device); h = (h+h.T)/2
        factors = .2*torch.randn(protocol["small_coulomb_factors"], sites, sites, device=device)
        factors = (factors+factors.transpose(-1, -2))/2
        eri = torch.einsum("pij,pkl->ijkl", factors, factors)
        # This full quartic construction exists only in the independent tiny
        # reference. The new operator never constructs these words or ERIs.
        words, coefficients = molecular_words(h, eri, threshold=0.)
        bits = torch.arange(1 << (2*sites), device=device)
        independent = dense_operator(words, coefficients, bits, sites)
        operator = DensitySquareOperator(h, factors, protocol["maximum_product_storage_bytes"])
        represented = independent.new_zeros(independent.shape)
        for weight, term in operator.terms(): represented += weight*diagnostic_matrix(term)
        # diagnostic_matrix prepends each newest site, so site zero already
        # occupies the least significant bits, matching dense_operator.
        check("all_number_and_spin_sectors_full_fermion_matrix", (represented-independent).abs().max(), protocol["matrix_tolerance_hartree"])
        zero_eri = torch.zeros_like(eri)
        words_one, coefficients_one = molecular_words(h, zero_eri, threshold=0.)
        independent_one = dense_operator(words_one, coefficients_one, bits, sites)
        direct_one = diagnostic_matrix(OneBodyMPO(h))
        check("direct_one_body_jordan_wigner_signs", (direct_one-independent_one).abs().max(), protocol["matrix_tolerance_hartree"])
        literal = FermionMPO.molecular(h, eri)
        state, _ = canonical_at(product_reference(sites, protocol["small_electrons"], 0, device), protocol["small_center"])
        source = pair_problem(literal, state, protocol["small_center"], protocol["maximum_environment_bytes"])
        parameters = torch.randn(source["basis"].dimension, device=device)
        parameters /= parameters.norm()
        correlated, _ = split_pair(state, protocol["small_center"], source["basis"], parameters*source["basis"].inverse_norm)
        correlated = correlated.normalized()
        check("correlated_tensor_literal_energy", (operator.expectation(correlated.cores)-literal.expectation(correlated.cores)).abs(), protocol["matrix_tolerance_hartree"])
        state, _ = canonical_at(correlated, protocol["small_center"])
        source = pair_problem(literal, state, protocol["small_center"], protocol["maximum_environment_bytes"])
        changed = operator.problem(operator, state, protocol["small_center"], protocol["maximum_environment_bytes"])
        exact_local, new_local = local_matrix(source), local_matrix(changed)
        check("complete_local_section_action", (new_local-exact_local).abs().max(), protocol["matrix_tolerance_hartree"])
        check("local_section_hermiticity", (new_local-new_local.T).abs().max(), protocol["matrix_tolerance_hartree"])
        # Explicitly retain the tempting wrong implementation as a
        # counterexample: (P E P)^2 omits excursions outside the section.
        wrong = local_matrix(pair_problem(operator.one_body, state, protocol["small_center"], protocol["maximum_environment_bytes"]))
        for factor in factors:
            projected = local_matrix(pair_problem(OneBodyMPO(factor), state, protocol["small_center"], protocol["maximum_environment_bytes"]))
            wrong += .5*(projected @ projected)
        difference = (wrong-exact_local).abs().max()
        report["checks"].append({"name": "project_then_square_counterexample_detected", "passed": bool(difference > protocol["minimum_wrong_projected_square_difference_hartree"]),
                                  "wrong_operator_difference_hartree": float(difference)})
        log(report["checks"][-1])
        if not all(row["passed"] for row in report["checks"]):
            raise RuntimeError("Do not start the larger check after a failed independent physical comparison")
        log({"phase": "larger_provided_product_operator_check", "sites": protocol["large_sites"], "electrons": protocol["large_electrons"]})
        sites = protocol["large_sites"]
        h = torch.randn(sites, sites, device=device); h = (h+h.T)/2
        factors = .02*torch.randn(protocol["large_coulomb_factors"], sites, sites, device=device)
        factors = (factors+factors.transpose(-1, -2))/2
        operator = DensitySquareOperator(h, factors, protocol["maximum_product_storage_bytes"])
        cores = product_reference(sites, protocol["large_electrons"], 0, device).cores
        occupied = protocol["large_electrons"]//2
        expected = 2*h.diagonal()[:occupied].sum()
        for factor in factors:
            block = factor[:occupied, :occupied]
            expected += 2*block.trace().square()-(block*block.T).sum()
        actual = operator.expectation(cores)
        check("larger_product_energy_vs_independent_wick_contraction", (actual-expected).abs(), protocol["large_product_energy_tolerance_hartree"])
        density = OneBodyMPO(factors[0])
        product = ProductMPO(density, density, protocol["maximum_product_storage_bytes"])
        report["larger_operator_summary"] = {"sites": sites, "provided_product_electrons": protocol["large_electrons"],
            "density_operator": density.summary(), "squared_density_operator": product.summary(),
            "four_index_eri_allocated": False, "quartic_word_list_enumerated": False,
            "coupled_large_system_accuracy_measured": False}
    except Exception as exc:
        report["failures"].append({"error": f"{type(exc).__name__}: {exc}"}); log(report["failures"][-1])
    report.update(status="completed", passed=not report["failures"] and len(report["checks"]) == 7 and all(row["passed"] for row in report["checks"]),
                  peak_cuda_bytes=torch.cuda.max_memory_allocated(), actual_gmtkn55_model_predictions=0,
                  wtmad2_measured=False)
    report["validation_basis_order_correction"] = "Original invalid reference permutation preserved in density_square_operator_checks_gpu_20261117_basis_order_invalid.json; only test ordering corrected, model/seed/tolerances unchanged"
    output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({key: report[key] for key in ("passed", "peak_cuda_bytes", "actual_gmtkn55_model_predictions")}), flush=True)
    if not report["passed"]: raise RuntimeError("New physical operator representation failed its fixed independent checks")


if __name__ == "__main__": main()
