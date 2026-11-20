"""Structural accuracy audit of the coherent state's fragment closure.

For paired determinants the alpha/beta coefficient matrix has rank <=K.
A product of independent fragments multiplies their Schmidt ranks. Compute
this certificate from saved states, without fitting or reference regeneration.
"""
from .runtime import configure, torch_setup
ROOT = configure(gpu_only_compatible=True)

from datetime import datetime, timezone
from itertools import combinations
import json
from pathlib import Path
import torch
from .gpu_coherent_orbitals import CofactorKernel


@torch.no_grad()
def main():
    device = torch_setup()
    name = "LiH_neutral_1.595"
    folder = ROOT / "data/four_particle_ccpvdz"
    checkpoint = torch.load(folder / f"{name}.fast_coherent.pt", map_location=device, weights_only=True)
    data = torch.load(folder / f"{name}.pt", map_location=device, weights_only=True)
    orbitals, coefficients = checkpoint["orbitals"], checkpoint["coefficients"]
    pairs = torch.tensor(list(combinations(range(len(data["h"])), 2)), device=device)
    wedge = (orbitals[:, pairs[:, 0], 0]*orbitals[:, pairs[:, 1], 1]
             -orbitals[:, pairs[:, 1], 0]*orbitals[:, pairs[:, 0], 1])
    amplitude = torch.einsum("a,ai,aj->ij", coefficients, wedge, wedge)
    amplitude = amplitude/amplitude.norm()
    singular = torch.linalg.svdvals(amplitude)
    rank = int((singular > singular[0]*1e-10).sum())
    budget = len(orbitals)
    squared = singular.square()
    largest = squared.new_ones(1)
    fidelity_bounds = []
    for count in range(1, 9):
        largest = (largest[:, None]*squared[None]).flatten().topk(min(budget, largest.numel()*len(squared))).values
        fidelity_bounds.append({"identical_fragments": count, "global_determinant_budget": budget,
                                "maximum_possible_squared_overlap": float(largest.sum())})
    kernel = CofactorKernel(data["h"], data["eri"], 2)
    matrix, metric = kernel.matrices(orbitals)
    norm = coefficients @ metric @ coefficients
    local = coefficients @ matrix @ coefficients/norm
    product_coefficients = torch.kron(coefficients, coefficients)
    product_metric = torch.kron(metric.contiguous(), metric.contiguous())
    product_matrix = torch.kron(matrix.contiguous(), metric.contiguous())+torch.kron(metric.contiguous(), matrix.contiguous())
    product_energy = product_coefficients @ product_matrix @ product_coefficients/(product_coefficients @ product_metric @ product_coefficients)
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "device": device,
              "source": name, "basis": data["basis"], "local_determinants": budget,
              "local_spin_schmidt_rank_at_relative_1e_minus10": rank,
              "two_fragment_product_rank": rank**2,
              "fixed_global_budget_can_represent_product": rank**2 <= budget,
              "global_coherent_family_is_fragment_closed": rank**2 <= budget,
              "product_representation": "Factored local states; no global determinant enumeration needed for storage",
              "two_fragment_energy_additivity_error_hartree": float((product_energy-2*local).abs()),
              "fidelity_upper_bounds_for_fixed_global_budget": fidelity_bounds,
              "reference_recomputed": False, "optimization_run": False,
              "interpretation": "Each global paired determinant has alpha/beta Schmidt rank one. A rank-K sum cannot represent a fragment product of larger rank. Fidelity bounds are not energy-error bounds.",
              "decision": "Use coherent states as local blocks; a fixed global determinant budget is not an acceptable final composition rule",
              "peak_cuda_bytes": torch.cuda.max_memory_allocated()}
    Path("research/coherent_composition_accuracy_gpu_20261116.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
