"""Repair the canonical lift certificate using saved model predictions only.

No model energy solve, molecular integral or reference generation is repeated.
"""
from .runtime import configure, torch_setup
ROOT = configure(gpu_only_compatible=True)

import json
from pathlib import Path
import torch
from .gpu_canonical_density import canonical_lift
from .gpu_manybody_ops import spin_words, dense_operator


@torch.no_grad()
def main():
    device = torch_setup()
    destination = Path("research/dual_fixedpoint_transfer_gpu_20261116.json")
    report = json.loads(destination.read_text())
    report["canonical_lift_correction"] = "Fixed systematic-sampling breakpoints to (-cumulative occupation) modulo 1; reused saved states/densities, no energy/reference recomputation"
    for row in report["results"]:
        checkpoint_path = Path("research") / f"{row['id']}_dual_fixedpoint.pt"
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        data = torch.load(ROOT / "data/coulomb_moment_functional" / f"{row['id']}.pt", map_location=device, weights_only=True)
        occupations, orbitals = torch.linalg.eigh(checkpoint["noninteracting_density"])
        lift = canonical_lift(orbitals, occupations, row["electrons"], data["bits"])
        state = checkpoint["state"]
        target = torch.einsum("i,aij,j->a", state, data["q_matrices"], state)
        moments = torch.einsum("b,bi,aij,bj->a", lift["weights"], lift["vectors"], data["q_matrices"], lift["vectors"])
        sw, sc = spin_words(row["spatial_orbitals"], device)
        spin_matrix = dense_operator(sw, sc, data["bits"], row["spatial_orbitals"])
        spin = torch.einsum("b,bi,ij,bj->", lift["weights"], lift["vectors"], spin_matrix, lift["vectors"])
        row["previous_invalid_lift_moment_residual"] = row["canonical_lift_moment_residual"]
        row["previous_invalid_lift_occupation_residual"] = row["canonical_lift_occupation_residual"]
        row["canonical_lift_components"] = len(lift["weights"])
        row["canonical_lift_moment_residual"] = float((moments-target).norm())
        row["canonical_lift_occupation_residual"] = lift["occupation_residual"]
        row["canonical_lift_spin_squared"] = float(spin)
        row["canonical_lift_number_rounding_allowance"] = lift["number_rounding_allowance"]
        row["prediction_valid"] = (row["interacting_converged"] and row["kinetic_density_number_residual"] < 1e-8
                                    and row["canonical_lift_moment_residual"] < 1e-8
                                    and row["canonical_lift_occupation_residual"] < 1e-8
                                    and abs(row["spin_squared"]) < 1e-7 and float(spin.abs()) < 1e-7
                                    and row["ks_commutator_residual"] < 1e-8)
        checkpoint["report"] = row
        checkpoint["canonical_lift_weights"] = lift["weights"].cpu()
        checkpoint["canonical_lift_vectors"] = lift["vectors"].cpu()
        torch.save(checkpoint, checkpoint_path)
        print(json.dumps({key: value for key, value in row.items() if not key.endswith("history")}), flush=True)
    report["peak_cuda_bytes_lift_correction"] = torch.cuda.max_memory_allocated()
    destination.write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    main()
