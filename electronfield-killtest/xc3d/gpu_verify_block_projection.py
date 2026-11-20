"""Independent physical check of the new block projector, no reference rerun.

Compare one coherent sum of product Slaters through the local projection and
the independently implemented full-coordinate operator. This checks off-
diagonal couplings as well as diagonal energies before interpreting failures.
"""
from .runtime import configure, torch_setup
ROOT = configure(gpu_only_compatible=True)

import json
from pathlib import Path
import torch
from .gpu_coherent_blocks import project_two_blocks
from .gpu_coherent_orbitals import CoherentOrbitals
from .gpu_four_particle import FourParticleOperator


@torch.no_grad()
def main():
    device = torch_setup()
    path = Path("research/linked_block_accuracy_gpu_20261116.json")
    report = json.loads(path.read_text())
    cases = {case["id"]: case for case in report["protocol"]["cases"]}
    for row in report["results"]:
        data = torch.load(ROOT / "data/linked_block_accuracy" / f"{row['id']}.pt", map_location=device, weights_only=True)
        case = cases[row["id"]]
        split = data["atom_ao_slices"][case["first_block_atoms"]-1][1]
        projection = project_two_blocks(data["local_h"], data["local_eri"], split, data["nuclear"])
        blocks = projection["first"], projection["second"]
        physical_orbitals, projected_coefficients = [], []
        for _ in range(2):
            occupied = data["local_h"].new_zeros(len(data["local_h"]), 2)
            coefficients = []
            for index, block in enumerate(blocks):
                local = torch.randn(len(block["orbitals"][0]), device=device)
                local /= local.norm()
                occupied[:split, 0] = local if index == 0 else occupied[:split, 0]
                if index == 1:
                    occupied[split:, 1] = local
                overlaps = (block["orbitals"][:, :, 0] @ local).square()
                coefficients.append(block["transform"].T @ overlaps)
            physical_orbitals.append(occupied)
            projected_coefficients.append(torch.kron(coefficients[0], coefficients[1]))
        weights = data["local_h"].new_tensor([1., .37])
        coefficient = weights @ torch.stack(projected_coefficients)
        coefficient /= coefficient.norm()
        projected_energy = coefficient @ projection["matrix"] @ coefficient
        state = CoherentOrbitals.four_electron_vector(torch.stack(physical_orbitals), weights)
        operator = FourParticleOperator(data["local_h"], data["local_eri"], 4, data["nuclear"])
        physical_energy = state @ operator(state)
        discrepancy = float((projected_energy-physical_energy).abs())
        row["independent_projection_check_error_hartree"] = discrepancy
        row["projection_kernel_verified"] = discrepancy < 1e-9
        print(json.dumps({"id": row["id"], "projection_check_error_hartree": discrepancy,
                          "projection_kernel_verified": row["projection_kernel_verified"]}), flush=True)
    report["projection_checks_use_reference_energies"] = False
    report["projection_checks_recompute_references"] = False
    path.write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    main()
