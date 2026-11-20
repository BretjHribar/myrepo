"""Evaluate the rejected saved candidate with the full physical Hamiltonian."""
from .runtime import configure, torch_setup
ROOT = configure(gpu_only_compatible=True)

from datetime import datetime, timezone
import json
from pathlib import Path
import torch
from .gpu_fock_blocks import SingletProjector
from .gpu_charge_spin_accuracy import PhysicalAction


@torch.no_grad()
def main():
    device = torch_setup()
    name = "BeH2_631g_1.65_new"
    data = torch.load(ROOT / "data/charge_spin_accuracy" / f"{name}.pt", map_location=device, weights_only=True)
    saved = torch.load(ROOT / "data/auxiliary_completed" / f"{name}.pt", map_location=device, weights_only=True)
    basis, ensemble = saved["basis"], saved["reduced_ensemble"]
    populations, rotation = torch.linalg.eigh((ensemble+ensemble.T)/2)
    selected = populations > 1e-12
    states, weights = basis @ rotation[:, selected], populations[selected]
    singlet = SingletProjector(data["bits"], len(data["h"]), 3)
    physical = PhysicalAction(data)
    universal = PhysicalAction({**data, "h": data["kinetic"], "nuclear": data["nuclear"].new_zeros(())})
    actual, projected, spin, singlet_weight = data["h"].new_zeros(4)
    for index, vector in enumerate(states.unbind(1)):
        probability = weights[index]
        actual += probability*(vector @ physical(vector))
        purified = singlet(vector)
        projected += probability*(purified @ universal(purified))
        spin += probability*(vector @ singlet.spin_squared(vector))
        singlet_weight += probability*purified.square().sum()
    reported = saved["report"]["physical_total_energy_hartree"]
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "device": device, "id": name,
              "reported_rejected_total_hartree": reported, "unprojected_physical_total_hartree": float(actual),
              "energy_evaluation_error_hartree": float(actual-reported),
              "ensemble_spin_squared": float(spin), "ensemble_singlet_weight": float(singlet_weight),
              "basis_orthogonality_error": float((basis.T @ basis-torch.eye(basis.shape[1], device=device)).norm()),
              "projection_removed_universal_energy_hartree": float(projected),
              "new_fit_or_reference_generated": False,
              "conclusion": "Use the full physical Hamiltonian for energy and explicitly purify every trial-space extension; reject spin leakage immediately"}
    Path("research/auxiliary_spin_failure_gpu_20261117.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
