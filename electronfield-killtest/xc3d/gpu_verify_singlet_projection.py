"""Independent CG-spin and operator checks for the new singlet projector."""
from .runtime import configure, torch_setup
ROOT = configure(gpu_only_compatible=True)

from datetime import datetime, timezone
import json
from pathlib import Path
import torch
from .gpu_fock_blocks import SingletProjector, word_connections
from .gpu_manybody_ops import fixed_sector
from .gpu_spin_chain import SpinNumberState, amplitudes
from .gpu_charge_spin_accuracy import PhysicalAction


@torch.no_grad()
def main():
    device = torch_setup()
    data = torch.load(ROOT / "data/charge_spin_accuracy/BeH2_631g_1.65_new.pt", map_location=device, weights_only=True)
    sites, pairs = len(data["h"]), 3
    bits = data["bits"]
    projector = SingletProjector(bits, sites, pairs)
    words = torch.tensor([((i, 1, 1), (i, 0, 0)) for i in range(sites)], device=device)
    coefficients = torch.ones(sites, device=device)
    rows = []
    for spin in range(4):
        state = SpinNumberState(sites, 6, twice_spin=2*spin, channels=1, device=device)
        source_bits, configurations = fixed_sector(sites, pairs+spin, pairs-spin, device)
        vector = amplitudes(state.cores(), configurations)
        vector /= vector.norm()
        for magnetic in range(spin, 0, -1):
            target_bits, _ = fixed_sector(sites, pairs+magnetic-1, pairs-magnetic+1, device)
            row, column, factor = word_connections(words, coefficients, target_bits, source_bits)
            lowered = vector.new_zeros(len(target_bits))
            lowered.scatter_add_(0, row, factor*vector[column])
            vector = lowered/lowered.norm(); source_bits = target_bits
        casimir = float((projector.spin_squared(vector)-spin*(spin+1)*vector).norm())
        projected = projector(vector)
        projection_error = float((projected-vector).norm()) if spin == 0 else float(projected.norm())
        rows.append({"independent_cg_total_spin": spin, "casimir_state_error": casimir,
                     "singlet_retention_or_rejection_error": projection_error})
    first, second = torch.randn(len(bits), device=device), torch.randn(len(bits), device=device)
    first /= first.norm(); second /= second.norm()
    projected = projector(first)
    idempotence = float((projector(projected)-projected).norm())
    hermiticity = float((first @ projector(second)-projected @ second).abs())
    action = PhysicalAction(data)
    commutator = float((projector(action(first))-action(projected)).norm())
    valid = (all(max(row["casimir_state_error"], row["singlet_retention_or_rejection_error"]) < 1e-9 for row in rows)
             and max(idempotence, hermiticity, commutator) < 1e-9)
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "device": device, "passed": valid,
              "independent_spin_tests": rows, "idempotence_error": idempotence,
              "hermiticity_error": hermiticity, "physical_hamiltonian_commutator_norm": commutator,
              "ground_state_or_reference_generated": False, "peak_cuda_bytes": torch.cuda.max_memory_allocated()}
    Path("research/singlet_projection_checks_gpu_20261117.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report), flush=True)
    if not valid:
        raise RuntimeError("The singlet projector failed independent physical checks")


if __name__ == "__main__":
    main()
