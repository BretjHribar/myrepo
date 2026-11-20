"""A CUDA lower-bound certificate for the radial-only two-site He state."""
from .runtime import configure, torch_setup
ROOT = configure(gpu_only_compatible=True)

import json
from pathlib import Path
import torch


def main():
    device = torch_setup()
    case = torch.load(ROOT / "data/gpu_pair_state/He_atom.pt", map_location=device, weights_only=True)
    mass, interaction = case["mass"], case["interaction"]
    p, q = mass[:, 0, 0], mass[:, 1, 0]
    lower = (p+q-1).clamp_min(0)
    upper = torch.minimum(p, q)
    slope = interaction[:, 0, 0]-interaction[:, 0, 1]-interaction[:, 1, 0]+interaction[:, 1, 1]
    x = torch.where(slope >= 0, lower, upper)
    best = torch.stack((torch.stack((x, p-x), -1), torch.stack((q-x, 1-p-q+x), -1)), -2)
    product = mass[:, 0, :, None]*mass[:, 1, None, :]
    most_negative = (interaction*(best-product)).sum((-1, -2))
    required = case["energy"]-case["energy_hf_input"]
    gap = (most_negative-required).clamp_min(0)*627.509474
    report = {"scope": "Analytic 2x2 transport minimum, with kinetic cost removed entirely; a lower bound on this coarse state's possible energy",
              "device": device, "case": "He_atom", "required_correlation_hartree": float(required[0]),
              "most_negative_allowed_correlation_hartree": float(most_negative[0]),
              "unavoidable_energy_error_floor_kcal": float(gap[0]),
              "conclusion": "Radial-only state cannot represent the reference" if float(gap[0]) > 1 else "This bound alone does not reject representability"}
    output = Path("research/joint_transport_radial_capacity_gpu_20261116.json")
    output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
