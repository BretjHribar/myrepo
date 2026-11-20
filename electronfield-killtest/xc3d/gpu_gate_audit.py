"""Audit whether a numerical small-angle cap is limiting the physical state."""
from .runtime import configure, torch_setup
ROOT = configure(gpu_only_compatible=True)

import json
from pathlib import Path
import torch
from .gpu_angular_flow import AngularFlow
from .gpu_sector_flow import variational_energy


def main():
    device = torch_setup()
    rows = []
    for name in ("H3_neutral_1.00", "H3_neutral_1.80"):
        for rank, prefix, suffix in ((1, "angular_connected_flow_gpu_20261116", "angular"),
                                      (2, "quadrupole_connected_flow_gpu_20261116", "quadrupole")):
            data = torch.load(ROOT / "data/gpu_odd_operators" / f"{name}.{suffix}_seed.pt", map_location=device, weights_only=True)
            checkpoint = torch.load(Path("research") / f"{prefix}_{name}.pt", map_location=device, weights_only=True)
            model = AngularFlow(data)
            model.load_state_dict(checkpoint["model"])
            energy, _ = variational_energy(model(), data["h"], data["eri"])
            gradient = torch.autograd.grad(energy, model.raw)[0]
            tau = .1*torch.tanh(model.raw)
            physical_gradient = gradient/(.1*(1-torch.tanh(model.raw).square())).clamp_min(1e-16)
            rows.append({"id": name, "angular_rank": rank, "maximum_abs_angle": float(tau.abs().max()),
                         "fraction_angles_above_95_percent_cap": float((tau.abs() > .095).double().mean()),
                         "raw_gradient_norm": float(gradient.norm()), "physical_angle_gradient_norm": float(physical_gradient.norm())})
    report = {"scope": "Numerical parameterization audit; no additional fit or benchmark", "device": device, "cases": rows}
    Path("research/angular_gate_capacity_audit_gpu_20261116.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
