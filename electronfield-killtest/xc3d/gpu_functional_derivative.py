"""One necessary scalar/potential derivative check for the new functional.

Uses saved water fixed-point density, CUDA-only constrained solves and no
reference-energy labels. Orbital rotation keeps D in 0<=D<=2 and Tr D=N.
"""
from .runtime import configure, torch_setup
ROOT = configure(gpu_only_compatible=True)

from datetime import datetime, timezone
import json
from pathlib import Path
import time
import torch
from .coulomb_moment_model import evaluate_xc


@torch.no_grad()
def main():
    device = torch_setup()
    started = time.perf_counter()
    name = "H2O_closed_0.957"
    data = torch.load(ROOT / "data/coulomb_moment_functional" / f"{name}.pt", map_location=device, weights_only=True)
    checkpoint = torch.load(Path("research") / f"{name}_dual_fixedpoint.pt", map_location=device, weights_only=True)
    density = checkpoint["noninteracting_density"]
    direction = data["moments"][0] @ density-density @ data["moments"][0]
    direction = direction/direction.norm()
    step = 1e-5
    rotation_plus, rotation_minus = torch.matrix_exp(step*direction), torch.matrix_exp(-step*direction)
    plus_density = rotation_plus @ density @ rotation_plus.T
    minus_density = rotation_minus @ density @ rotation_minus.T
    central = evaluate_xc(data, density)
    plus = evaluate_xc(data, plus_density)
    minus = evaluate_xc(data, minus_density)
    finite = (plus["xc_energy"]-minus["xc_energy"])/(2*step)
    density_tangent = direction @ density-density @ direction
    predicted = (central["xc_potential"]*density_tangent).sum()
    difference = float((finite-predicted).abs())
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "device": device,
              "scope": "Frozen Coulomb frame, one ensemble-valid density rotation; no full-density or force claim",
              "finite_difference_hartree": float(finite), "potential_prediction_hartree": float(predicted),
              "absolute_difference_hartree": difference, "step": step,
              "derivative_check_passed": difference < 1e-4,
              "density_constraints": [item["interacting"]["density_residual"] for item in (central, plus, minus)],
              "state_stationarity_residuals": [item["interacting"]["stationarity_residual"] for item in (central, plus, minus)],
              "kinetic_density_number_residuals": [item["kinetic"]["residual"] for item in (central, plus, minus)],
              "reference_energy_used": False, "seconds": time.perf_counter()-started,
              "peak_cuda_bytes": torch.cuda.max_memory_allocated()}
    Path("research/coulomb_functional_derivative_gpu_20261116.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
