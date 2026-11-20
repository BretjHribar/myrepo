"""Closed-form positive-ensemble checks, including a rank-deficient face."""
from .runtime import configure, torch_setup
ROOT = configure(gpu_only_compatible=True)

from datetime import datetime, timezone
import json
from pathlib import Path
import torch
from .gpu_psd_ensemble import optimize_ensemble


@torch.no_grad()
def main():
    device = torch_setup()
    matrix = torch.tensor([[0., -1., 0.], [-1., 0., 0.], [0., 0., .5]], device=device)
    identity = torch.eye(3, device=device)
    first = identity.new_zeros(3, 3); first[0, 0] = 1
    last = identity.new_zeros(3, 3); last[2, 2] = 1
    feasible = torch.diag(torch.tensor([.4, .6, 0.], device=device))
    exact = -2*(matrix.new_tensor(.4)*.6).sqrt()
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "device": device, "results": []}
    for face in (False, True):
        operators = torch.stack((identity, first, last) if face else (identity, first))
        target = torch.tensor([1., .4, 0.] if face else [1., .4], device=device)
        result = optimize_ensemble(matrix, operators, target, feasible, maximum_iterations=12000)
        energy_error = float((result["energy"]-exact).abs())
        valid = result["converged"] and energy_error < 1e-7
        report["results"].append({"rank_deficient_face": face, "passed": valid, "energy_error_hartree": energy_error,
                                  "constraint_residual": result["density_residual"], "primal_dual_gap_hartree": result["gap"],
                                  "iterations": result["history"][-1]["iteration"]})
    report["passed"] = all(row["passed"] for row in report["results"])
    Path("research/positive_ensemble_checks_gpu_20261117.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report), flush=True)
    if not report["passed"]:
        raise RuntimeError("Positive-ensemble optimizer failed independent analytic checks")


if __name__ == "__main__":
    main()
