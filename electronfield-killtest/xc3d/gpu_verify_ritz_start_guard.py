"""Independent spectral counterexamples for the new physical start policy."""
from .runtime import configure, torch_setup
ROOT = configure(gpu_only_compatible=True)

from datetime import datetime, timezone
import json
from pathlib import Path
import torch
from .gpu_four_particle import incremental_ritz
from .physical_ritz_seed import mixed_physical_start


@torch.no_grad()
def main():
    output = Path("research/ritz_start_guard_checks_gpu_20261117.json")
    if output.exists(): raise RuntimeError("Do not repeat this new Krylov-start check")
    protocol = json.loads(Path("research/ritz_start_guard_protocol_20261117.json").read_text())
    device = torch_setup(protocol["seed"])
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "device": device,
              "protocol": protocol, "status": "running", "results": [], "failures": []}
    try:
        matrices = [torch.diag(torch.tensor([1., 0.], device=device)),
                    torch.tensor([[1., .2, 0., 0.], [.2, 2., 0., 0.], [0., 0., -1., .15], [0., 0., .15, .5]], device=device)]
        for index, (name, matrix) in enumerate(zip(protocol["cases"], matrices)):
            source = matrix.new_zeros(len(matrix)); source[0] = 1.
            retained_only = incremental_ritz(lambda value: matrix @ value, source, maximum=len(matrix),
                tolerance=protocol["ritz_residual_tolerance_hartree"])
            start, details = mixed_physical_start(source, protocol["seed"]+index)
            corrected = incremental_ritz(lambda value: matrix @ value, start, maximum=len(matrix),
                tolerance=protocol["ritz_residual_tolerance_hartree"])
            exact = torch.linalg.eigvalsh(matrix)[0]
            old_error = float((retained_only["energy"]-exact).abs())
            new_error = float((corrected["energy"]-exact).abs())
            exposed = retained_only["converged"] and old_error > .1
            valid = exposed and corrected["converged"] and new_error < protocol["energy_tolerance_hartree"]
            row = {"id": name, "passed": valid, "old_residual_passed_while_missing_lower_sector": exposed,
                   "old_residual_hartree": retained_only["residual"], "old_ground_energy_error_hartree": old_error,
                   "new_residual_hartree": corrected["residual"], "new_ground_energy_error_hartree": new_error,
                   "start": details, "independent_eigenvalue_checked_after_prediction": True}
            report["results"].append(row); print(json.dumps(row), flush=True)
    except Exception as exc:
        report["failures"].append({"error": f"{type(exc).__name__}: {exc}"}); print(json.dumps(report["failures"][-1]), flush=True)
    report["status"] = "completed"
    report["passed"] = (not report["failures"] and len(report["results"]) == len(protocol["cases"])
                        and all(row["passed"] for row in report["results"]))
    report["peak_cuda_bytes"] = torch.cuda.max_memory_allocated()
    output.write_text(json.dumps(report, indent=2)+"\n")
    if not report["passed"]: raise RuntimeError("The fixed physical-start safeguard failed its independent counterexamples")


if __name__ == "__main__": main()
