"""Test the universal interface once; assess saved oracle values afterward."""
from .runtime import configure, torch_setup
ROOT = configure(gpu_only_compatible=True)

from datetime import datetime, timezone
import json
from pathlib import Path
import torch
from .spatial_xc_model import evaluate_xc


def main():
    device = torch_setup()
    specification = json.loads(Path("research/universal_search_protocol_20261117.json").read_text())
    if specification.get("status") == "closed_failed":
        raise RuntimeError("This bounded inverse-search protocol is closed after failure. A changed physical hypothesis needs a new protocol; do not rerun this one.")
    name = specification["id"]
    folder = ROOT / "data/charge_spin_accuracy"
    data = torch.load(folder / f"{name}.pt", map_location=device, weights_only=True)
    frame = torch.load(folder / f"{name}.spatial_density.pt", map_location=device, weights_only=True)["frame"]
    path = torch.load(folder / f"{name}.spatial_derivative.pt", map_location=device, weights_only=True)
    density = path["points"][0]["density"]
    universal = {"kinetic": data["kinetic"], "eri": data["eri"], "bits": data["bits"], "operators": frame["operators"], "electrons": 6}
    output = Path("research/universal_search_gpu_20261117.json")
    history = []
    def progress(row):
        history.append(row)
        output.write_text(json.dumps({"status": "running", "protocol": specification, "history": history}, indent=2)+"\n")
        print(json.dumps(row), flush=True)
    try:
        result = evaluate_xc(universal, density, maximum_updates=specification["maximum_dual_updates"],
                              density_tolerance=specification["density_residual_tolerance"],
                              state_tolerance=specification["state_residual_tolerance_hartree"],
                              maximum_ritz_dimension=specification["maximum_ritz_dimension"],
                              entropy_allowance=specification["kinetic_entropy_allowance_hartree"], callback=progress)
        # No external potential or target scalar was supplied to evaluate_xc.
        expected = path["points"][0]["universal"]
        difference = float((result["universal_functional"]-expected)*627.509474)
        report = {"created_utc": datetime.now(timezone.utc).isoformat(), "device": device, "status": "completed",
                  "protocol": specification, "converged": result["converged"],
                  "density_residual": result["density_residual"], "state_residual": result["state_residual"],
                  "noninteracting_residual": result["noninteracting_residual"],
                  "universal_functional_minus_saved_oracle_kcal": difference,
                  "xc_energy_available": result["xc_energy"] is not None,
                  "ground_state_evaluations": result["ground_state_evaluations"], "history": result["history"],
                  "peak_cuda_bytes": torch.cuda.max_memory_allocated(), "new_reference_generated": False,
                  "external_potential_passed_to_model": False}
        torch.save({key: value.cpu() if isinstance(value, torch.Tensor) else value for key, value in result.items()},
                   folder / f"{name}.universal_search.pt")
    except Exception as exc:
        report = {"created_utc": datetime.now(timezone.utc).isoformat(), "device": device, "status": "failed",
                  "protocol": specification, "error": f"{type(exc).__name__}: {exc}", "history": history,
                  "converged": False, "xc_energy_available": False,
                  "external_potential_passed_to_model": False, "new_reference_generated": False}
    output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({key: value for key, value in report.items() if key not in ("history", "protocol")}), flush=True)


if __name__ == "__main__":
    main()
