"""Finish only unfinished physical residuals in the development panel."""
from .runtime import configure, torch_setup
ROOT = configure(gpu_only_compatible=True)

from datetime import datetime, timezone
import json
from pathlib import Path
import time
import torch
from .gpu_charge_spin_accuracy import prepare, model_prediction

KCAL = 627.509474


def main():
    device = torch_setup()
    original = json.loads(Path("research/charge_spin_accuracy_gpu_20261117.json").read_text())
    specification = json.loads(Path("research/stationary_panel_protocol_20261117.json").read_text())
    continued = json.loads(Path("research/charge_spin_continuation_gpu_20261117.json").read_text())
    latest = {row["id"]: row for row in original["results"]}
    latest[continued["id"]] = {**latest[continued["id"]], **continued}
    residuals = torch.tensor([row["full_residual"] for row in latest.values()], device=device)
    names = list(latest)
    ordered = residuals.argsort(descending=True).tolist()
    output = Path("research/stationary_panel_completion_gpu_20261117.json")
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "device": device,
              "protocol": specification, "results": [], "kept_completed": [], "failures": []}
    protocol = {**original["protocol"], "maximum_adaptive_updates": specification["maximum_adaptive_updates"]}
    for index in ordered:
        name, previous = names[index], latest[names[index]]
        if previous["full_residual"] <= protocol["full_state_residual_target_hartree"]:
            report["kept_completed"].append({"id": name, "full_residual": previous["full_residual"],
                                             "energy_error_kcal": previous["energy_error_kcal"]})
            continue
        started = time.perf_counter()
        try:
            case = next(value for value in protocol["cases"] if value["id"] == name)
            data = prepare(case, device)
            saved = torch.load(ROOT / "data/charge_spin_accuracy" / f"{name}.model.pt", map_location=device, weights_only=True)
            prediction = model_prediction(data, protocol, saved["state"])
            error = float((prediction["energy"]-previous["reference_energy_hartree"])*KCAL)
            row = {"id": name, "previous_full_residual": previous["full_residual"],
                   "energy_error_kcal": error, "total_energy_hartree": float(prediction["energy"]),
                   "reference_energy_hartree": previous["reference_energy_hartree"],
                   "reference_recomputed": False, "reference_energy_fitted": False,
                   "seconds": time.perf_counter()-started}
            row.update({key: value for key, value in prediction.items() if key not in ("state", "energy", "block_checkpoints", "unprojected_reference_seed")})
            report["results"].append(row)
            torch.save({"state": prediction["state"].cpu(), "block_checkpoints": prediction["block_checkpoints"], "report": row},
                       ROOT / "data/charge_spin_accuracy" / f"{name}.stationary.pt")
            print(json.dumps({key: value for key, value in row.items() if key not in ("history", "local_multiplets")}), flush=True)
        except Exception as exc:
            row = {"id": name, "error": f"{type(exc).__name__}: {exc}"}
            report["failures"].append(row); print(json.dumps(row), flush=True)
        output.write_text(json.dumps(report, indent=2)+"\n")
    report["peak_cuda_bytes"] = torch.cuda.max_memory_allocated()
    output.write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    main()
