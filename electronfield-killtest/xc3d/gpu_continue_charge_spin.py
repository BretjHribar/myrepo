"""Continue the largest physical residual, not a reference-error tuning loop."""
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
    started = time.perf_counter()
    original = json.loads(Path("research/charge_spin_accuracy_gpu_20261117.json").read_text())
    specification = json.loads(Path("research/charge_spin_continuation_protocol_20261117.json").read_text())
    residuals = torch.tensor([row["full_residual"] for row in original["results"]], device=device)
    selected = int(residuals.argmax())
    previous = original["results"][selected]
    case = next(value for value in original["protocol"]["cases"] if value["id"] == previous["id"])
    protocol = {**original["protocol"], "maximum_adaptive_updates": specification["maximum_adaptive_updates"]}
    data = prepare(case, device)
    saved = torch.load(ROOT / "data/charge_spin_accuracy" / f"{case['id']}.model.pt", map_location=device, weights_only=True)
    prediction = model_prediction(data, protocol, saved["state"])
    # Only now assess against the existing label.
    error = float((prediction["energy"]-previous["reference_energy_hartree"])*KCAL)
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "device": device, "protocol": specification,
              "id": case["id"], "previous_full_residual": previous["full_residual"],
              "previous_error_kcal": previous["energy_error_kcal"], "energy_error_kcal": error,
              "below_one_kcal": abs(error) < 1, "total_energy_hartree": float(prediction["energy"]),
              "reference_energy_hartree": previous["reference_energy_hartree"], "reference_recomputed": False,
              "reference_energy_fitted": False, "seconds": time.perf_counter()-started}
    report.update({key: value for key, value in prediction.items() if key not in ("state", "energy", "block_checkpoints", "unprojected_reference_seed")})
    output = Path("research/charge_spin_continuation_gpu_20261117.json")
    output.write_text(json.dumps(report, indent=2)+"\n")
    torch.save({"state": prediction["state"].cpu(), "block_checkpoints": prediction["block_checkpoints"], "report": report},
               ROOT / "data/charge_spin_accuracy" / f"{case['id']}.continued.pt")
    print(json.dumps({key: value for key, value in report.items() if key not in ("history", "local_multiplets", "protocol")}), flush=True)


if __name__ == "__main__":
    main()
