"""Continue an unfinished physical residual from exact saved block subspaces."""
from .runtime import configure, torch_setup
ROOT = configure(gpu_only_compatible=True)

from datetime import datetime, timezone
import json
from pathlib import Path
import torch
from .gpu_charge_spin_accuracy import model_prediction


def main():
    device = torch_setup()
    folder = ROOT / "data/charge_spin_accuracy"
    name = "H2O2_skew_new"
    specification = json.loads(Path("research/peroxide_completion_protocol_20261117.json").read_text())
    protocol = json.loads(Path("research/charge_spin_accuracy_protocol_20261117.json").read_text())
    protocol["maximum_adaptive_updates"] = specification["maximum_additional_updates"]
    data = torch.load(folder / f"{name}.pt", map_location=device, weights_only=True)
    saved = torch.load(folder / f"{name}.model.pt", map_location=device, weights_only=True)
    if saved["report"]["full_residual"] <= specification["full_state_residual_target_hartree"]:
        raise RuntimeError("The saved solve is already complete; do not rerun it")
    prediction = model_prediction(data, protocol, saved["state"], saved["block_checkpoints"])
    exact = saved["report"]["reference_energy_hartree"]
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "device": device, "id": name,
              "protocol": specification, "total_energy_hartree": float(prediction["energy"]),
              "reference_energy_hartree": exact, "energy_error_kcal": float((prediction["energy"]-exact)*627.509474),
              "original_full_residual": saved["report"]["full_residual"], "new_reference_generated": False,
              "local_subspaces_restored_without_eigensolve": True,
              "post_label_development_completion": True, "peak_cuda_bytes": torch.cuda.max_memory_allocated()}
    report.update({key: value for key, value in prediction.items()
                   if key not in ("state", "energy", "block_checkpoints", "unprojected_reference_seed")})
    torch.save({"state": prediction["state"].cpu(), "block_checkpoints": prediction["block_checkpoints"], "report": report},
               folder / f"{name}.stationary.pt")
    Path("research/peroxide_completion_gpu_20261117.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({key: value for key, value in report.items() if key not in ("history", "local_multiplets", "protocol")}), flush=True)


if __name__ == "__main__":
    main()
