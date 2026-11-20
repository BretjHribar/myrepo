"""Prespecified new larger-basis accuracy cases; references follow prediction."""
from .runtime import configure, torch_setup
ROOT = configure(gpu_only_compatible=True)

from datetime import datetime, timezone
import json
from pathlib import Path
import torch
from .gpu_charge_spin_accuracy import prepare, model_prediction, PhysicalAction
from .gpu_four_particle import cached_ritz

KCAL = 627.509474


def main():
    device = torch_setup()
    specification = json.loads(Path("research/larger_charge_spin_protocol_20261117.json").read_text())
    base = json.loads(Path("research/charge_spin_accuracy_protocol_20261117.json").read_text())
    protocol = {**base, "maximum_adaptive_updates": specification["maximum_adaptive_updates"],
                "full_state_residual_target_hartree": specification["full_state_residual_target_hartree"]}
    output = Path("research/larger_charge_spin_accuracy_gpu_20261117.json")
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "device": device,
              "protocol": specification, "results": [], "failures": []}
    folder = ROOT / "data/charge_spin_accuracy"
    for case in specification["cases"]:
        destination = folder / f"{case['id']}.larger.pt"
        try:
            if destination.exists():
                saved = torch.load(destination, map_location=device, weights_only=True)
                if "report" in saved:
                    report["results"].append(saved["report"])
                    continue
                raise RuntimeError("An unfinished saved prediction exists; resume it explicitly instead of rerunning")
            print(json.dumps({"id": case["id"], "phase": "new_larger_basis_inputs"}), flush=True)
            data = prepare(case, device)
            prediction = model_prediction(data, protocol)
            torch.save({"state": prediction["state"].cpu(), "block_checkpoints": prediction["block_checkpoints"],
                        "energy_hartree": float(prediction["energy"]), "reference_generated": False}, destination)
            print(json.dumps({"id": case["id"], "phase": "prediction_saved_before_reference",
                              "full_residual": prediction["full_residual"]}), flush=True)
            reference = cached_ritz(PhysicalAction(data), prediction["unprojected_reference_seed"],
                                    maximum=specification["maximum_reference_ritz_dimension"],
                                    tolerance=specification["reference_residual_tolerance_hartree"])
            error = float((prediction["energy"]-reference["energy"])*KCAL) if reference["converged"] else None
            row = {"id": case["id"], "electrons": case["electrons"], "basis": case["basis"],
                   "spatial_orbitals": len(data["h"]), "full_fixed_sz_dimension": len(data["bits"]),
                   "energy_error_kcal": error, "total_energy_hartree": float(prediction["energy"]),
                   "reference_energy_hartree": float(reference["energy"]),
                   "reference_converged": reference["converged"], "reference_residual": reference["residual"],
                   "reference_energy_fitted": False}
            row.update({key: value for key, value in prediction.items()
                        if key not in ("state", "energy", "block_checkpoints", "unprojected_reference_seed")})
            report["results"].append(row)
            torch.save({"state": prediction["state"].cpu(), "reference_state": reference["state"].cpu(),
                        "block_checkpoints": prediction["block_checkpoints"], "report": row}, destination)
            print(json.dumps({key: value for key, value in row.items() if key not in ("history", "local_multiplets")}), flush=True)
            del prediction, reference, data
            torch.cuda.empty_cache()
        except Exception as exc:
            row = {"id": case["id"], "error": f"{type(exc).__name__}: {exc}"}
            report["failures"].append(row); print(json.dumps(row), flush=True)
        output.write_text(json.dumps(report, indent=2)+"\n")
    if len(report["results"]) == 2 and all(row["energy_error_kcal"] is not None for row in report["results"]):
        errors = torch.tensor([row["energy_error_kcal"] for row in report["results"]], device=device)
        report["balanced_stretch_contrast_error_kcal"] = float(errors[1]-errors[0])
    report["peak_cuda_bytes"] = torch.cuda.max_memory_allocated()
    output.write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    main()
