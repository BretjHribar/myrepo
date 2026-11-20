"""One prespecified feasible-ensemble diagnostic on the larger-basis inputs."""
from .runtime import configure, torch_setup
ROOT = configure(gpu_only_compatible=True)

from datetime import datetime, timezone
import argparse
import json
from pathlib import Path
import torch
from .ensemble_spatial_model import evaluate_xc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume-setup-error", action="store_true")
    parser.add_argument("--protocol", default="research/ensemble_spatial_protocol_20261117.json")
    parser.add_argument("--output", default="research/ensemble_spatial_search_gpu_20261117.json")
    args = parser.parse_args()
    protocol = json.loads(Path(args.protocol).read_text())
    if protocol.get("status", "").startswith("closed"):
        raise RuntimeError("This physical/numerical hypothesis is closed; preserve its failed report instead of rerunning it")
    device = torch_setup()
    output = Path(args.output)
    if output.exists():
        previous = json.loads(output.read_text())
        setup_only = (not previous["results"] and len(previous["failures"]) == len(protocol["cases"])
                      and all(row["error"] == "KeyError: 'bits'" and not row["history"] for row in previous["failures"]))
        if not (args.resume_setup_error and setup_only):
            raise RuntimeError("This experiment already has a report; resume an unfinished checkpoint explicitly, do not rerun it")
        output.with_name("ensemble_spatial_setup_error_gpu_20261117.json").write_text(json.dumps(previous, indent=2)+"\n")
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "device": device, "protocol": protocol,
              "results": [], "failures": [], "status": "running"}
    output.write_text(json.dumps(report, indent=2)+"\n")
    folder = ROOT / "data/charge_spin_accuracy"
    checkpoints = ROOT / protocol.get("checkpoint_folder", "data/ensemble_spatial_search")
    checkpoints.mkdir(exist_ok=True, parents=True)
    for case in protocol["cases"]:
        name = case["id"]
        data = torch.load(folder / f"{name}.pt", map_location=device, weights_only=True)
        saved = torch.load(folder / f"{name}.spatial_derivative.pt", map_location=device, weights_only=True)
        frame = torch.load(folder / f"{name}.spatial_density.pt", map_location=device, weights_only=True)["frame"]
        density = saved["points"][case["path_index"]]["density"]
        universal = {"kinetic": data["kinetic"], "eri": data["eri"], "bits": data["bits"],
                     "operators": frame["operators"], "electrons": data["case"]["electrons"]}
        history = []
        def progress(row):
            history.append(row)
            report["current_case"], report["current_history"] = name, history
            output.write_text(json.dumps(report, indent=2)+"\n")
            print(json.dumps({"id": name, **row}), flush=True)
        try:
            warm = None
            if protocol.get("initial_checkpoint_folder"):
                warm = torch.load(ROOT / protocol["initial_checkpoint_folder"] / f"{name}.pt", map_location=device, weights_only=True)
            def persist(payload):
                temporary = checkpoints / f"{name}.progress.tmp"
                torch.save({key: value.cpu() if isinstance(value, torch.Tensor) else value for key, value in payload.items()}, temporary)
                temporary.replace(checkpoints / f"{name}.progress.pt")
            result = evaluate_xc(universal, density, protocol, callback=progress, checkpoint=warm, checkpoint_callback=persist)
            expected = saved["points"][case["path_index"]]["universal"]
            row = {key: value for key, value in result.items() if not isinstance(value, torch.Tensor)}
            row.update({"id": name, "universal_upper_minus_saved_oracle_kcal": float((result["universal_upper_functional"]-expected)*627.509474),
                        "universal_energy_minus_saved_oracle_kcal": float((result["universal_energy"]-expected)*627.509474),
                        "xc_energy_available": result["xc_energy"] is not None, "xc_potential_available": result["xc_potential"] is not None,
                        "external_potential_passed_to_model": False, "new_reference_generated": False})
            report["results"].append(row)
            torch.save({key: value.cpu() if isinstance(value, torch.Tensor) else value for key, value in result.items()}, checkpoints / f"{name}.pt")
            print(json.dumps({key: value for key, value in row.items() if key not in ("history", "dual_history")}), flush=True)
            del result
        except Exception as exc:
            row = {"id": name, "error": f"{type(exc).__name__}: {exc}", "history": history,
                   "xc_energy_available": False, "xc_potential_available": False}
            report["failures"].append(row); print(json.dumps(row), flush=True)
        output.write_text(json.dumps(report, indent=2)+"\n")
        del data, saved, frame, density, universal
        torch.cuda.empty_cache()
    report["status"] = "completed"
    report.pop("current_history", None); report.pop("current_case", None)
    report["peak_cuda_bytes"] = torch.cuda.max_memory_allocated()
    output.write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    main()
