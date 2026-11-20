"""Frozen-response-space XC derivative checks on the accepted forward points."""
from .runtime import configure, torch_setup
ROOT = configure(gpu_only_compatible=True)

from datetime import datetime, timezone
import json
from pathlib import Path
import torch
from .auxiliary_density_model import AuxiliaryDensityModel


@torch.no_grad()
def main():
    device = torch_setup()
    protocol = json.loads(Path("research/auxiliary_forward_protocol_20261117.json").read_text())
    output = Path("research/auxiliary_derivative_gpu_20261117.json")
    if output.exists(): raise RuntimeError("These prespecified derivative checks already have a report")
    folder = ROOT / "data/charge_spin_accuracy"
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "device": device, "results": [], "failures": [],
              "scope": "One new auxiliary-coordinate direction per geometry, fixed compiled physical response space; not a nuclear derivative or full-space path-accuracy test"}
    for name in protocol["cases"]:
        try:
            data = torch.load(folder / f"{name}.pt", map_location=device, weights_only=True)
            saved = torch.load(ROOT / "data/auxiliary_stationary" / f"{name}.pt", map_location=device, weights_only=True)
            if not saved["report"]["prediction_valid"]: raise RuntimeError("Forward point did not pass its physical gate")
            frame = torch.load(folder / f"{name}.spatial_density.pt", map_location=device, weights_only=True)["frame"]
            number = data["case"]["electrons"]
            occupied = data["orbitals"][:, :number//2]
            seed_density = 2*occupied @ occupied.T
            universal = {"kinetic": data["kinetic"], "eri": data["eri"], "operators": frame["operators"], "bits": data["bits"], "electrons": number}
            model = AuxiliaryDensityModel(universal, seed_density, protocol, checkpoint=saved)
            auxiliary = saved["auxiliary"]
            central = model.saved_point(auxiliary)
            central_kinetic = model.kinetic_search(auxiliary, saved["kinetic_density"])
            central_xc = model.xc(auxiliary, central, central_kinetic)
            generator = torch.Generator(device=device).manual_seed(protocol["derivative_direction_seed"])
            direction = torch.randn(auxiliary.shape, device=device, generator=generator); direction /= direction.norm()
            step = protocol["derivative_step"]
            points = []
            for sign in (-1, 1):
                changed = auxiliary+sign*step*direction
                interacting = model.evaluate(changed, maximum_updates=0)
                kinetic = model.kinetic_search(changed, central_kinetic["state"])
                xc = model.xc(changed, interacting, kinetic)
                points.append({"xc_energy": xc["energy"], "interacting_converged": interacting["converged"],
                               "kinetic_converged": kinetic["converged"], "interacting_gap": interacting["gap"],
                               "kinetic_gap": kinetic["gap"], "trace_error": interacting["trace_error"],
                               "minimum_ensemble_occupation": interacting["minimum_occupation"],
                               "maximum_kinetic_occupation": kinetic["maximum_occupation"]})
            finite_difference = (points[1]["xc_energy"]-points[0]["xc_energy"])/(2*step)
            analytic = central_xc["gradient"] @ direction
            error = float((finite_difference-analytic).abs())
            valid = (error < protocol["derivative_tolerance_hartree"] and central_kinetic["converged"] and
                     all(point["interacting_converged"] and point["kinetic_converged"] for point in points))
            row = {"id": name, "passed": valid, "scalar_directional_derivative": float(finite_difference),
                   "potential_directional_derivative": float(analytic), "absolute_difference_hartree": error,
                   "fixed_response_dimension": saved["basis"].shape[1], "maximum_allowed_difference_hartree": protocol["derivative_tolerance_hartree"],
                   "points": [{key: float(value) if isinstance(value, torch.Tensor) else value for key, value in point.items()} for point in points],
                   "response_space_refitted": False, "molecular_potential_passed_to_universal_evaluation": False,
                   "new_reference_generated": False}
            report["results"].append(row); print(json.dumps(row), flush=True)
            del model, data, saved, universal
            torch.cuda.empty_cache()
        except Exception as exc:
            row = {"id": name, "error": f"{type(exc).__name__}: {exc}"}
            report["failures"].append(row); print(json.dumps(row), flush=True)
        output.write_text(json.dumps(report, indent=2)+"\n")
    report["passed"] = not report["failures"] and len(report["results"]) == len(protocol["cases"]) and all(row["passed"] for row in report["results"])
    report["peak_cuda_bytes"] = torch.cuda.max_memory_allocated()
    output.write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__": main()
