"""Directional XC derivative along physical density paths in a fixed frame."""
from .runtime import configure, torch_setup
ROOT = configure(gpu_only_compatible=True)

from datetime import datetime, timezone
import json
import math
from pathlib import Path
import torch
from .intrinsic_density_frame import one_body_density, project_operator
from .gpu_charge_spin_accuracy import PhysicalAction
from .gpu_four_particle import cached_ritz
from .gpu_ensemble_dual import ensemble_dual


@torch.no_grad()
def main():
    device = torch_setup()
    specification = json.loads(Path("research/spatial_derivative_protocol_20261117.json").read_text())
    output = Path("research/spatial_derivative_gpu_20261117.json")
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "device": device,
              "protocol": specification, "results": [], "failures": []}
    folder = ROOT / "data/charge_spin_accuracy"
    for name in specification["cases"]:
        try:
            data = torch.load(folder / f"{name}.pt", map_location=device, weights_only=True)
            saved = torch.load(folder / f"{name}.larger.pt", map_location=device, weights_only=True)
            certificate = torch.load(folder / f"{name}.spatial_density.pt", map_location=device, weights_only=True)
            if not certificate["report"]["certificate_passed"]:
                raise RuntimeError("The required independent density-frame certificate did not pass")
            frame, operators = certificate["frame"], certificate["frame"]["operators"]
            generator = torch.Generator(device=device).manual_seed(specification["direction_seed"])
            direction = torch.randn(data["h"].shape, device=device, generator=generator)
            direction = (direction+direction.T)/2
            direction -= torch.eye(len(direction), device=device)*direction.trace()/len(direction)
            direction = project_operator(direction, frame)[1]
            direction /= direction.norm()
            points = []
            for sign in (-1, 1):
                changed = {**data, "h": data["h"]+sign*specification["potential_step_hartree"]*direction}
                action = PhysicalAction(changed)
                solution = cached_ritz(action, saved["state"], maximum=specification["maximum_ritz_dimension"],
                                       tolerance=specification["state_residual_tolerance_hartree"])
                if not solution["converged"]:
                    raise RuntimeError(f"Potential-path state failed its physical residual: {solution['residual']}")
                density = one_body_density(solution["state"], data["bits"], len(data["h"]))
                target = torch.einsum("aij,ij->a", operators, density)
                allowance = certificate["report"]["kinetic_entropy_allowance_hartree"]
                beta = 2*len(density)*math.log(2)/allowance
                occupations, natural = torch.linalg.eigh(density)
                effective = (natural*((2/occupations.clamp(1e-14, 2-1e-14)-1).log()/beta)) @ natural.T
                initial = project_operator(effective-data["kinetic"], frame)[0]
                kinetic = ensemble_dual(data["kinetic"], operators, target, beta,
                                        statistics="onebody", initial=initial)
                ks_density = (kinetic["vectors"]*kinetic["population"]) @ kinetic["vectors"].T
                external = changed["h"]-data["kinetic"]
                universal = solution["energy"]-data["nuclear"]-(external*density).sum()
                hartree = .5*torch.einsum("ij,ijkl,kl->", ks_density, data["eri"], ks_density)
                xc = universal-kinetic["dual"]-hartree
                points.append({"xc": xc, "density": ks_density, "state": solution["state"],
                               "state_residual": solution["residual"], "kinetic_residual": kinetic["residual"],
                               "sign": sign, "universal": universal, "kinetic": kinetic["dual"], "hartree": hartree})
                print(json.dumps({"id": name, "sign": sign, "state_residual": solution["residual"],
                                  "kinetic_residual": kinetic["residual"]}), flush=True)
                del action, solution
            step = specification["potential_step_hartree"]
            scalar_change = (points[1]["xc"]-points[0]["xc"])/(2*step)
            density_change = (points[1]["density"]-points[0]["density"])/(2*step)
            potential_change = (certificate["xc_potential"]*density_change).sum()
            error = float((scalar_change-potential_change).abs())
            row = {"id": name, "directional_derivative_passed":
                   error < specification["directional_derivative_tolerance_hartree"] and
                   all(point["kinetic_residual"] < specification["kinetic_density_residual_tolerance"] for point in points),
                   "scalar_directional_derivative": float(scalar_change),
                   "potential_directional_derivative": float(potential_change), "absolute_difference_hartree": error,
                   "density_path_derivative_norm": float(density_change.norm()),
                   "state_residuals": [point["state_residual"] for point in points],
                   "kinetic_density_residuals": [point["kinetic_residual"] for point in points],
                   "central_optimization_repeated": False, "new_reference_labels_generated": False}
            report["results"].append(row)
            torch.save({"direction": direction.cpu(), "points": [
                {key: value.cpu() if isinstance(value, torch.Tensor) else value for key, value in point.items()}
                for point in points], "report": row}, folder / f"{name}.spatial_derivative.pt")
            print(json.dumps(row), flush=True)
        except Exception as exc:
            row = {"id": name, "error": f"{type(exc).__name__}: {exc}"}
            report["failures"].append(row); print(json.dumps(row), flush=True)
        output.write_text(json.dumps(report, indent=2)+"\n")
    report["peak_cuda_bytes"] = torch.cuda.max_memory_allocated()
    output.write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    main()
