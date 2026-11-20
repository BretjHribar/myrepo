"""Prespecified forward test of the auxiliary-density physical state."""
from .runtime import configure, torch_setup
ROOT = configure(gpu_only_compatible=True)

from datetime import datetime, timezone
import json
from pathlib import Path
import torch
from .auxiliary_density_model import AuxiliaryDensityModel


def main():
    device = torch_setup()
    protocol = json.loads(Path("research/auxiliary_forward_protocol_20261117.json").read_text())
    output = Path("research/auxiliary_forward_gpu_20261117.json")
    if output.exists(): raise RuntimeError("This bounded experiment already has a report; do not restart a completed solve")
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "device": device, "protocol": protocol,
              "status": "running", "results": [], "failures": []}
    output.write_text(json.dumps(report, indent=2)+"\n")
    folder = ROOT / "data/charge_spin_accuracy"
    destination = ROOT / "data/auxiliary_forward"
    destination.mkdir(parents=True, exist_ok=True)
    for name in protocol["cases"]:
        history = []
        try:
            data = torch.load(folder / f"{name}.pt", map_location=device, weights_only=True)
            frame = torch.load(folder / f"{name}.spatial_density.pt", map_location=device, weights_only=True)["frame"]
            number, moments = data["case"]["electrons"], frame["operators"][1:]
            occupied = data["orbitals"][:, :number//2]
            seed_density = 2*occupied @ occupied.T
            universal = {"kinetic": data["kinetic"], "eri": data["eri"], "operators": frame["operators"],
                         "bits": data["bits"], "electrons": number}
            external = data["h"]-data["kinetic"]
            constant = external.trace()/len(external)
            potential = torch.einsum("aij,ij->a", moments, external)
            span_error = float((external-constant*torch.eye(len(external), device=device)-torch.einsum("a,aij->ij", potential, moments)).norm())
            if span_error > 1e-7: raise RuntimeError("Frozen spatial frame does not span the physical external potential")
            model = AuxiliaryDensityModel(universal, seed_density, protocol)
            epsilon = protocol["epsilon_inverse_hartree"]
            auxiliary = torch.einsum("aij,ij->a", moments, seed_density)-epsilon*potential
            previous_energy = None
            for outer in range(protocol["maximum_self_consistency_updates"]+1):
                def progress(row):
                    print(json.dumps({"id": name, "self_consistency_update": outer, **row}), flush=True)
                interacting = model.evaluate(auxiliary, callback=progress)
                physical_energy = interacting["universal_energy"]+potential @ interacting["moments"]+constant*number+data["nuclear"]
                reconstructed = interacting["functional"]+potential @ auxiliary+epsilon*potential.square().sum()/2+constant*number+data["nuclear"]
                residual = interacting["gradient"]+potential
                mismatch = interacting["moments"]-auxiliary-epsilon*potential
                identity_error = float((reconstructed-physical_energy-mismatch.square().sum()/(2*epsilon)).abs())
                if identity_error > 1e-9: raise RuntimeError("Exact physical-energy recovery identity failed")
                if previous_energy is not None and float(physical_energy) > previous_energy+1e-7:
                    raise RuntimeError("The physical variational energy increased during self-consistency")
                row = {"update": outer, "physical_total_energy_hartree": float(physical_energy),
                       "potential_self_consistency_residual": float(residual.norm()), "outside_stationarity_residual": interacting["outside_stationarity_residual"],
                       "reduced_gap_hartree": interacting["gap"], "basis_dimension": interacting["basis_dimension"],
                       "physical_energy_identity_error_hartree": identity_error}
                history.append(row); print(json.dumps({"id": name, **row}), flush=True)
                report["current_case"], report["current_history"] = name, history
                output.write_text(json.dumps(report, indent=2)+"\n")
                checkpoint = {**model.checkpoint(), "auxiliary": auxiliary, "physical_density": interacting["physical_density"], "history": history}
                temporary = destination / f"{name}.progress.tmp"
                torch.save({key: value.cpu() if isinstance(value, torch.Tensor) else value for key, value in checkpoint.items()}, temporary)
                temporary.replace(destination / f"{name}.progress.pt")
                self_consistent = (interacting["converged"] and float(residual.norm()) < protocol["self_consistency_potential_tolerance"]
                                   and interacting["outside_stationarity_residual"] < protocol["outside_stationarity_tolerance_hartree"])
                if self_consistent or outer == protocol["maximum_self_consistency_updates"] or not interacting["converged"]:
                    break
                previous_energy = float(physical_energy)
                auxiliary = interacting["moments"]-epsilon*potential
            certification = model.certify(interacting) if self_consistent else None
            kinetic = model.kinetic_search(auxiliary, seed_density)
            xc = model.xc(auxiliary, interacting, kinetic)
            ks_matrix = data["kinetic"]+torch.einsum("a,aij->ij", potential+xc["auxiliary_hartree_gradient"]+xc["gradient"], moments)
            ks_error = float((ks_matrix @ kinetic["state"]-kinetic["state"] @ ks_matrix).norm())
            physical = interacting["physical_density"]
            occupations = torch.linalg.eigvalsh((physical+physical.T)/2)
            number_error = float((physical.trace()-number).abs())
            valid = (self_consistent and certification is not None and certification["passed"] and kinetic["converged"]
                     and ks_error < protocol["ks_commutator_tolerance"] and number_error < 1e-8
                     and float(occupations.min()) > -1e-9 and float(occupations.max()) < 2+1e-9)
            # Reference labels are read only after prediction and its checks.
            reference = torch.load(folder / f"{name}.larger.pt", map_location=device, weights_only=True)["report"]["reference_energy_hartree"]
            result = {"id": name, "prediction_valid": valid, "self_consistent": self_consistent,
                      "physical_total_energy_hartree": float(physical_energy), "reference_energy_hartree": reference,
                      "energy_error_kcal": float((physical_energy-reference)*627.509474),
                      "external_span_error_hartree": span_error, "number_error": number_error,
                      "minimum_physical_occupation": float(occupations.min()), "maximum_physical_occupation": float(occupations.max()),
                      "kinetic_converged": kinetic["converged"], "kinetic_gap_hartree": kinetic["gap"], "ks_commutator_norm": ks_error,
                      "global_check": certification, "history": history, "oracle_density_used_for_prediction": False,
                      "new_reference_generated": False, "regularized_xc_energy_hartree": float(xc["energy"]),
                      "xc_potential_accepted": valid}
            report["results"].append(result)
            torch.save({**{key: value.cpu() for key, value in model.checkpoint().items()}, "auxiliary": auxiliary.cpu(),
                        "physical_density": physical.cpu(), "kinetic_density": kinetic["state"].cpu(),
                        "xc_gradient": xc["gradient"].cpu(), "report": result}, destination / f"{name}.pt")
            print(json.dumps({key: value for key, value in result.items() if key != "history"}), flush=True)
            del model, data, universal, interacting, kinetic
            torch.cuda.empty_cache()
        except Exception as exc:
            result = {"id": name, "error": f"{type(exc).__name__}: {exc}", "history": history, "xc_potential_accepted": False}
            report["failures"].append(result); print(json.dumps(result), flush=True)
        output.write_text(json.dumps(report, indent=2)+"\n")
    report["status"] = "completed"
    report.pop("current_case", None); report.pop("current_history", None)
    report["peak_cuda_bytes"] = torch.cuda.max_memory_allocated()
    if len(report["results"]) == 2:
        errors = torch.tensor([row["energy_error_kcal"] for row in report["results"]], device=device)
        report["balanced_stretch_contrast_error_kcal"] = float(errors[1]-errors[0])
    output.write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    main()
