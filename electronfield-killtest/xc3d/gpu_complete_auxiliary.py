"""Recover and continue saved auxiliary-density states after verified repairs."""
from .runtime import configure, torch_setup
ROOT = configure(gpu_only_compatible=True)

from datetime import datetime, timezone
import argparse
import gc
import json
from pathlib import Path
import torch
from .auxiliary_density_model import AuxiliaryDensityModel, recover_physical_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default="research/auxiliary_completion_protocol_20261117.json")
    parser.add_argument("--output", default="research/auxiliary_completion_gpu_20261117.json")
    args = parser.parse_args()
    device = torch_setup()
    specification = json.loads(Path(args.protocol).read_text())
    if specification.get("status") == "closed_invalid_spin":
        raise RuntimeError("This numerical version is closed after a verified spin/energy mismatch; use the strict physical-state protocol")
    protocol = json.loads(Path(specification["original_protocol"]).read_text())
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "device": device, "protocol": specification,
              "status": "running", "original_recovered": [], "results": [], "failures": []}
    output = Path(args.output)
    if output.exists(): raise RuntimeError("This completion already has a report; resume its checkpoint instead of restarting")
    output.write_text(json.dumps(report, indent=2)+"\n")
    folder = ROOT / "data/charge_spin_accuracy"
    destination = ROOT / specification.get("checkpoint_folder", "data/auxiliary_completed")
    destination.mkdir(parents=True, exist_ok=True)
    for name in specification["cases"]:
        history = []
        try:
            data = torch.load(folder / f"{name}.pt", map_location=device, weights_only=True)
            frame = torch.load(folder / f"{name}.spatial_density.pt", map_location=device, weights_only=True)["frame"]
            source_folder = ROOT / specification.get("initial_checkpoint_folder", "data/auxiliary_forward")
            saved = torch.load(source_folder / f"{name}{specification.get('initial_checkpoint_suffix', '.progress.pt')}", map_location=device, weights_only=True)
            saved_history = saved.get("history", saved.get("report", {}).get("history", []))
            if specification.get("finish_unconverged_only") and saved_history[-1]["outside_stationarity_residual"] <= protocol["outside_stationarity_tolerance_hartree"]:
                raise RuntimeError("Source physical residual is already complete; do not repeat it")
            number, moments = data["case"]["electrons"], frame["operators"][1:]
            occupied = data["orbitals"][:, :number//2]
            seed_density = 2*occupied @ occupied.T
            universal = {"kinetic": data["kinetic"], "eri": data["eri"], "operators": frame["operators"],
                         "bits": data["bits"], "electrons": number}
            external = data["h"]-data["kinetic"]
            constant, potential = external.trace()/len(external), torch.einsum("aij,ij->a", moments, external)
            if specification.get("strict_spin_recovery"):
                checkpoint, restart = recover_physical_checkpoint(universal, seed_density, saved, protocol)
                model = AuxiliaryDensityModel(universal, seed_density, protocol, checkpoint=checkpoint)
            else:
                model = AuxiliaryDensityModel(universal, seed_density, protocol, checkpoint=saved)
            auxiliary = saved["auxiliary"]
            original = model.saved_point(auxiliary)
            old_kinetic = model.kinetic_search(auxiliary, seed_density)
            old_energy = original["universal_energy"]+potential @ original["moments"]+constant*number+data["nuclear"]
            recovered = {"id": name, "saved_update": saved_history[-1]["update"],
                         "physical_total_energy_hartree": float(old_energy),
                         "potential_self_consistency_residual": float((original["gradient"]+potential).norm()),
                         "outside_stationarity_residual": original["outside_stationarity_residual"],
                         "reduced_gap_hartree": original["gap"], "kinetic_converged": old_kinetic["converged"],
                         "kinetic_gap_hartree": old_kinetic["gap"], "old_optimization_repeated": False}
            if not specification.get("strict_spin_recovery"):
                checkpoint, restart = model.compact_checkpoint()
            report["original_recovered"].append(recovered)
            print(json.dumps({"id": name, "phase": "checked_restart", **restart}), flush=True)
            del model, saved, original, old_kinetic
            gc.collect(); torch.cuda.empty_cache()
            model = AuxiliaryDensityModel(universal, seed_density, protocol, checkpoint=checkpoint)
            epsilon = protocol["epsilon_inverse_hartree"]
            previous_energy = float(old_energy)
            for update in range(specification["maximum_additional_updates"]+1):
                model.refresh()
                total = model.matrix+torch.einsum("a,aij->ij", potential, model.reduced)
                _, vectors = torch.linalg.eigh((total+total.T)/2)
                model.ensemble = vectors[:, :1] @ vectors[:, :1].T
                auxiliary = torch.einsum("aij,ij->a", model.reduced, model.ensemble)-epsilon*potential
                def progress(row):
                    print(json.dumps({"id": name, "completion_update": update, **row}), flush=True)
                interacting = model.evaluate(auxiliary, callback=progress)
                physical_energy = interacting["universal_energy"]+potential @ interacting["moments"]+constant*number+data["nuclear"]
                reconstructed = interacting["functional"]+potential @ auxiliary+epsilon*potential.square().sum()/2+constant*number+data["nuclear"]
                mismatch = interacting["moments"]-auxiliary-epsilon*potential
                identity_error = float((reconstructed-physical_energy-mismatch.square().sum()/(2*epsilon)).abs())
                if identity_error > 1e-9 or float(physical_energy) > previous_energy+1e-7:
                    raise RuntimeError("Completed state failed exact energy recovery or variational monotonicity")
                self_consistency = float((interacting["gradient"]+potential).norm())
                row = {"update": update, "physical_total_energy_hartree": float(physical_energy),
                       "potential_self_consistency_residual": self_consistency,
                       "outside_stationarity_residual": interacting["outside_stationarity_residual"],
                       "reduced_gap_hartree": interacting["gap"], "basis_dimension": interacting["basis_dimension"],
                       "physical_energy_identity_error_hartree": identity_error}
                history.append(row); print(json.dumps({"id": name, **row}), flush=True)
                report["current_case"], report["current_history"] = name, history
                output.write_text(json.dumps(report, indent=2)+"\n")
                temporary = destination / f"{name}.progress.tmp"
                torch.save({**model.checkpoint(), "auxiliary": auxiliary, "history": history}, temporary)
                temporary.replace(destination / f"{name}.progress.pt")
                gc.collect(); torch.cuda.empty_cache()
                self_consistent = (interacting["converged"] and self_consistency < protocol["self_consistency_potential_tolerance"]
                                   and interacting["outside_stationarity_residual"] < protocol["outside_stationarity_tolerance_hartree"])
                capacity_reached = any(item.get("response_capacity_reached", False) for item in interacting["response_history"])
                if self_consistent or update == specification["maximum_additional_updates"] or not interacting["converged"] or capacity_reached:
                    break
                previous_energy = float(physical_energy)
            certification = model.certify(interacting) if self_consistent else None
            kinetic = model.kinetic_search(auxiliary, seed_density)
            xc = model.xc(auxiliary, interacting, kinetic)
            ks_matrix = data["kinetic"]+torch.einsum("a,aij->ij", potential+xc["auxiliary_hartree_gradient"]+xc["gradient"], moments)
            ks_error = float((ks_matrix @ kinetic["state"]-kinetic["state"] @ ks_matrix).norm())
            density = interacting["physical_density"]
            occupations = torch.linalg.eigvalsh((density+density.T)/2)
            number_error = float((density.trace()-number).abs())
            valid = (self_consistent and certification is not None and certification["passed"] and kinetic["converged"]
                     and ks_error < protocol["ks_commutator_tolerance"] and number_error < 1e-8
                     and float(occupations.min()) > -1e-9 and float(occupations.max()) < 2+1e-9)
            # Energy labels are accessed after the completed prediction/checks.
            reference = torch.load(folder / f"{name}.larger.pt", map_location=device, weights_only=True)["report"]["reference_energy_hartree"]
            result = {"id": name, "prediction_valid": valid, "self_consistent": self_consistent,
                      "physical_total_energy_hartree": float(physical_energy), "energy_error_kcal": float((physical_energy-reference)*627.509474),
                      "reference_energy_hartree": reference, "number_error": number_error,
                      "minimum_physical_occupation": float(occupations.min()), "maximum_physical_occupation": float(occupations.max()),
                      "kinetic_converged": kinetic["converged"], "kinetic_gap_hartree": kinetic["gap"], "ks_commutator_norm": ks_error,
                      "global_check": certification, "restart_checks": restart, "history": history,
                      "xc_potential_accepted": valid, "new_reference_generated": False,
                      "oracle_density_used_for_prediction": False, "post_interruption_development_completion": True}
            recovered["energy_error_kcal"] = float((old_energy-reference)*627.509474)
            report["results"].append(result)
            torch.save({**model.checkpoint(), "auxiliary": auxiliary, "physical_density": density,
                        "kinetic_density": kinetic["state"], "xc_gradient": xc["gradient"], "report": result}, destination / f"{name}.pt")
            print(json.dumps({key: value for key, value in result.items() if key != "history"}), flush=True)
            del model, data, universal, interacting, kinetic, checkpoint
            gc.collect(); torch.cuda.empty_cache()
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
