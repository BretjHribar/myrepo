"""Matched larger-basis acceleration of the same coherent state family."""
from .runtime import configure, torch_setup, require_speed_authorization
ROOT = configure(gpu_only_compatible=True)

from datetime import datetime, timezone
import json
from pathlib import Path
import time
import torch
from .gpu_coherent_orbitals import CoherentOrbitals
from .gpu_four_particle import FourParticleOperator

KCAL = 627.509474


def main():
    require_speed_authorization()
    device = torch_setup()
    protocol = json.loads(Path("research/fast_coherent_protocol_20261116.json").read_text())
    original = json.loads(Path("research/coherent_response_larger_gpu_20261116.json").read_text())
    previous = {row["id"]: row for row in original["results"]}
    base_protocol = original["protocol"]
    folder = ROOT / "data/four_particle_ccpvdz"
    output = Path("research/fast_coherent_larger_gpu_20261116.json")
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "device": device,
              "protocol": protocol, "results": [], "failures": []}
    for name in protocol["cases"]:
        started = time.perf_counter()
        try:
            data = torch.load(folder / f"{name}.pt", map_location=device, weights_only=True)
            checkpoint = torch.load(folder / f"{name}.coherent.pt", map_location=device, weights_only=True)
            seed = base_protocol["response_seed"]
            model = CoherentOrbitals(data["h"], data["eri"], 2, proposed_modes=seed["proposed_modes"],
                                     displacement=seed["displacement"], representation="thouless")
            initial = model.angles.detach().clone()
            operator = FourParticleOperator(data["h"], data["eri"], 4, data["nuclear"])
            with torch.no_grad():
                model.angles.copy_(model.coordinates(checkpoint["orbitals"][1:]))
                same_energy, coefficient, orbitals, _ = model()
                vector = model.four_electron_vector(orbitals, coefficient)
                saved = model.four_electron_vector(checkpoint["orbitals"], checkpoint["coefficients"])
                span_overlap = float((vector @ saved).abs().square())
                same_difference = abs(float(same_energy+data["nuclear"])-checkpoint["report"]["total_energy_hartree"])
                if same_difference > 1e-8 or abs(span_overlap-1) > 1e-8:
                    raise RuntimeError(f"Fast coordinates changed the physical span: energy={same_difference}, overlap={span_overlap}")
                model.angles.copy_(initial)
            config = base_protocol["orbital_optimization"]
            optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
            schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config["steps"], eta_min=config["final_learning_rate"])
            best, best_angles = None, initial.clone()
            history = []
            torch.cuda.synchronize()
            training_started = time.perf_counter()
            for step in range(config["steps"]):
                optimizer.zero_grad(set_to_none=True)
                energy, coefficient, orbitals, chosen = model()
                energy.backward()
                if not bool(torch.isfinite(energy)) or not bool(torch.isfinite(model.angles.grad).all()):
                    raise RuntimeError("Nonfinite fast orbital gradient")
                value = float(energy.detach())
                if best is None or value < best:
                    best, best_angles = value, model.angles.detach().clone()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
                optimizer.step(); schedule.step()
                if step == 0 or (step+1) % 100 == 0:
                    row = {"id": name, "step": step+1, "energy_hartree": float(energy.detach()+data["nuclear"]),
                           "seconds": time.perf_counter()-training_started}
                    history.append(row); print(json.dumps(row), flush=True)
            torch.cuda.synchronize()
            training_seconds = time.perf_counter()-training_started
            with torch.no_grad():
                model.angles.copy_(best_angles)
                energy, coefficient, orbitals, chosen = model()
                vector = model.four_electron_vector(orbitals, coefficient)
                image = operator(vector)
                independent = vector @ image
                reference = torch.load(folder / f"{name}.reference.pt", map_location=device, weights_only=True)
                error = float((independent-reference["report"]["total_energy_hartree"])*KCAL)
                discrepancy = float((independent-energy-data["nuclear"]).abs())
                input_energy = operator.seed() @ operator(operator.seed())
            row = {"id": name, "spatial_orbitals": len(data["h"]), "basis": data["basis"],
                   "energy_error_kcal": error, "previous_error_kcal": previous[name]["energy_error_kcal"],
                   "accuracy_change_kcal": error-previous[name]["energy_error_kcal"],
                   "below_one_kcal": abs(error) < 1, "reference_not_degraded": float(independent) <= float(input_energy)+1e-8,
                   "same_saved_span_energy_error_hartree": same_difference, "same_saved_span_overlap_squared": span_overlap,
                   "independent_final_kernel_error_hartree": discrepancy, "state_residual": float((image-independent*vector).norm()),
                   "training_seconds": training_seconds, "total_seconds": time.perf_counter()-started,
                   "previous_total_seconds": previous[name]["seconds"],
                   "conservative_total_speedup": previous[name]["seconds"]/(time.perf_counter()-started),
                   "retained_determinants": len(chosen), "proposed_determinants": len(orbitals),
                   "reference_energy_fitted": False, "reference_recomputed": False, "history": history}
            report["results"].append(row)
            torch.save({"coordinates": best_angles.cpu(), "orbitals": orbitals.cpu(), "coefficients": coefficient.cpu(),
                        "representation": "thouless", "report": row}, folder / f"{name}.fast_coherent.pt")
            print(json.dumps({key: value for key, value in row.items() if key != "history"}), flush=True)
        except Exception as exc:
            row = {"id": name, "error": f"{type(exc).__name__}: {exc}"}
            report["failures"].append(row); print(json.dumps(row), flush=True)
        output.write_text(json.dumps(report, indent=2)+"\n")
    report["peak_cuda_bytes"] = torch.cuda.max_memory_allocated()
    output.write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    main()
