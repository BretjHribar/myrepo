"""One frozen test of the new coherent orbital state on larger bases."""
from .runtime import configure, torch_setup
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
    device = torch_setup()
    protocol = json.loads(Path("research/coherent_response_protocol_20261116.json").read_text())
    folder = ROOT / "data/four_particle_ccpvdz"
    output = Path("research/coherent_response_larger_gpu_20261116.json")
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "device": device, "protocol": protocol,
              "results": [], "failures": []}
    for name in protocol["cases"]:
        started = time.perf_counter()
        try:
            data = torch.load(folder / f"{name}.pt", map_location=device, weights_only=True)
            seed = protocol["response_seed"]
            model = CoherentOrbitals(data["h"], data["eri"], 2, proposed_modes=seed["proposed_modes"], displacement=seed["displacement"])
            operator = FourParticleOperator(data["h"], data["eri"], 4, data["nuclear"])
            # Kernel implementation identity on the new state, not a rerun of
            # a published method or the earlier scientific benchmark fits.
            with torch.no_grad():
                energy, coefficient, orbitals, _ = model()
                vector = model.four_electron_vector(orbitals, coefficient)
                initial_error = float((vector @ operator(vector)-energy-data["nuclear"]).abs())
            if initial_error > 1e-8:
                raise RuntimeError(f"Cofactor Hamiltonian disagrees with independent coordinate action: {initial_error}")
            config = protocol["orbital_optimization"]
            optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
            schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config["steps"], eta_min=config["final_learning_rate"])
            best, best_angles = float(energy), model.angles.detach().clone()
            history = []
            for step in range(config["steps"]):
                optimizer.zero_grad(set_to_none=True)
                energy, coefficient, orbitals, chosen = model()
                energy.backward()
                if not bool(torch.isfinite(energy)) or not bool(torch.isfinite(model.angles.grad).all()):
                    raise RuntimeError("Nonfinite coherent-state orbital gradient")
                if float(energy.detach()) < best:
                    best, best_angles = float(energy.detach()), model.angles.detach().clone()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
                optimizer.step(); schedule.step()
                if step == 0 or (step+1) % 100 == 0:
                    row = {"id": name, "step": step+1, "energy_hartree": float(energy.detach()+data["nuclear"]),
                           "retained_determinants": len(chosen), "seconds": time.perf_counter()-started}
                    history.append(row); print(json.dumps(row), flush=True)
            with torch.no_grad():
                model.angles.copy_(best_angles)
                energy, coefficient, orbitals, chosen = model()
                vector = model.four_electron_vector(orbitals, coefficient)
                image = operator(vector)
                independent = vector @ image
                discrepancy = float((independent-energy-data["nuclear"]).abs())
                # No reference energy was an objective, parameter or stopping input.
                reference = torch.load(folder / f"{name}.reference.pt", map_location=device, weights_only=True)
                error = float((independent-reference["report"]["total_energy_hartree"])*KCAL)
                input_energy = operator.seed() @ operator(operator.seed())
            row = {"id": name, "basis": data["basis"], "complete_stated_basis": True,
                   "electrons": 4, "spatial_orbitals": len(data["h"]), "response_modes": model.response_modes,
                   "proposed_determinants": len(orbitals), "retained_determinants": len(chosen),
                   "parameters": model.angles.numel(), "total_energy_hartree": float(independent),
                   "energy_error_kcal": error, "state_residual": float((image-independent*vector).norm()),
                   "reference_not_degraded": float(independent) <= float(input_energy)+1e-8,
                   "initial_kernel_discrepancy_hartree": initial_error, "final_kernel_discrepancy_hartree": discrepancy,
                   "coulomb_factorization_error": model.kernel.factorization_error,
                   "reference_energy_fitted": False, "seconds": time.perf_counter()-started, "history": history}
            report["results"].append(row)
            torch.save({"angles": best_angles.cpu(), "coefficients": coefficient.cpu(), "orbitals": orbitals.cpu(),
                        "report": row}, folder / f"{name}.coherent.pt")
            print(json.dumps({key: value for key, value in row.items() if key != "history"}), flush=True)
        except Exception as exc:
            row = {"id": name, "error": f"{type(exc).__name__}: {exc}"}
            report["failures"].append(row); print(json.dumps(row), flush=True)
        output.write_text(json.dumps(report, indent=2)+"\n")
    report["peak_cuda_bytes"] = torch.cuda.max_memory_allocated()
    output.write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    main()
