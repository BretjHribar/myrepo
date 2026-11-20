"""Accuracy-only tests of the local coherent primitive at N=6,8,10."""
from .runtime import configure, torch_setup
ROOT = configure(gpu_only_compatible=True)

from datetime import datetime, timezone
import json
from pathlib import Path
import time
import torch
from .gpu_coherent_orbitals import CoherentOrbitals
from .gpu_chemistry_transfer import cuda_operators
from .gpu_manybody_ops import fixed_sector, molecular_words, dense_operator, spin_words
from .gpu_canonical_density import slater_vector

KCAL = 627.509474


@torch.no_grad()
def physical_vector(orbitals, coefficients, bits):
    occupied = torch.arange(orbitals.shape[-1], device=orbitals.device)
    vectors = torch.stack([slater_vector(value, occupied, bits) for value in orbitals])
    result = coefficients @ vectors
    return result/result.norm()


def main():
    device = torch_setup()
    protocol = json.loads(Path("research/local_sector_accuracy_protocol_20261116.json").read_text())
    folder = ROOT / "data/coherent_local_accuracy"
    folder.mkdir(parents=True, exist_ok=True)
    output = Path("research/local_sector_accuracy_gpu_20261116.json")
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "device": device,
              "protocol": protocol, "results": [], "failures": []}
    for case in protocol["cases"]:
        started = time.perf_counter()
        try:
            cache = folder / f"{case['id']}.pt"
            if cache.exists():
                data = torch.load(cache, map_location=device, weights_only=True)
            else:
                if "operator_cache" in case:
                    data = torch.load(ROOT / "data" / case["operator_cache"] / f"{case['id']}.pt", map_location=device, weights_only=True)
                else:
                    physical = {key: case[key] for key in ("id", "atom", "charge", "spin", "electrons")}
                    data = cuda_operators(physical, protocol["basis"])
                pairs, sites = case["electrons"]//2, len(data["h"])
                bits, configurations = fixed_sector(sites, pairs, pairs, device)
                words, coefficient = molecular_words(data["h"], data["eri"])
                matrix = dense_operator(words, coefficient, bits, sites)
                data.update(bits=bits, matrix=matrix)
                torch.save({key: value.cpu() if isinstance(value, torch.Tensor) else value for key, value in data.items()}, cache)
            seed = protocol["response_seed"]
            model = CoherentOrbitals(data["h"], data["eri"], case["electrons"]//2,
                                     proposed_modes=seed["proposed_modes"], displacement=seed["displacement"],
                                     representation="thouless")
            with torch.no_grad():
                initial_energy, coefficient, orbitals, _ = model()
                vector = physical_vector(orbitals, coefficient, data["bits"])
                initial_error = float((vector @ data["matrix"] @ vector-initial_energy).abs())
            if initial_error > 1e-8:
                raise RuntimeError(f"Local-sector cofactor kernel disagrees with independent fermionic action: {initial_error}")
            config = protocol["orbital_optimization"]
            optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
            schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config["steps"], eta_min=config["final_learning_rate"])
            best, best_coordinates = float(initial_energy), model.angles.detach().clone()
            history = []
            for step in range(config["steps"]):
                optimizer.zero_grad(set_to_none=True)
                energy, coefficient, orbitals, selected = model()
                energy.backward()
                if not bool(torch.isfinite(energy)) or not bool(torch.isfinite(model.angles.grad).all()):
                    raise RuntimeError("Nonfinite local-sector physical-state gradient")
                if float(energy.detach()) < best:
                    best, best_coordinates = float(energy.detach()), model.angles.detach().clone()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
                optimizer.step(); schedule.step()
                if step == 0 or (step+1) % 100 == 0:
                    row = {"id": case["id"], "step": step+1, "energy_hartree": float(energy.detach()+data["nuclear"]),
                           "retained_determinants": len(selected), "seconds": time.perf_counter()-started}
                    history.append(row); print(json.dumps(row), flush=True)
            with torch.no_grad():
                model.angles.copy_(best_coordinates)
                energy, coefficient, orbitals, selected = model()
                vector = physical_vector(orbitals, coefficient, data["bits"])
                independent = vector @ data["matrix"] @ vector+data["nuclear"]
                discrepancy = float((independent-energy-data["nuclear"]).abs())
                sw, sc = spin_words(len(data["h"]), device)
                spin_matrix = dense_operator(sw, sc, data["bits"], len(data["h"]))
                spin = vector @ spin_matrix @ vector
                if "reference_report" in case:
                    source = json.loads(Path(case["reference_report"]).read_text())
                    previous = next(row for row in source["results"] if row["id"] == case["id"])
                    exact = previous["exact_energy_hartree"]
                    exact_spin = 0.
                    new_reference = False
                else:
                    values, vectors = torch.linalg.eigh(data["matrix"])
                    exact = float(values[0]+data["nuclear"])
                    exact_spin = float(vectors[:, 0] @ spin_matrix @ vectors[:, 0])
                    new_reference = True
                error = float((independent-exact)*KCAL)
            row = {"id": case["id"], "electrons": case["electrons"], "spatial_orbitals": len(data["h"]),
                   "basis": protocol["basis"], "sector_dimension": len(data["bits"]),
                   "energy_error_kcal": error, "below_one_kcal": abs(error) < 1,
                   "total_energy_hartree": float(independent), "reference_energy_hartree": exact,
                   "spin_squared": float(spin), "reference_spin_squared": exact_spin,
                   "initial_kernel_error_hartree": initial_error, "final_kernel_error_hartree": discrepancy,
                   "retained_determinants": len(selected), "response_modes": model.response_modes,
                   "new_reference_generated": new_reference, "reference_energy_fitted": False,
                   "seconds": time.perf_counter()-started, "history": history}
            report["results"].append(row)
            torch.save({"orbitals": orbitals.cpu(), "coefficients": coefficient.cpu(), "report": row}, folder / f"{case['id']}.model.pt")
            print(json.dumps({key: value for key, value in row.items() if key != "history"}), flush=True)
        except Exception as exc:
            row = {"id": case["id"], "error": f"{type(exc).__name__}: {exc}"}
            report["failures"].append(row); print(json.dumps(row), flush=True)
        output.write_text(json.dumps(report, indent=2)+"\n")
    lookup = {row["id"]: row for row in report["results"]}
    report["balanced_contrast_errors_kcal"] = {}
    for label, first, second in (("BeH2_stretch", "BeH2_neutral_1.33", "BeH2_neutral_2.10"),
                                 ("NH3_supplied_geometry_contrast", "NH3_pyramidal", "NH3_planar")):
        if first in lookup and second in lookup:
            report["balanced_contrast_errors_kcal"][label] = lookup[second]["energy_error_kcal"]-lookup[first]["energy_error_kcal"]
    report["local_rejection_panel_passed"] = (not report["failures"] and len(report["results"]) == len(protocol["cases"])
                                              and all(row["below_one_kcal"] for row in report["results"])
                                              and all(abs(value) < 1 for value in report["balanced_contrast_errors_kcal"].values()))
    report["peak_cuda_bytes"] = torch.cuda.max_memory_allocated()
    output.write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    main()
