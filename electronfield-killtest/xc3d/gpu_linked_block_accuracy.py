"""Accuracy-first coupled-block tests, including a difficult H4 geometry."""
from .runtime import configure, torch_setup
ROOT = configure(gpu_only_compatible=True)

from datetime import datetime, timezone
import json
from pathlib import Path
import time
import torch
from .gpu_chemistry_transfer import cuda_operators
from .gpu_coherent_blocks import neutral_two_block
from .gpu_coherent_orbitals import CoherentOrbitals
from .gpu_four_particle import FourParticleOperator, cached_ritz

KCAL = 627.509474


@torch.no_grad()
def main():
    device = torch_setup()
    protocol = json.loads(Path("research/linked_block_accuracy_protocol_20261116.json").read_text())
    output = Path("research/linked_block_accuracy_gpu_20261116.json")
    folder = ROOT / "data/linked_block_accuracy"
    folder.mkdir(parents=True, exist_ok=True)
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "device": device, "protocol": protocol,
              "results": [], "failures": []}
    for case in protocol["cases"]:
        started = time.perf_counter()
        try:
            path = folder / f"{case['id']}.pt"
            if path.exists():
                data = torch.load(path, map_location=device, weights_only=True)
            else:
                physical = {key: value for key, value in case.items() if key != "first_block_atoms"}
                data = cuda_operators(physical, protocol["basis"])
                torch.save({key: value.cpu() if isinstance(value, torch.Tensor) else value for key, value in data.items()}, path)
            split = data["atom_ao_slices"][case["first_block_atoms"]-1][1]
            h, eri = data["local_h"], data["local_eri"]
            model = neutral_two_block(h, eri, split, data["nuclear"])
            print(json.dumps({"id": case["id"], "phase": "prediction_stopped", "energy_hartree": float(model["energy"]),
                              "projected_dimension": model["dimension"]}), flush=True)
            # Generate the independent full-space label only after prediction.
            operator = FourParticleOperator(h, eri, 4, data["nuclear"])
            random, _ = torch.linalg.qr(torch.randn(len(h), 2, device=device), mode="reduced")
            seed_orbitals = torch.stack((data["reference_orbitals"][:, :2], random))
            seed = CoherentOrbitals.four_electron_vector(seed_orbitals, h.new_tensor([1., .03]))
            reference = cached_ritz(operator, seed, maximum=protocol["maximum_reference_dimension"],
                                    tolerance=protocol["reference_residual_tolerance"])
            error = float((model["energy"]-reference["energy"])*KCAL) if reference["converged"] else None
            row = {"id": case["id"], "basis": protocol["basis"], "electrons": 4, "spatial_orbitals": len(h),
                   "model_energy_hartree": float(model["energy"]), "reference_energy_hartree": float(reference["energy"]),
                   "reference_converged": reference["converged"], "reference_residual": reference["residual"],
                   "energy_error_kcal": error, "below_one_kcal": error is not None and abs(error) < 1,
                   "coupling_lowering_from_product_kcal": float((model["energy"]-model["product_energy"])*KCAL),
                   "block_dimensions": model["block_dimensions"], "projected_dimension": model["dimension"],
                   "block_number_errors": model["block_number_errors"], "block_metric_errors": model["block_metric_errors"],
                   "leading_block_schmidt_values": model["block_entanglement_singular_values"][:8].tolist(),
                   "input_reference_status": data["reference_status"], "reference_energy_fitted": False,
                   "seconds": time.perf_counter()-started}
            report["results"].append(row)
            torch.save({"state_tensor": reference["state"].cpu(), "report": row}, folder / f"{case['id']}.reference.pt")
            print(json.dumps(row), flush=True)
        except Exception as exc:
            row = {"id": case["id"], "error": f"{type(exc).__name__}: {exc}"}
            report["failures"].append(row); print(json.dumps(row), flush=True)
        output.write_text(json.dumps(report, indent=2)+"\n")
    cases = {row["id"]: row for row in report["results"]}
    if all(name in cases and cases[name]["reference_converged"] for name in ("H4_parallel_far", "H4_square_close")):
        report["balanced_H4_contrast_error_kcal"] = cases["H4_square_close"]["energy_error_kcal"]-cases["H4_parallel_far"]["energy_error_kcal"]
    report["accuracy_rejection_panel_passed"] = (not report["failures"] and len(report["results"]) == len(protocol["cases"])
                                                   and all(row["below_one_kcal"] for row in report["results"])
                                                   and abs(report.get("balanced_H4_contrast_error_kcal", float("inf"))) < 1)
    report["peak_cuda_bytes"] = torch.cuda.max_memory_allocated()
    output.write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    main()
