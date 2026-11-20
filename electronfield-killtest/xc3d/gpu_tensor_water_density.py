"""Density contract of the saved larger-basis state, without repeating its solve."""
from .runtime import configure, torch_setup
ROOT = configure(gpu_only_compatible=True)

from datetime import datetime, timezone
import json
from pathlib import Path
import torch
from .gpu_spin_response import SpinChain
from .gpu_tensor_density import density_matrix


def main():
    output = Path("research/tensor_water_density_gpu_20261117.json")
    if output.exists(): raise RuntimeError("The saved molecular density has already been assessed")
    gate = json.loads(Path("research/tensor_density_checks_gpu_20261117.json").read_text())
    if not gate["passed"]: raise RuntimeError("Tensor density interface has not passed its independent tests")
    device = torch_setup(17112047)
    name = "H2O_ccpvdz_tensor_scope_new"
    source = ROOT / "data/tensor_scope" / f"{name}.physical_metric.state.pt"
    saved = torch.load(source, map_location=device, weights_only=True)
    state = SpinChain(saved["cores"], saved["labels"], saved["electrons"], saved["twice_spin"])
    matrix = density_matrix(state)
    with torch.no_grad():
        occupations = torch.linalg.eigvalsh(matrix)
        number_error = float((matrix.trace()-state.electrons).abs())
        hermiticity = float((matrix-matrix.T).norm())
        valid = (bool(torch.isfinite(matrix).all()) and max(number_error, hermiticity) < 1e-8
                 and float(occupations.min()) >= -1e-8 and float(occupations.max()) <= 2+1e-8)
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "device": device, "id": name,
              "passed": valid, "spatial_orbitals": len(matrix), "electrons": state.electrons,
              "number_error": number_error, "hermiticity_error": hermiticity,
              "minimum_spatial_occupation": float(occupations.min()), "maximum_spatial_occupation": float(occupations.max()),
              "physical_density_method": "Fixed-state derivative of the complete literal one-body tensor operator",
              "global_occupation_vector_allocated": False, "state_solve_repeated": False,
              "new_reference_generated_or_loaded": False, "xc_functional_evaluated": False,
              "chemical_accuracy_measured": False, "wtmad2_measured": False,
              "peak_cuda_bytes": torch.cuda.max_memory_allocated()}
    torch.save({"density": matrix.detach().cpu(), "occupations": occupations.cpu(), "report": report},
               ROOT / "data/tensor_scope" / f"{name}.physical_metric.density.pt")
    output.write_text(json.dumps(report, indent=2)+"\n"); print(json.dumps(report), flush=True)
    if not valid: raise RuntimeError("Saved tensor state did not pass the physical density contract")


if __name__ == "__main__": main()
