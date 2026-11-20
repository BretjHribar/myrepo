"""Recover the completed molecular conditional solve with zero new iterations."""
from .runtime import configure, torch_setup
ROOT = configure(gpu_only_compatible=True)

from datetime import datetime, timezone
import json
from pathlib import Path
import torch
from .gpu_spin_response import SpinChain
from .tensor_auxiliary_model import UniversalTensorOperators


@torch.no_grad()
def main():
    output = Path("research/tensor_section_recovery_gpu_20261117.json")
    if output.exists(): raise RuntimeError("Completed section recovery already has an authoritative check")
    protocol = json.loads(Path("research/tensor_water_conditional_protocol_20261117.json").read_text())
    completed = json.loads(Path("research/tensor_water_conditional_gpu_20261117.json").read_text())
    if not completed["passed"] or completed["status"] != "completed": raise RuntimeError("There is no accepted completed conditional state to recover")
    device = torch_setup(17112053)
    data = torch.load(ROOT / protocol["source_input"], map_location=device, weights_only=True)
    saved = torch.load(ROOT / protocol["source_state"], map_location=device, weights_only=True)
    context = SpinChain(saved["cores"], saved["labels"], saved["electrons"], saved["twice_spin"])
    cache = ROOT / protocol["cache_folder"]
    operators = torch.load(cache / "spatial_frame.pt", map_location=device, weights_only=True)["operators"]
    domain = UniversalTensorOperators({"kinetic": data["kinetic"], "eri": data["eri"], "operators": operators,
                                       "electrons": protocol["electrons"], "twice_spin": protocol["twice_spin"]})
    compiled = torch.load(cache / "compiled_base.pt", map_location=device, weights_only=True)
    compiled["density_chunks"] = [torch.load(path, map_location=device, weights_only=True) for path in sorted(cache.glob("density_*.pt"))]
    recompiled = []
    section = domain.section(context, protocol["center"], protocol, compiled=compiled,
                              checkpoint_callback=lambda value: recompiled.append(value["phase"]))
    checkpoint = torch.load(cache / "central.optimizer.pt", map_location=device, weights_only=True)
    original = torch.load(cache / "central_result.pt", map_location=device, weights_only=True)
    updates = []
    recovered = section.evaluate(checkpoint["auxiliary"], checkpoint=checkpoint,
                                  checkpoint_callback=lambda value: updates.append(value["iteration"]))
    state_error = float((recovered["state"]-original["state"]).norm())
    gradient_error = float((recovered["gradient"]-original["gradient"]).norm())
    scalar_error = float((recovered["functional"]-original["functional"]).abs())
    effective = recovered["effective"]
    gap = float((effective*recovered["state"]).sum()-torch.linalg.eigvalsh((effective+effective.T)/2)[0])
    valid = (not updates and not recompiled and state_error == 0. and gradient_error < 1e-12 and scalar_error == 0.
             and recovered["iteration"] == original["iteration"] and abs(gap) < 1e-10)
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "device": device, "passed": valid,
              "compiled_density_chunks_restored": len(compiled["density_chunks"]), "local_operator_matrices_recompiled": len(recompiled),
              "universal_mpo_rebuilt_from_saved_TW": True,
              "new_optimizer_iterations": len(updates), "completed_iteration": recovered["iteration"],
              "physical_state_difference": state_error, "auxiliary_gradient_difference": gradient_error,
              "conditional_scalar_difference_hartree": scalar_error, "independently_recomputed_conditional_gap_hartree": gap,
              "new_chemical_or_reference_solve": False, "peak_cuda_bytes": torch.cuda.max_memory_allocated()}
    output.write_text(json.dumps(report, indent=2)+"\n"); print(json.dumps(report), flush=True)
    if not valid: raise RuntimeError("Completed tensor section could not be recovered without repeating work")


if __name__ == "__main__": main()
