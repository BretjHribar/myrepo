"""Check state/energy/density retention and inspect the failed seed sectors."""
from .runtime import configure, torch_setup
ROOT = configure(gpu_only_compatible=True)

from datetime import datetime, timezone
import argparse
import json
from pathlib import Path
import torch
from .physical_tensor_forward import restore, payload
from .gpu_tensor_gauge import canonical_at
from .gpu_local_tensor import physical_local_space, norm_environments
from .gpu_fermion_mpo import FermionMPO
from .gpu_tensor_density import density_matrix
from .gpu_tensor_water_conditional import atomic_save


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="research/tensor_gauge_recovery_gpu_20261117.json")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists(): raise RuntimeError("This state-preserving recovery already has a report")
    device = torch_setup(17112056)
    checkpoint = torch.load(ROOT / "data/tensor_forward_water/forward_checkpoint.pt", map_location=device, weights_only=True)["run"]
    state = restore(checkpoint["state"])
    data = torch.load(ROOT / "data/tensor_scope/H2O_ccpvdz_tensor_scope_new.inputs.pt", map_location=device, weights_only=True)
    minimum_prefix = min(label[0] for label in state.labels[state.electrons//2])
    restriction = {"cut_after_initial_occupied_orbitals": state.electrons//2,
                   "minimum_prefix_electrons": minimum_prefix,
                   "maximum_virtual_electrons_permitted_by_seed_sectors": state.electrons-minimum_prefix}
    old_density = density_matrix(state)
    operator = FermionMPO.molecular(data["h"], data["eri"])
    old_energy = operator.expectation(state.cores)
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "device": device,
              "seed_sector_restriction": restriction, "old_completed_updates": len(checkpoint["history"]),
              "history": [], "status": "running", "failures": [],
              "method": "Direct reduced-tensor SVD with reconstructed orthogonal factors; no inverse-small-singular-value multiplication"}
    def log(row):
        report["history"].append(row); print(json.dumps(row), flush=True)
        output.write_text(json.dumps(report, indent=2)+"\n")
    try:
        recovered, history = canonical_at(state, 4, log)
        energy = operator.expectation(recovered.cores)
        density = density_matrix(recovered)
        energy_error = float((energy-old_energy).abs())
        density_error = float((density-old_density).norm())
        space = physical_local_space(recovered, 4)
        left, right = norm_environments(recovered.cores, 4)
        metric_error = max(float((left-torch.eye(len(left), device=device)).norm()), float((right-torch.eye(len(right), device=device)).norm()))
        valid = max(energy_error, density_error, metric_error, space["literal_metric_orthogonality_error"]) < 1e-8
        report["result"] = {"passed": valid, "physical_energy_change_hartree": energy_error, "density_change": density_error,
                            "left_right_canonical_metric_error": metric_error, "local_metric_error": space["literal_metric_orthogonality_error"],
                            "physical_local_dimension": space["physical_dimension"], "original_input_retention_error": space["input_retention_error"],
                            "optimizer_steps_repeated": 0, "precision_tolerance_relaxed": False}
        atomic_save({**payload(recovered), "report": report["result"]}, ROOT / "data/tensor_forward_water/canonical_recovery.pt")
        print(json.dumps(report["result"]), flush=True)
    except Exception as exc:
        report["failures"].append({"error": f"{type(exc).__name__}: {exc}"}); print(json.dumps(report["failures"][-1]), flush=True)
    report["status"] = "completed"; report["passed"] = not report["failures"] and report.get("result", {}).get("passed", False)
    report["peak_cuda_bytes"] = torch.cuda.max_memory_allocated()
    output.write_text(json.dumps(report, indent=2)+"\n")
    if not report["passed"]: raise RuntimeError("Exact tensor rebasing failed physical retention/metric checks")


if __name__ == "__main__": main()
