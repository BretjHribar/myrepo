"""Independent physical-wavefunction test of the new bounded spin split."""
from .runtime import configure, torch_setup
ROOT = configure(gpu_only_compatible=True)

from datetime import datetime, timezone
import json
from pathlib import Path
import torch
from .gpu_purified_spin import PurifiedSpinState, purified_norm
from .gpu_spin_response import SpinChain
from .gpu_tensor_gauge import canonical_at
from .gpu_two_site_tensor import TwoSiteSpinBasis, split_pair
from .gpu_verify_purified_operator import randomize, exhaustive_purification
from .physical_tensor_forward import physical_checks


@torch.no_grad()
def main():
    output = Path("research/bounded_spin_split_checks_gpu_20261117.json")
    if output.exists(): raise RuntimeError("Do not repeat this bounded split check")
    protocol = json.loads(Path("research/bounded_spin_split_checks_protocol_20261117.json").read_text())
    device = torch_setup(protocol["seed"])
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "device": device,
              "protocol": protocol, "status": "running", "failures": []}
    try:
        model = PurifiedSpinState(protocol["sites"], protocol["electrons"], channels=protocol["channels"], ancilla=1, device=device)
        randomize(model)
        state, _ = canonical_at(SpinChain.from_state(model).normalized(), protocol["center"])
        center = protocol["center"]
        basis = TwoSiteSpinBasis(state.labels[center], state.labels[center+2], device)
        merged = torch.einsum("ldm,mer->lder", state.cores[center][:, :, 0, :], state.cores[center+1][:, :, 0, :])
        parameters = basis.coordinates(merged)
        truncated, split = split_pair(state, center, basis, parameters, maximum_bond_dimension=protocol["maximum_bond_dimension"])
        before = exhaustive_purification(state.cores)[2][:, 0]
        after = exhaustive_purification(truncated.cores)[2][:, 0]
        loss = (before-after).square().sum()
        norm = after.square().sum()
        fidelity = (before @ after).square()/norm
        loss_error = abs(float(loss)-split["discarded_squared_norm"])
        norm_error = abs(float(norm)-1+split["discarded_squared_norm"])
        fidelity_error = float((fidelity-norm).abs())
        _, checks = physical_checks(truncated.normalized())
        valid = (max(loss_error, norm_error, fidelity_error) < protocol["identity_tolerance"] and checks["passed"]
                 and split["middle_bond_dimension"] <= protocol["maximum_bond_dimension"] and split["discarded_squared_norm"] > 1e-6)
        report["result"] = {"passed": valid, "split": split, "independent_discarded_norm_error": loss_error,
                            "independent_retained_norm_error": norm_error, "independent_fidelity_error": fidelity_error,
                            "physical_checks_after_normalization": checks}
        print(json.dumps(report["result"]), flush=True)
    except Exception as exc:
        report["failures"].append({"error": f"{type(exc).__name__}: {exc}"}); print(json.dumps(report["failures"][-1]), flush=True)
    report["status"] = "completed"; report["passed"] = not report["failures"] and report.get("result", {}).get("passed", False)
    report["peak_cuda_bytes"] = torch.cuda.max_memory_allocated()
    output.write_text(json.dumps(report, indent=2)+"\n")
    if not report["passed"]: raise RuntimeError("Bounded physical spin splitting failed its independent gate")


if __name__ == "__main__": main()
