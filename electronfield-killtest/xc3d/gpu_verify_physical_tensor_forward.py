"""Independent all-site/reference and checkpoint checks for the forward model."""
from .runtime import configure, torch_setup
ROOT = configure(gpu_only_compatible=True)

from datetime import datetime, timezone
import json
from pathlib import Path
import torch
from .gpu_purified_spin import PurifiedSpinState
from .gpu_spin_response import SpinChain
from .gpu_fermion_mpo import FermionMPO
from .gpu_manybody_ops import molecular_words, dense_operator, spin_words, fixed_sector
from .gpu_verify_purified_operator import random_hamiltonian, randomize
from .physical_tensor_forward import solve_forward


@torch.no_grad()
def main():
    output = Path("research/physical_tensor_forward_checks_gpu_20261117.json")
    if output.exists(): raise RuntimeError("This new all-site check already has a report")
    protocol = json.loads(Path("research/physical_tensor_forward_checks_protocol_20261117.json").read_text())
    device = torch_setup(protocol["seed"])
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "device": device, "protocol": protocol,
              "results": [], "failures": [], "status": "running"}
    try:
        h, eri = random_hamiltonian(protocol["sites"], device)
        operator = FermionMPO.molecular(h, eri)
        for case in protocol["cases"]:
            state = PurifiedSpinState(protocol["sites"], case["electrons"], case["twice_spin"], channels=protocol["channels"], ancilla=1, device=device)
            randomize(state); initial = SpinChain.from_state(state)
            saved, interrupted = None, False
            class DeliberateInterruption(Exception): pass
            def stop_after_one(value):
                nonlocal saved, interrupted
                if value["stage"] == "sweep" and len(value["history"]) == 1:
                    saved, interrupted = value, True
                    raise DeliberateInterruption()
            if case["twice_spin"] == 0:
                try:
                    solve_forward(operator, initial, protocol, checkpoint_callback=stop_after_one)
                except DeliberateInterruption:
                    pass
                if not interrupted: raise RuntimeError("The intended first completed update was not checkpointed")
                result = solve_forward(operator, initial, protocol, checkpoint=saved)
            else:
                result = solve_forward(operator, initial, protocol)
            # The model and its stopping rule have completed before reference work.
            up, down = (case["electrons"]+case["twice_spin"])//2, (case["electrons"]-case["twice_spin"])//2
            bits, _ = fixed_sector(protocol["sites"], up, down, device)
            words, coefficients = molecular_words(h, eri, threshold=0.)
            dense = dense_operator(words, coefficients, bits, protocol["sites"])
            sw, sc = spin_words(protocol["sites"], device)
            casimir = dense_operator(sw, sc, bits, protocol["sites"])+(case["twice_spin"]/2)**2*torch.eye(len(bits), device=device)
            spins, rotation = torch.linalg.eigh(casimir)
            target = (case["twice_spin"]/2)*(case["twice_spin"]/2+1)
            spin_basis = rotation[:, (spins-target).abs() < 1e-10]
            exact = torch.linalg.eigvalsh(spin_basis.T @ dense @ spin_basis)[0]
            error = float((exact-result["electronic_energy_hartree"]).abs())
            entries = [(row["sweep"], row["position"]) for row in result["history"]]
            duplicates = len(entries)-len(set(entries))
            valid = (result["physical_checks"]["passed"] and result["coordinate_stationarity_passed"] and duplicates == 0
                     and error < protocol["independent_reference_tolerance_hartree"])
            row = {"case": case, "passed": valid, "reference_energy_difference_hartree": error,
                   "common_context_maximum_gap_hartree": result["maximum_common_context_gap_hartree"],
                   "physical_checks": result["physical_checks"], "sweeps": len(result["sweep_history"]),
                   "updates": len(entries), "audit_centers": len(result["audit"]), "duplicate_updates": duplicates,
                   "interrupted_first_update_resumed": interrupted, "reference_computed_after_prediction": True,
                   "full_hilbert_stationarity_claimed": False}
            report["results"].append(row); print(json.dumps(row), flush=True)
            output.write_text(json.dumps(report, indent=2)+"\n")
            if not valid: raise RuntimeError("New all-site forward algorithm failed its independent reference or stationarity check")
    except Exception as exc:
        report["failures"].append({"error": f"{type(exc).__name__}: {exc}"}); print(json.dumps(report["failures"][-1]), flush=True)
    report["status"] = "completed"
    report["passed"] = len(report["results"]) == len(protocol["cases"]) and not report["failures"] and all(row["passed"] for row in report["results"])
    report["peak_cuda_bytes"] = torch.cuda.max_memory_allocated()
    output.write_text(json.dumps(report, indent=2)+"\n")
    if not report["passed"]: raise RuntimeError("All-site forward calculation is not approved for molecular use")


if __name__ == "__main__": main()
