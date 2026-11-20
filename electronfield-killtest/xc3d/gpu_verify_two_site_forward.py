"""Independent six-orbital verification of the variable-sector forward model."""
from .runtime import configure, torch_setup
ROOT = configure(gpu_only_compatible=True)

from datetime import datetime, timezone
import json
from pathlib import Path
import torch
from .gpu_spin_response import product_reference
from .gpu_fermion_mpo import FermionMPO
from .gpu_verify_purified_operator import random_hamiltonian
from .gpu_manybody_ops import fixed_sector, molecular_words, dense_operator, spin_words
from .physical_two_site_forward import solve_two_site


@torch.no_grad()
def main():
    output = Path("research/two_site_forward_checks_gpu_20261117.json")
    if output.exists(): raise RuntimeError("Do not repeat the new two-site forward check")
    protocol = json.loads(Path("research/two_site_forward_checks_protocol_20261117.json").read_text())
    device = torch_setup(protocol["seed"])
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "device": device, "protocol": protocol,
              "status": "running", "history": [], "failures": []}
    def log(row):
        report["history"].append(row); print(json.dumps(row), flush=True)
        output.write_text(json.dumps(report, indent=2)+"\n")
    try:
        if not json.loads(Path("research/bounded_spin_split_checks_gpu_20261117.json").read_text())["passed"]:
            raise RuntimeError("The bounded physical split must pass its independent check")
        h, eri = random_hamiltonian(protocol["sites"], device)
        operator = FermionMPO.molecular(h, eri)
        initial = product_reference(protocol["sites"], protocol["electrons"], 0, device)
        saved = None
        class DeliberateInterruption(Exception): pass
        def interrupt(value):
            nonlocal saved
            if len(value["history"]) == 1:
                saved = value
                raise DeliberateInterruption()
        try:
            solve_two_site(operator, initial, protocol, checkpoint_callback=interrupt, callback=log)
        except DeliberateInterruption:
            pass
        if saved is None: raise RuntimeError("The first completed pair update was not checkpointed")
        result = solve_two_site(operator, initial, protocol, checkpoint=saved, callback=log)
        # The full-matrix reference is neither available to the solver nor
        # consulted before its frozen stopping/audit has finished.
        bits, _ = fixed_sector(protocol["sites"], 3, 3, device)
        words, coefficients = molecular_words(h, eri, threshold=0.)
        dense = dense_operator(words, coefficients, bits, protocol["sites"])
        sw, sc = spin_words(protocol["sites"], device)
        casimir = dense_operator(sw, sc, bits, protocol["sites"])
        spins, rotation = torch.linalg.eigh(casimir)
        singlets = rotation[:, spins.abs() < 1e-10]
        exact = torch.linalg.eigvalsh(singlets.T @ dense @ singlets)[0]
        error = abs(result["electronic_energy_hartree"]-float(exact))
        entries = [(row["sweep"], row["position"]) for row in result["history"]]
        duplicates = len(entries)-len(set(entries))
        valid = result["coordinate_stationarity_passed"] and result["physical_checks"]["passed"] and duplicates == 0 and error < protocol["independent_energy_tolerance_hartree"]
        report["result"] = {"passed": valid, "independent_spin_sector_energy_error_hartree": error,
                            "maximum_common_context_gap_hartree": result["maximum_common_context_gap_hartree"],
                            "physical_checks": result["physical_checks"], "sweeps": len(result["sweep_history"]),
                            "updates": len(entries), "audit_pairs": len(result["audit"]), "duplicate_updates": duplicates,
                            "rejected_updates": sum(not row["accepted"] for row in result["history"]),
                            "interrupted_first_completed_update_resumed": True,
                            "reference_computed_after_prediction": True, "full_hilbert_residual_claimed": False}
        log({"phase": "two_site_forward_check_completed", **report["result"]})
    except Exception as exc:
        report["failures"].append({"error": f"{type(exc).__name__}: {exc}"}); log({"phase": "failure", **report["failures"][-1]})
    report["status"] = "completed"; report["passed"] = not report["failures"] and report.get("result", {}).get("passed", False)
    report["peak_cuda_bytes"] = torch.cuda.max_memory_allocated()
    output.write_text(json.dumps(report, indent=2)+"\n")
    if not report["passed"]: raise RuntimeError("The variable-sector forward model failed its independent test")


if __name__ == "__main__": main()
