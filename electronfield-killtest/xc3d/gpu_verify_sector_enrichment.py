"""Independent small-space check of physical residual sector enrichment."""
from .runtime import configure, torch_setup
ROOT = configure(gpu_only_compatible=True)

from datetime import datetime, timezone
import json
from pathlib import Path
import torch
from .gpu_spin_response import product_reference
from .gpu_tensor_gauge import canonical_at, rebase_bond
from .gpu_two_site_tensor import pair_problem, enrich_left_support, prefix_number_probability, split_pair
from .gpu_fermion_mpo import FermionMPO
from .physical_tensor_forward import update_site, physical_checks
from .gpu_verify_purified_operator import random_hamiltonian, exhaustive_purification
from .gpu_manybody_ops import molecular_words, dense_operator


@torch.no_grad()
def main():
    output = Path("research/sector_enrichment_checks_gpu_20261117.json")
    if output.exists(): raise RuntimeError("This sector enrichment check already has a report")
    protocol = json.loads(Path("research/sector_enrichment_checks_protocol_20261117.json").read_text())
    device = torch_setup(protocol["seed"])
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "device": device, "protocol": protocol,
              "status": "running", "failures": []}
    try:
        if not json.loads(Path("research/two_site_tensor_checks_gpu_20261117.json").read_text())["passed"]:
            raise RuntimeError("Independent two-site operator checks must pass first")
        h, eri = random_hamiltonian(protocol["sites"], device)
        mpo = FermionMPO.molecular(h, eri)
        state, _ = canonical_at(product_reference(protocol["sites"], protocol["electrons"], 0, device), protocol["center"])
        state, _ = rebase_bond(state, protocol["center"]+1, "left")
        problem = pair_problem(mpo, state, protocol["center"])
        seed, basis = problem["seed"], problem["basis"]
        image = problem["action"](seed)
        residual = image-(seed @ image)*seed
        enriched, support = enrich_left_support(state, protocol["center"], basis, residual*basis.inverse_norm,
            maximum_bond_dimension=protocol["maximum_bond_dimension"],
            maximum_added_magnetic_states=protocol["maximum_added_magnetic_states"])
        before = exhaustive_purification(state.cores)[2]
        embedded = exhaustive_purification(enriched.cores)[2]
        retention = float((before-embedded).norm())
        settings = json.loads(Path("research/tensor_water_forward_protocol_20261117.json").read_text())
        updated, update = update_site(mpo, enriched, protocol["center"]+1, settings)
        wave = exhaustive_purification(updated.cores)[2][:, 0]
        columns = []
        for index in range(basis.dimension):
            values = seed.new_zeros(basis.dimension); values[index] = basis.inverse_norm[index]
            column, _ = split_pair(state, protocol["center"], basis, values)
            columns.append(exhaustive_purification(column.cores)[2][:, 0])
        frame = torch.stack(columns, 1)
        words, coefficient = molecular_words(h, eri, threshold=0.)
        dense = dense_operator(words, coefficient, torch.arange(4**protocol["sites"], device=device), protocol["sites"])
        expected = torch.linalg.eigvalsh(frame.T @ dense @ frame)[0]
        literal_error = float((wave @ dense @ wave-mpo.expectation(updated.cores)).abs())
        minimum_error = float((expected-mpo.expectation(updated.cores)).abs())
        _, checks = physical_checks(updated)
        old_energy = float(mpo.expectation(state.cores)); new_energy = float(mpo.expectation(updated.cores))
        old_probability = prefix_number_probability(state, protocol["center"]+1, 3)
        new_probability = prefix_number_probability(updated, protocol["center"]+1, 3)
        valid = (retention < protocol["state_tolerance"] and max(literal_error, minimum_error) < protocol["energy_tolerance_hartree"]
                 and checks["passed"] and new_energy < old_energy and support["added_magnetic_states"] == 3
                 and old_probability == 0. and new_probability > 1e-8)
        report["result"] = {"passed": valid, "support": support, "independent_retained_wavefunction_error": retention,
                            "independent_literal_energy_error_hartree": literal_error,
                            "independent_complete_pair_minimum_error_hartree": minimum_error,
                            "energy_lowering_hartree": old_energy-new_energy,
                            "old_new_sector_probability": old_probability, "new_sector_probability": new_probability,
                            "physical_checks": checks, "local_update": update}
        print(json.dumps(report["result"]), flush=True)
    except Exception as exc:
        report["failures"].append({"error": f"{type(exc).__name__}: {exc}"}); print(json.dumps(report["failures"][-1]), flush=True)
    report["status"] = "completed"; report["passed"] = not report["failures"] and report.get("result", {}).get("passed", False)
    report["peak_cuda_bytes"] = torch.cuda.max_memory_allocated()
    output.write_text(json.dumps(report, indent=2)+"\n")
    if not report["passed"]: raise RuntimeError("Physical residual sector enrichment failed its independent gate")


if __name__ == "__main__": main()
