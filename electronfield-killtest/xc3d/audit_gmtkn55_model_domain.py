"""Full model-domain inventory using only text, integer metadata and storage shapes.

No integrals, SCF, scientific floating-point computation or benchmark energy
evaluation runs on the CPU. PySCF is used only to parse basis definitions.
"""
from .runtime import configure
ROOT = configure(gpu_only_compatible=True)

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from pyscf import gto
from .gmtkn55 import GMTKN55Suite


def main():
    output = Path("research/gmtkn55_model_domain_audit_20261117.json")
    if output.exists(): raise RuntimeError("Do not repeat this completed domain inventory")
    scoring = json.loads(Path("research/gmtkn55_scoring_protocol_20261117.json").read_text())
    protocol_path = Path("research/gmtkn55_model_domain_protocol_20261117.json")
    protocol_bytes = protocol_path.read_bytes()
    protocol = json.loads(protocol_bytes)
    if protocol["benchmark_commit"] != scoring["source_commit"]: raise RuntimeError("Mismatched benchmark version")
    suite = GMTKN55Suite(ROOT, scoring)
    source_hashes = {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in protocol["source_files"]}
    source_sha256 = hashlib.sha256(json.dumps(source_hashes, sort_keys=True).encode()).hexdigest()
    candidate = {"id": protocol["candidate_id"], "basis": protocol["basis"],
                 "protocol_sha256": hashlib.sha256(protocol_bytes).hexdigest(), "source_sha256": source_sha256}
    source_folder = ROOT/"data/gmtkn55_official_v1/data"
    atom_basis = {}
    statuses, structures = [], []
    for name in suite.required_ids:
        metadata = suite.structures[name]
        atoms = [line.split()[0] for line in (source_folder/metadata["coordinate_path"]).read_text().splitlines()[2:2+metadata["atom_count"]]]
        orbitals, ecp_elements, problems = 0, [], []
        for symbol in atoms:
            if symbol not in atom_basis:
                basis = gto.basis.load(protocol["basis"], symbol)
                # Shell angular momentum and contraction counts are integer
                # shape metadata. No basis-value arithmetic is performed.
                functions = sum((2*shell[0]+1)*(len(shell[1])-1) for shell in basis)
                ecp = gto.basis.load_ecp(protocol["basis"], symbol)
                atom_basis[symbol] = {"spherical_basis_functions": functions,
                                      "ecp_core_electrons": int(ecp[0]) if ecp else 0}
            orbitals += atom_basis[symbol]["spherical_basis_functions"]
            if atom_basis[symbol]["ecp_core_electrons"]: ecp_elements.append(symbol)
        # Exact integer enumeration counts of the current molecular_words
        # loops, before any numerical coefficient thresholding.
        one_words = 2*orbitals**2
        two_words = 2*orbitals**4+2*orbitals**2*(orbitals-1)**2
        words = one_words+two_words
        minimum_cuda_input = 8*orbitals**4+96*words+8*words+8*two_words
        estimated_host_words = protocol["estimated_host_bytes_per_enumerated_word"]*words
        if metadata["unpaired_electrons"] not in protocol["supported_physical_twice_spins"]:
            problems.append("physical_triplet_or_quartet_domain_not_validated")
        if ecp_elements and not protocol["ecp_implementation_validated"]:
            problems.append("target_basis_requires_unvalidated_gpu_ecp_operator")
        if minimum_cuda_input > protocol["gpu_input_storage_limit_bytes"]:
            problems.append("literal_operator_inputs_exceed_cuda_allocation_bound")
        if estimated_host_words > protocol["host_word_enumeration_budget_bytes"]:
            problems.append("literal_word_enumeration_exceeds_host_memory_budget")
        status = "unsupported" if problems else "uncomputed"
        reason = "; ".join(problems) if problems else "State-prediction capacity and chemical accuracy are unmeasured in this basis; successful allocation and convergence are not inferred from this preflight"
        statuses.append({"id": name, "status": status, "reason": reason})
        structures.append({"id": name, "physical_electrons": metadata["physical_electrons"],
            "physical_twice_spin": metadata["unpaired_electrons"], "spatial_basis_functions": orbitals,
            "ecp_elements": sorted(set(ecp_elements)), "literal_word_count": words,
            "minimum_cuda_operator_input_bytes": minimum_cuda_input,
            "estimated_host_word_enumeration_bytes": estimated_host_words,
            "state_prediction_preflight_blockers": problems, "state_prediction_computed": False})
    lookup = {row["id"]: row for row in structures}
    reactions = [{"id": row["id"], "subset": row["subset"],
                  "state_preflight_blocked_structure_ids": [name for name in row["structure_ids"] if lookup[name]["state_prediction_preflight_blockers"]]}
                 for row in suite.reactions]
    subset_coverage = {}
    for name in suite.subsets:
        subset = [row for row in reactions if row["subset"] == name]
        subset_coverage[name] = {"total_reactions": len(subset),
            "reactions_with_no_identified_state_preflight_blocker": sum(not row["state_preflight_blocked_structure_ids"] for row in subset),
            "actual_scored_reactions": 0}
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "scope": protocol["purpose"],
              "protocol": protocol, "candidate": candidate, "source_files_sha256": source_hashes,
              "manifest_sha256": suite.manifest_sha256, "basis_metadata": atom_basis,
              "required_structures": len(structures), "total_reactions": len(reactions), "evaluation_sets": len(subset_coverage),
              "status_counts": dict(Counter(row["status"] for row in statuses)),
              "blocker_counts": dict(Counter(reason for row in structures for reason in row["state_prediction_preflight_blockers"])),
              "minimum_spatial_basis_functions": min(row["spatial_basis_functions"] for row in structures),
              "maximum_spatial_basis_functions": max(row["spatial_basis_functions"] for row in structures),
              "subset_coverage": subset_coverage, "structures": structures, "reactions": reactions,
              "current_complete_xc_model_runnable_on_full_suite": False,
              "actual_model_predictions": 0, "initial_full_suite_estimate_kcal_per_mol": None,
              "wtmad2_measured": False, "goal_accuracy_achieved": False,
              "cpu_scientific_arithmetic": False, "benchmark_reference_energies_used_for_domain_selection": False,
              "locked_confirmation_panel_opened": False, "speed_work_performed": False}
    prediction_file = ROOT/"data/gmtkn55_pipeline/current_candidate_coverage.json"
    prediction_file.write_text(json.dumps({"schema": "gmtkn55_total_energies_v1", "benchmark_commit": scoring["source_commit"],
        "manifest_sha256": suite.manifest_sha256, "candidate": candidate, "units": "hartree", "energy_kind": scoring["prediction_energy_kind"],
        "benchmark_reference_used_for_prediction": False, "predictions": statuses}, indent=2)+"\n")
    output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({key: report[key] for key in ("required_structures", "total_reactions", "evaluation_sets", "status_counts", "blocker_counts", "minimum_spatial_basis_functions", "maximum_spatial_basis_functions", "current_complete_xc_model_runnable_on_full_suite", "initial_full_suite_estimate_kcal_per_mol")}), flush=True)


if __name__ == "__main__": main()
