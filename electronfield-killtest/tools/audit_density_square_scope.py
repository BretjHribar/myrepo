"""Exact integer storage inventory for the expanded operator implementation.

Shape/graph/storage metadata only; no CPU energies, tensor numerics or
scientific fallback. Every public reaction and structure stays in scope.
"""
import ctypes
import json
from datetime import datetime, timezone
from pathlib import Path


def main():
    ctypes.windll.kernel32.SetProcessAffinityMask(ctypes.windll.kernel32.GetCurrentProcess(), 1)
    root = Path(__file__).resolve().parents[2]
    research = root/"electronfield-killtest/research"
    destination = research/"density_square_full_scope_audit_20261117.json"
    if destination.exists(): raise RuntimeError("Do not repeat this terminal scope audit")
    previous = json.loads((research/"gmtkn55_model_domain_audit_20261117.json").read_text())
    budget = json.loads((research/"resource_budget.json").read_text())
    device_bytes = int(budget["gpu"]["observed_memory_mib"])*1024*1024
    checks = json.loads((research/"density_square_operator_checks_gpu_20261117.json").read_text())
    if not checks["passed"]: raise RuntimeError("Literal operator implementation checks must pass first")
    rows = []
    for structure in previous["structures"]:
        n = structure["spatial_basis_functions"]
        nonzeros = 16*(n-2)*(n-1)*(2*n-3)//6 if n > 2 else 0
        rows.append({"id": structure["id"], "spatial_basis_functions": n,
            "parity_product_nonzeros_alone": nonzeros,
            "minimum_expanded_sparse_product_bytes": nonzeros*24,
            "estimated_explicit_one_body_host_metadata_bytes": 2*n*n*256,
            "expanded_product_exceeds_frozen_500MB_cap": nonzeros*24 > 500_000_000,
            "expanded_product_exceeds_entire_observed_device": nonzeros*24 > device_bytes,
            "one_body_host_metadata_exceeds_1GB_guard": 2*n*n*256 > 1_000_000_000})
    lookup = {row["id"]: row for row in rows}
    # Reaction support must use actual full definitions, not only the older
    # implementation's blocker list. Import the text-only parser without any
    # numerical-package initialization.
    import sys
    sys.path.insert(0, str(root/"electronfield-killtest"))
    from xc3d.gmtkn55 import GMTKN55Suite
    protocol = json.loads((research/"gmtkn55_scoring_protocol_20261117.json").read_text())
    suite = GMTKN55Suite(root/".research", protocol)
    reactions = [{"id": row["id"], "subset": row["subset"],
        "blocked_structure_ids": sorted({name for name in row["structure_ids"] if lookup[name]["expanded_product_exceeds_frozen_500MB_cap"]})} for row in suite.reactions]
    n = checks["larger_operator_summary"]["sites"]
    measured = checks["larger_operator_summary"]["squared_density_operator"]["cuda_operator_storage_bytes"]
    formula_consistent = 24*16*(n-2)*(n-1)*(2*n-3)//6 <= measured
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "scope": "All 2442 required structures and 1505 public reactions; integer implementation-storage bounds only",
        "formula": "16*sum(k^2,k=1..n-2) distinct parity-product propagation nonzeros, each >=24 bytes; excludes all other blocks/environments",
        "formula_consistent_with_existing_measured_operator": formula_consistent,
        "blocked_structures_at_frozen_product_cap": sum(row["expanded_product_exceeds_frozen_500MB_cap"] for row in rows),
        "observed_device_bytes": device_bytes,
        "blocked_structures_even_at_entire_device_size": sum(row["expanded_product_exceeds_entire_observed_device"] for row in rows),
        "host_metadata_guard_blocked_structures": sum(row["one_body_host_metadata_exceeds_1GB_guard"] for row in rows),
        "blocked_reactions_at_frozen_product_cap": sum(bool(row["blocked_structure_ids"]) for row in reactions),
        "largest_case": max(rows, key=lambda row: row["minimum_expanded_sparse_product_bytes"]),
        "structures": rows, "reactions": reactions,
        "decision": "Expanded ProductMPO is closed as the full-suite inference route. Retain it only as a correctness oracle in its declared small domain. Next representation must contract density factors implicitly without forming this product or losing physical intermediate states.",
        "density_factor_input_producer": "coulomb_factor_source_checks_gpu_20261117.json; largest-case factor generation succeeded, but does not repair expanded operator storage",
        "full_suite_model_score_available": False, "wtmad2_measured": False, "goal_accuracy_achieved": False,
        "numerical_energy_calculations_performed": False, "speed_experiments_performed": False}
    if not formula_consistent: raise RuntimeError("Storage derivation contradicts the existing measured implementation")
    destination.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({key: report[key] for key in ("blocked_structures_at_frozen_product_cap", "blocked_structures_even_at_entire_device_size", "blocked_reactions_at_frozen_product_cap", "largest_case", "decision")}))


if __name__ == "__main__": main()
