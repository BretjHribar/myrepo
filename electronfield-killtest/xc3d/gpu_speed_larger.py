"""Matched GPU acceleration checks alongside new full cc-pVDZ cases."""
from .runtime import configure, torch_setup, require_speed_authorization
ROOT = configure(gpu_only_compatible=True)

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time
import torch
from .gpu_four_particle import FourParticleOperator, cached_ritz
from .gpu_chemistry_transfer import cuda_operators

KCAL = 627.509474


@torch.no_grad()
def gpu_ms(function, warmup=5, repeat=50):
    for _ in range(warmup):
        result = function()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeat):
        result = function()
    end.record(); end.synchronize()
    return start.elapsed_time(end)/repeat


@torch.no_grad()
def cached_case(name, device, tolerance):
    data = torch.load(ROOT / "data/coulomb_moment_functional" / f"{name}.pt", map_location=device, weights_only=True)
    checkpoint = torch.load(Path("research") / f"{name}_dual_fixedpoint.pt", map_location=device, weights_only=True)
    electronic = data["universal"]+torch.einsum("a,aij->ij", data["external_coefficients"], data["q_matrices"])
    dense = electronic+torch.eye(len(electronic), device=device)*(data["external_constant"]*data["case"]["electrons"]+data["nuclear"])
    operator = FourParticleOperator(data["h"], data["eri"], data["case"]["electrons"], data["nuclear"])
    state = checkpoint["state"]
    tensor = operator.from_fock(state, data["bits"])
    reconstructed = operator.to_fock(tensor, data["bits"])
    image = operator.to_fock(operator(tensor), data["bits"])
    discrepancy = float((image-dense @ state).abs().max())
    roundtrip = float((reconstructed-state).abs().max())
    naive_discrepancy = float((operator(tensor)-operator.naive(tensor)).abs().max())
    if discrepancy > tolerance or roundtrip > tolerance or naive_discrepancy > tolerance:
        raise RuntimeError(f"New operator is not equivalent: action={discrepancy}, map={roundtrip}, naive={naive_discrepancy}")
    dense_ms = gpu_ms(lambda: dense @ state)
    naive_ms = gpu_ms(lambda: operator.naive(tensor))
    fast_ms = gpu_ms(lambda: operator(tensor))
    return {"id": name, "electrons": data["case"]["electrons"], "spatial_orbitals": len(data["h"]),
            "particle_hole_conjugated": operator.holes, "action_max_error_hartree": discrepancy,
            "mapping_roundtrip_max_error": roundtrip, "antisymmetry_contraction_error_hartree": naive_discrepancy,
            "dense_matvec_ms": dense_ms, "naive_matrix_free_ms": naive_ms, "antisymmetric_matrix_free_ms": fast_ms,
            "contraction_speedup": naive_ms/fast_ms, "dense_to_matrix_free_kernel_speedup": dense_ms/fast_ms,
            "dense_hamiltonian_bytes": dense.numel()*dense.element_size(),
            "new_hamiltonian_bytes": (operator.h.numel()+operator.pair.numel()+1)*operator.h.element_size(),
            "reference_recomputed": False, "timing_scope": "GPU operator application only; not full calculation timing"}


@torch.no_grad()
def larger_case(case, protocol, device):
    started = time.perf_counter()
    folder = ROOT / "data/four_particle_ccpvdz"
    folder.mkdir(parents=True, exist_ok=True)
    destination = folder / f"{case['id']}.pt"
    if destination.exists():
        data = torch.load(destination, map_location=device, weights_only=True)
        if data["basis"] != protocol["larger_basis"] or data["case"] != case:
            raise RuntimeError("Operator cache does not match the frozen larger-case protocol")
    else:
        data = cuda_operators(case, protocol["larger_basis"])
        torch.save({key: value.cpu() if isinstance(value, torch.Tensor) else value for key, value in data.items()}, destination)
    preparation_seconds = time.perf_counter()-started
    operator = FourParticleOperator(data["h"], data["eri"], case["electrons"], data["nuclear"])
    seed = operator.seed()
    density = 2*torch.eye(len(data["h"]), device=device); density[2:, 2:] = 0.
    coulomb = torch.einsum("ijkl,kl->ij", data["eri"], density)
    exchange = torch.einsum("ikjl,kl->ij", data["eri"], density)
    wick = (density*(data["h"]+.5*coulomb-.25*exchange)).sum()+data["nuclear"]
    seed_error = float((seed @ operator(seed)-wick).abs())
    naive_error = float((operator(seed)-operator.naive(seed)).abs().max())
    if max(seed_error, naive_error) > protocol["equivalence_tolerance_hartree"]:
        raise RuntimeError(f"Larger-basis kernel failed its independent Wick/action identity: {seed_error}, {naive_error}")
    naive_ms = gpu_ms(lambda: operator.naive(seed))
    fast_ms = gpu_ms(lambda: operator(seed))
    solve_started = time.perf_counter()
    result = cached_ritz(operator, seed, maximum=protocol["maximum_ritz_dimension"], tolerance=protocol["state_residual_tolerance"])
    torch.cuda.synchronize()
    solve_seconds = time.perf_counter()-solve_started
    independent = result["state"] @ operator.naive(result["state"])
    dimension = math.comb(operator.sites, 2)**2
    row = {"id": case["id"], "electrons": case["electrons"], "basis": data["basis"],
           "spatial_orbitals": operator.sites, "complete_stated_basis": True, "sector_dimension": dimension,
           "state_tensor_entries": seed.numel(), "converged": result["converged"], "state_residual": result["residual"],
           "total_energy_hartree": float(result["energy"]), "lowering_from_input_kcal": float((result["energy"]-wick)*KCAL),
           "input_reference_status": data["reference_status"], "independent_wick_error_hartree": seed_error,
           "independent_final_action_energy_error_hartree": float((independent-result["energy"]).abs()),
           "naive_matrix_free_ms": naive_ms, "antisymmetric_matrix_free_ms": fast_ms,
           "contraction_speedup": naive_ms/fast_ms,
           "dense_hamiltonian_bytes_avoided_estimate": dimension**2*8,
           "operator_bytes": (operator.h.numel()+operator.pair.numel()+1)*8,
           "basis_image_bytes": result["basis_image_bytes"], "operator_applications": result["operator_applications"],
           "ritz_dimension": result["dimension"], "preparation_seconds": preparation_seconds,
           "solve_seconds": solve_seconds, "total_seconds": time.perf_counter()-started,
           "reference_role": "New full-basis physical reference for original model research; not model transfer error",
           "history": result["history"]}
    torch.save({"state_tensor": result["state"].cpu(), "report": row, "case": case, "basis": data["basis"]},
               folder / f"{case['id']}.reference.pt")
    return row


def main():
    require_speed_authorization()
    device = torch_setup()
    protocol_path = Path("research/four_particle_speed_larger_protocol_20261116.json")
    protocol = json.loads(protocol_path.read_text())
    output = Path("research/speed_larger_gpu_20261116.json")
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "device": device,
              "protocol": protocol, "protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
              "cached_equivalence_timings": [], "larger_results": [], "failures": []}
    for name in protocol["cached_equivalence_and_timing_cases"]:
        try:
            row = cached_case(name, device, protocol["equivalence_tolerance_hartree"])
            report["cached_equivalence_timings"].append(row); print(json.dumps(row), flush=True)
        except Exception as exc:
            report["failures"].append({"id": name, "phase": "cached_equivalence", "error": f"{type(exc).__name__}: {exc}"})
            output.write_text(json.dumps(report, indent=2)+"\n")
            raise
    for case in protocol["larger_cases"]:
        try:
            row = larger_case(case, protocol, device)
            report["larger_results"].append(row); print(json.dumps({key: value for key, value in row.items() if key != "history"}), flush=True)
        except Exception as exc:
            report["failures"].append({"id": case["id"], "phase": "larger_case", "error": f"{type(exc).__name__}: {exc}"})
            print(json.dumps(report["failures"][-1]), flush=True)
        output.write_text(json.dumps(report, indent=2)+"\n")
    report["peak_cuda_bytes"] = torch.cuda.max_memory_allocated()
    output.write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    main()
