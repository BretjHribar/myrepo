"""Mixed search plus final FP64 physical refinement on saved larger cases."""
from .runtime import configure, torch_setup, require_speed_authorization
ROOT = configure(gpu_only_compatible=True)

from datetime import datetime, timezone
import json
from pathlib import Path
import time
import torch
from .gpu_four_particle import FourParticleOperator, GraphOperator, incremental_ritz
from .gpu_speed_larger import gpu_ms

KCAL = 627.509474


@torch.no_grad()
def main():
    require_speed_authorization()
    device = torch_setup()
    protocol = json.loads(Path("research/mixed_speed_protocol_20261116.json").read_text())
    baseline = json.loads(Path("research/speed_larger_gpu_20261116.json").read_text())
    prior = {row["id"]: row for row in baseline["larger_results"]}
    graph_cache = {}
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "device": device,
              "protocol": protocol, "results": [], "failures": [], "tf32_enabled": torch.backends.cuda.matmul.allow_tf32}
    output = Path("research/mixed_speed_larger_gpu_20261116.json")
    folder = ROOT / "data/four_particle_ccpvdz"
    for name in protocol["cases"]:
        try:
            started = time.perf_counter()
            data = torch.load(folder / f"{name}.pt", map_location=device, weights_only=True)
            reference = torch.load(folder / f"{name}.reference.pt", map_location=device, weights_only=True)
            operator = FourParticleOperator(data["h"], data["eri"], data["case"]["electrons"], data["nuclear"])
            seed = operator.seed()
            shift = seed @ operator(seed)
            setup_started = time.perf_counter()
            backends = {}
            fresh_graphs = 0
            for dtype in (torch.float32, torch.float64):
                cast = operator.cast(dtype, shift)
                key = operator.sites, dtype
                if key in graph_cache:
                    graph_cache[key].replace(cast)
                else:
                    graph_cache[key] = GraphOperator(cast, seed.to(dtype))
                    fresh_graphs += 1
                backends[dtype] = graph_cache[key]
            torch.cuda.synchronize()
            setup_seconds = time.perf_counter()-setup_started
            graph32_ms = gpu_ms(lambda: backends[torch.float32](seed.float()))
            graph64_ms = gpu_ms(lambda: backends[torch.float64](seed))
            solve_started = time.perf_counter()
            search = incremental_ritz(backends[torch.float32], seed.float(),
                                      maximum=protocol["search_maximum_dimension"],
                                      tolerance=protocol["search_residual_tolerance"])
            torch.cuda.synchronize()
            search_seconds = time.perf_counter()-solve_started
            polish_started = time.perf_counter()
            final = incremental_ritz(backends[torch.float64], search["state"].double(),
                                     maximum=protocol["final_maximum_dimension"],
                                     tolerance=protocol["final_residual_tolerance"])
            torch.cuda.synchronize()
            polish_seconds = time.perf_counter()-polish_started
            # Certify with the original FP64 physical operator, outside graphs.
            state = final["state"]/final["state"].norm()
            image = operator(state)
            energy = state @ image
            residual = float((image-energy*state).norm())
            energy_difference = float((energy-reference["report"]["total_energy_hartree"])*KCAL)
            overlap = float((state @ reference["state_tensor"]).abs().square())
            solve_seconds = search_seconds+polish_seconds
            valid = residual < protocol["final_residual_tolerance"] and abs(energy_difference) < protocol["energy_agreement_tolerance_kcal"]
            row = {"id": name, "spatial_orbitals": operator.sites, "prediction_valid": valid,
                   "energy_hartree": float(energy), "energy_difference_kcal": energy_difference,
                   "final_independent_fp64_residual": residual, "squared_overlap_with_saved_fp64_state": overlap,
                   "fp32_search_converged": search["converged"], "search_dimension": search["dimension"],
                   "polish_dimension": final["dimension"], "search_seconds": search_seconds,
                   "polish_seconds": polish_seconds, "warm_solve_seconds": solve_seconds,
                   "graph_setup_seconds": setup_seconds, "graphs_created": fresh_graphs,
                   "setup_plus_solve_seconds": setup_seconds+solve_seconds,
                   "saved_fp64_solve_seconds": prior[name]["solve_seconds"],
                   "warm_solve_speedup": prior[name]["solve_seconds"]/solve_seconds,
                   "setup_included_speedup": prior[name]["solve_seconds"]/(setup_seconds+solve_seconds),
                   "graph_fp32_operator_ms": graph32_ms, "graph_fp64_operator_ms": graph64_ms,
                   "prior_eager_fp64_operator_ms": prior[name]["antisymmetric_matrix_free_ms"],
                   "operator_fp32_graph_speedup": prior[name]["antisymmetric_matrix_free_ms"]/graph32_ms,
                   "operator_fp64_graph_speedup": prior[name]["antisymmetric_matrix_free_ms"]/graph64_ms,
                   "reference_recomputed": False, "total_seconds_including_io_and_timing": time.perf_counter()-started}
            report["results"].append(row)
            torch.save({"state_tensor": state.cpu(), "report": row}, folder / f"{name}.mixed.pt")
            print(json.dumps(row), flush=True)
        except Exception as exc:
            row = {"id": name, "error": f"{type(exc).__name__}: {exc}"}
            report["failures"].append(row); print(json.dumps(row), flush=True)
        output.write_text(json.dumps(report, indent=2)+"\n")
    report["peak_cuda_bytes"] = torch.cuda.max_memory_allocated()
    output.write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    main()
