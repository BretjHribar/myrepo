"""New tensor density interface versus independent finite-occupation operators."""
from .runtime import configure, torch_setup
ROOT = configure(gpu_only_compatible=True)

from datetime import datetime, timezone
import json
from pathlib import Path
import torch
from .gpu_purified_spin import PurifiedSpinState
from .gpu_spin_response import SpinChain, overlap
from .gpu_tensor_density import density_matrix, symmetric_transition_density
from .gpu_manybody_ops import dense_operator
from .gpu_verify_purified_operator import exhaustive_purification, randomize


def main():
    output = Path("research/tensor_density_checks_gpu_20261117.json")
    if output.exists(): raise RuntimeError("This density-interface test already has a report")
    device = torch_setup(17112046)
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "device": device, "results": [], "failures": [],
              "scope": "Fixed-state spatial density and cross-density interface, no new molecular/reference solve"}
    try:
        for number, spin in ((4, 0), (3, 1)):
            states = [PurifiedSpinState(4, number, spin, channels=2, ancilla=2, device=device) for _ in range(2)]
            for state in states: randomize(state)
            chains = [SpinChain.from_state(state).normalized() for state in states]
            waves = [exhaustive_purification(chain.cores)[2].detach() for chain in chains]
            self_density = density_matrix(chains[0])
            cross_density = symmetric_transition_density(chains[0], chains[1])
            expected_self, expected_cross = torch.zeros_like(self_density), torch.zeros_like(cross_density)
            bits = torch.arange(4**4, device=device)
            with torch.no_grad():
                for first in range(4):
                    for second in range(first, 4):
                        words = torch.tensor([((first, s, 1), (second, s, 0), (-1, 0, 0), (-1, 0, 0)) for s in range(2)], device=device)
                        matrix = dense_operator(words, torch.ones(2, device=device), bits, 4)
                        diagonal = (waves[0]*(matrix @ waves[0])).sum()
                        transition = (waves[0]*(matrix @ waves[1])).sum()
                        expected_self[first, second] = expected_self[second, first] = diagonal
                        expected_cross[first, second] = expected_cross[second, first] = transition
                self_error = float((self_density-expected_self).abs().max())
                cross_error = float((cross_density-expected_cross).abs().max())
                number_error = float((self_density.trace()-number).abs())
                transition_trace_error = float((cross_density.trace()-number*overlap(chains[0], chains[1])).abs())
                occupations = torch.linalg.eigvalsh(self_density)
                valid = (max(self_error, cross_error, number_error, transition_trace_error) < 1e-10
                         and float(occupations.min()) >= -1e-10 and float(occupations.max()) <= 2+1e-10)
                row = {"electrons": number, "twice_spin": spin, "passed": valid,
                       "maximum_self_density_difference": self_error, "maximum_cross_density_difference": cross_error,
                       "number_error": number_error, "cross_number_identity_error": transition_trace_error,
                       "minimum_spatial_occupation": float(occupations.min()), "maximum_spatial_occupation": float(occupations.max())}
                report["results"].append(row); print(json.dumps(row), flush=True)
                output.write_text(json.dumps(report, indent=2)+"\n")
                if not valid: raise RuntimeError("Tensor density failed independent occupation-space checks")
    except Exception as exc:
        report["failures"].append({"error": f"{type(exc).__name__}: {exc}"}); print(json.dumps(report["failures"][-1]), flush=True)
    report["passed"] = len(report["results"]) == 2 and not report["failures"] and all(row["passed"] for row in report["results"])
    report["peak_cuda_bytes"] = torch.cuda.max_memory_allocated()
    output.write_text(json.dumps(report, indent=2)+"\n")
    if not report["passed"]: raise RuntimeError("Tensor density is not approved for functional use")


if __name__ == "__main__": main()
