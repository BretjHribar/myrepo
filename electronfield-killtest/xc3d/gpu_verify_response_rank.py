"""Exact-rank source checks for the adaptive response compiler."""
from .runtime import configure, torch_setup
ROOT = configure(gpu_only_compatible=True)

from datetime import datetime, timezone
import argparse
import json
from pathlib import Path
import torch
from .ensemble_spatial_model import leading_completion


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="research/response_rank_before_repair_gpu_20261117.json")
    args = parser.parse_args()
    device = torch_setup()
    space, _ = torch.linalg.qr(torch.randn(240, 20, device=device), mode="reduced")
    basis = space[:, :5]
    weights = torch.randn(1, 100, device=device)
    exact_rank_one = space[:, 5:6] @ weights
    updated, added, capacity = leading_completion(basis, exact_rank_one, 8, 192)
    degenerate = space[:, 6:9] @ torch.eye(3, device=device)
    second, degenerate_added, _ = leading_completion(basis, degenerate, 1, 192)
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "device": device,
              "exact_rank_one_added": added, "expected_rank_one_added": 1,
              "complete_degenerate_triplet_added": degenerate_added, "expected_degenerate_added": 3,
              "unexpected_capacity": capacity,
              "passed": added == 1 and degenerate_added == 3 and not capacity,
              "new_ground_state_or_reference": False}
    Path(args.output).write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
