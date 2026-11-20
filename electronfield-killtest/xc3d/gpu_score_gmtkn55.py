"""Score a frozen candidate's actual total energies on the full public suite."""
from .runtime import configure, torch_setup
ROOT = configure(gpu_only_compatible=True)

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import torch
from .gmtkn55 import GMTKN55Suite


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise RuntimeError("Preserve existing scoring reports; choose a new report path")
    if not json.loads(Path("research/gmtkn55_runner_checks_gpu_20261117.json").read_text())["passed"]:
        raise RuntimeError("The full-suite parser/scorer validation must pass first")
    protocol = json.loads(Path("research/gmtkn55_scoring_protocol_20261117.json").read_text())
    device = torch_setup(17112069)
    suite = GMTKN55Suite(ROOT, protocol)
    energies, accepted, metadata = suite.read_predictions(args.predictions, device)
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "device": device,
              "protocol": protocol, **metadata, **suite.score(energies, accepted, suite.tensors(device)),
              "peak_cuda_bytes": torch.cuda.max_memory_allocated(), "speed_measured": False}
    # A small-basis score can be useful research evidence, but cannot establish
    # the organizer's target-basis acceptance or any private-test result.
    report["target_basis_declared"] = metadata["candidate"]["basis"].lower() == "def2-qzvp"
    report["challenge_accuracy_gate_met"] = False
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({key: report[key] for key in ("complete_public_suite_scored", "wtmad2_kcal_per_mol", "measured_reactions", "total_reactions", "prediction_status_counts", "target_basis_declared", "challenge_accuracy_gate_met")}), flush=True)


if __name__ == "__main__": main()
