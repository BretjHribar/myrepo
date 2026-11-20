"""Score a prediction file with the pinned suite/scorer and print a compact per-subset table.

Unlike xc3d.gpu_score_gmtkn55 this never claims a full-suite number while reactions are missing:
it lists measured subsets only and labels partial coverage explicitly.
"""
import argparse
import json
import sys
from pathlib import Path

from .common import REPO, RESEARCH, pin_one_core

pin_one_core()
sys.path.insert(0, str(REPO / "electronfield-killtest"))
import torch
from xc3d.gmtkn55 import GMTKN55Suite


def score_file(pred_path):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    protocol = json.loads((REPO / "electronfield-killtest/research/gmtkn55_scoring_protocol_20261117.json").read_text())
    suite = GMTKN55Suite(RESEARCH, protocol)
    energies, accepted, meta = suite.read_predictions(pred_path, "cuda")
    report = suite.score(energies, accepted, suite.tensors("cuda"))
    report.update(meta)
    return suite, report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    suite, report = score_file(args.predictions)
    done = {k: v for k, v in report["subsets"].items() if v["all_reactions_scored"]}
    print(f"measured reactions {report['measured_reactions']}/{report['total_reactions']}; "
          f"complete subsets {len(done)}/55; full WTMAD-2: {report['wtmad2_kcal_per_mol']}")
    print("status counts", report["prediction_status_counts"])
    for name in sorted(done):
        v = done[name]
        print(f"  {name:12s} n={v['total_reactions']:3d} MAD={v['mad_kcal_per_mol']:8.3f}")
    for group, v in report["groups"].items():
        print(f"  group {group}: {v['measured_reactions']}/{v['total_reactions']} wtmad2={v['wtmad2_kcal_per_mol']}")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
