"""Fit one model on every available reaction, write its per-structure total energies in the pinned
prediction contract, and score them with the existing suite scorer.

This is the IN-SAMPLE score of a fitted functional (the parameters were fitted to the same benchmark).
Cross-validated numbers come from xcgpu.fit; they cannot be written as one energy per structure
because each fold has its own parameters. The scorer reports a full WTMAD-2 only at 1505/1505.
"""
import argparse
import hashlib
import json
from pathlib import Path

from .common import BULK, PRED_INPUTS, pin_one_core

pin_one_core()
import torch
from . import fitlib as fl
from .collect import source_hash
from .fit import LADDER
from .quick_score import score_file


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scf-tag", default="pbe_qzvp_g3")
    ap.add_argument("--model", default="N:")
    ap.add_argument("--lam", type=float, default=1e-6)
    ap.add_argument("--name", required=True, help="candidate id and output file stem")
    args = ap.parse_args()
    rows, problems = fl.structure_table(args.scf_tag)
    label, base, blocks = next(m for m in LADDER if m[0].startswith(args.model))
    keys = [k for b in blocks for k in fl.FEATURE_SETS[b]]
    prob = fl.Problem(rows, keys, base)
    theta = fl.solve(prob.A, prob.y, prob.w, args.lam)[0] if keys else torch.zeros(0, dtype=torch.float64, device=fl.DEV)
    fit_value = prob.wtmad2(prob.A @ theta - prob.y) if keys else prob.wtmad2(-prob.y)
    inputs = json.loads(PRED_INPUTS.read_text())
    card = {"candidate": args.name, "model": label, "base": base, "feature_blocks": blocks, "n_parameters": len(keys),
            "lambda": args.lam, "fitted_on": f"{len(prob.y)} of {prob.n_total} GMTKN55 reactions (in-sample score)",
            "inputs": f"PBE/def2-QZVP orbitals, run {args.scf_tag}, passes corr_v1 and semi_v1; evaluated after the SCF",
            "parameters": dict(zip(map(str, keys), theta.tolist()))}
    card_bytes = json.dumps(card, sort_keys=True).encode()
    preds = []
    for s in inputs["structures"]:
        sid = s["id"]
        if sid in problems:
            preds.append({"id": sid, "status": "failed", "reason": "pipeline problem: " + ",".join(problems[sid])})
            continue
        if sid not in rows:
            continue
        e0, f = fl.structure_energy_terms(rows, [sid], keys, base)
        e = float(e0[0] + (f[0] @ theta if keys else 0.0))
        if e != e:
            preds.append({"id": sid, "status": "failed", "reason": "an ingredient is missing for this structure"})
            continue
        preds.append({"id": sid, "status": "accepted", "converged": True, "physical_checks_passed": True,
                      "total_energy_hartree": e})
    doc = {"schema": "gmtkn55_total_energies_v1", "benchmark_commit": inputs["benchmark_commit"],
           "manifest_sha256": inputs["manifest_sha256"], "units": "hartree",
           "energy_kind": "total_electronic_plus_nuclear_hartree",
           "candidate": {"id": args.name, "basis": "def2-qzvp", "protocol_sha256": hashlib.sha256(card_bytes).hexdigest(),
                         "source_sha256": source_hash(), "run_card": card},
           "benchmark_reference_used_for_prediction": False,
           "parameters_fitted_to_this_benchmark": bool(keys), "predictions": preds}
    out = BULK / "gmtkn55" / args.scf_tag / "models"
    out.mkdir(exist_ok=True)
    pred_path = out / (args.name + ".predictions.json")
    pred_path.write_text(json.dumps(doc))
    suite, report = score_file(pred_path)
    report["run_card"] = card
    (out / (args.name + ".score.json")).write_text(json.dumps(report, indent=1))
    n = report["measured_reactions"]
    part = report["measured_nonnegative_contribution_lower_bound_kcal_per_mol"]
    print(f"{args.name}: {label}")
    print(f"  measured reactions {n}/{report['total_reactions']}; full WTMAD-2: {report['wtmad2_kcal_per_mol']}")
    if n:
        print(f"  mean weighted |error| over the measured reactions: {part * report['total_reactions'] / n:.4f} "
              f"(fit library value on its {len(prob.y)} reactions: {fit_value:.4f})")
    for g, v in report["groups"].items():
        print(f"  {g}: {v['measured_reactions']}/{v['total_reactions']}  wtmad2 {v['wtmad2_kcal_per_mol']}")


if __name__ == "__main__":
    main()
