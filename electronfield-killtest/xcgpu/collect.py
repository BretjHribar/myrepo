"""Turn per-structure run results into the pinned prediction contract (file I/O only)."""
import argparse
import hashlib
import json
from pathlib import Path

from .common import BULK, PRED_INPUTS, pin_one_core

S2_TOLERANCE = 1.0  # a larger deviation means the wrong spin state was found


def source_hash():
    h = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        h.update(path.name.encode() + b"\0" + path.read_bytes())
    return h.hexdigest()


def build_predictions(tag, energy_key="e_tot", candidate_id=None, energies=None):
    """energies: optional {structure id: hartree} overriding the stored SCF totals (post-SCF models)."""
    inputs = json.loads(PRED_INPUTS.read_text())
    run = BULK / "gmtkn55" / tag
    card_bytes = (run / "run_card.json").read_bytes()
    rows = []
    for s in inputs["structures"]:
        path = run / "results" / (s["id"].replace("/", "__") + ".json")
        if not path.exists():
            continue  # stays "uncomputed" in the scorer
        r = json.loads(path.read_text())
        if r.get("error"):
            rows.append({"id": s["id"], "status": "failed", "reason": "exception: " + r["error"][:200]})
            continue
        if not r.get("converged") or not r.get("finite"):
            rows.append({"id": s["id"], "status": "failed", "reason": "SCF not converged after all retry stages"})
            continue
        if r.get("unpaired") and abs(r["s_squared"] - r["s_squared_exact"]) > S2_TOLERANCE:
            rows.append({"id": s["id"], "status": "failed", "reason": f"spin contamination {r['s_squared']:.3f} vs {r['s_squared_exact']:.3f}"})
            continue
        value = r[energy_key] if energies is None else energies.get(s["id"])
        if value is None:
            rows.append({"id": s["id"], "status": "failed", "reason": "post-SCF model energy unavailable"})
            continue
        rows.append({"id": s["id"], "status": "accepted", "converged": True, "physical_checks_passed": True,
                     "total_energy_hartree": float(value)})
    card = json.loads(card_bytes)
    return {"schema": "gmtkn55_total_energies_v1", "benchmark_commit": inputs["benchmark_commit"],
            "manifest_sha256": inputs["manifest_sha256"], "units": "hartree",
            "energy_kind": "total_electronic_plus_nuclear_hartree",
            "candidate": {"id": candidate_id or f"{card['xc']}_{tag}", "basis": card["basis"],
                          "protocol_sha256": hashlib.sha256(card_bytes).hexdigest(), "source_sha256": source_hash(),
                          "run_card": card},
            "benchmark_reference_used_for_prediction": False, "predictions": rows}


def main():
    pin_one_core()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    doc = build_predictions(args.tag)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(doc))
    counts = {}
    for row in doc["predictions"]:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    print(json.dumps({"written": str(args.output), "counts": counts}))


if __name__ == "__main__":
    main()
