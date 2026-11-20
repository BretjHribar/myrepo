"""Score our new functional on balanced reaction contrasts with full accounting.

Frozen inputs diagnose the energy representation. The SCF mode times complete
one-core molecular calculations and does not need isolated atomic references.
This small panel is developmental and has explicit fitting-overlap flags.
"""
from .runtime import configure, torch_setup
ROOT = configure()

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import numpy as np
import torch
from pyscf import lib
from .ensemble_screened import EnsembleScreenedXC
from .screened_state import ScreenedStateXC
from .train_broad import Input
from .evaluate_broad_scf import solve

KCAL = 627.509474


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=ROOT / "data/ensemble_reactions_tzvp")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("frozen", "scf"), default="frozen")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--limit", type=int, default=18)
    parser.add_argument("--molecule-seconds", type=float, default=30)
    args = parser.parse_args()
    torch_setup()
    lib.num_threads(1)
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
    model_class = EnsembleScreenedXC if checkpoint.get("architecture") == "ensemble_screened" else ScreenedStateXC
    model = model_class(checkpoint["width"], checkpoint["channels"], checkpoint.get("operator", "diffusive"),
                        quantum=checkpoint.get("quantum", False)).to(args.device)
    model = model.double() if args.mode == "scf" else model.float()
    model.load_compatible(checkpoint["model"])
    model.requires_grad_(False)
    manifest = json.loads((args.data / "manifest.json").read_text())
    reactions = manifest["reactions"][:args.limit]
    needed = {formula for row in reactions for formula in row["stoichiometry"]}
    cases = {row["formula_label"]: row for row in manifest["molecules"] if row["formula_label"] in needed}
    report = {"created_utc": datetime.now(timezone.utc).isoformat(),
              "status": "Development reaction contrasts; not a blind benchmark, GMTKN55, or challenge score",
              "mode": args.mode, "device": args.device, "checkpoint": str(args.checkpoint),
              "basis": manifest["basis"], "grid_level": manifest["grid_level"],
              "molecules": {}, "reactions": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def record():
        args.output.write_text(json.dumps(report, indent=2)+"\n")

    for formula, info in sorted(cases.items()):
        started = time.perf_counter()
        result = {"id": info["id"], "seen_in_fitting": info["seen_in_fitting"]}
        try:
            if args.mode == "scf":
                result.update(solve(model, args.device, list(zip(info["symbols"], info["geometry"])),
                                    int(info["molecular_charge"]), int(info["molecular_multiplicity"])-1,
                                    manifest["basis"], manifest["grid_level"], args.molecule_seconds))
            else:
                case = Input(args.data / f"mol_{info['id']}.npz", args.device, model)
                with torch.no_grad():
                    result["energy_hartree"] = info["energy_pbe"]+float(case.correction(model))
                result["converged"] = None
                if args.device == "cuda":
                    torch.cuda.synchronize()
                result["seconds"] = time.perf_counter()-started
        except Exception as exc:
            result.update({"converged": False, "exception": f"{type(exc).__name__}: {exc}",
                           "seconds": time.perf_counter()-started})
        report["molecules"][formula] = result
        record()
        print(json.dumps({"formula": formula, **result}), flush=True)
    for reaction in reactions:
        stoichiometry = reaction["stoichiometry"]
        result = dict(reaction)
        result["pbe_input_error_kcal"] = (sum(coefficient*cases[formula]["energy_pbe"]
                                                   for formula, coefficient in stoichiometry.items())
                                            - reaction["reference_hartree"])*KCAL
        present = all("energy_hartree" in report["molecules"][formula] for formula in stoichiometry)
        result["all_energies_returned"] = present
        result["all_scf_converged"] = (all(report["molecules"][formula]["converged"]
                                           for formula in stoichiometry) if args.mode == "scf" else None)
        if present:
            result["predicted_hartree"] = sum(coefficient*report["molecules"][formula]["energy_hartree"]
                                               for formula, coefficient in stoichiometry.items())
            result["error_kcal"] = (result["predicted_hartree"]-reaction["reference_hartree"])*KCAL
        report["reactions"].append(result)
    rows = report["reactions"]
    report["summary"] = {"reactions": len(rows), "molecules": len(cases),
                         "reaction_failures": sum(not row["all_energies_returned"] or row["all_scf_converged"] is False for row in rows),
                         "all_returned_reaction_mae_kcal": float(np.mean([abs(row["error_kcal"]) for row in rows]))
                             if rows and all("error_kcal" in row for row in rows) else None,
                         "pbe_input_reaction_mae_kcal": float(np.mean([abs(row["pbe_input_error_kcal"]) for row in rows])) if rows else None,
                         "median_molecular_seconds": float(np.median([row["seconds"] for row in report["molecules"].values()])),
                         "timing_note": "Frozen mode is correction evaluation only; SCF mode includes full molecular calculation"}
    record()
    print(json.dumps({"summary": report["summary"]}), flush=True)


if __name__ == "__main__":
    main()
