"""Fresh-SCF atomization measurements for our screened-state functional."""
from .runtime import configure, torch_setup
ROOT = configure()

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import numpy as np
import torch
from pyscf import dft, gto, lib
from .screened_state import ScreenedStateXC
from .ensemble_screened import EnsembleScreenedXC
from .evaluate_scf import attach
from .prepare_broad import SPINS

KCAL = 627.509474


def solve(model, device, atom, charge, spin, basis, grid_level, max_seconds):
    start = time.perf_counter()
    mol = gto.M(atom=atom, unit="Bohr", basis=basis, charge=charge, spin=spin, verbose=0, max_memory=1800)
    mf = (dft.UKS(mol) if spin else dft.RKS(mol)).density_fit()
    mf.xc = "pbe"
    mf.grids.level = grid_level
    mf.conv_tol = 1e-8
    mf.max_cycle = 80
    mf.diis_start_cycle = 4
    mf.damp = .3
    correction = attach(mf, model, device)
    last_cycle = {}

    def callback(env):
        last_cycle.update({"cycle": int(env["cycle"])+1, "energy": float(env["e_tot"]),
                           "orbital_gradient_norm": float(env.get("norm_gorb", np.nan))})
        if time.perf_counter() - start > max_seconds:
            raise TimeoutError(f"Our calculation exceeded {max_seconds}s bounded attempt")

    mf.callback = callback
    failure = None
    try:
        energy = mf.kernel()
    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"
    if device == "cuda":
        torch.cuda.synchronize()
    result = {"converged": bool(mf.converged) and failure is None,
              "seconds": time.perf_counter()-start, "cycles": last_cycle.get("cycle", getattr(mf, "cycles", None)),
              "correction_calls": correction.calls, "correction_host_timings": correction.timings,
              "timing_scope": "Component host-wall times are synchronous on CPU; GPU work may be asynchronous",
              "ao_count": mol.nao_nr(), "grid_points": len(mf.grids.coords), "last_cycle": last_cycle}
    if failure:
        result["exception"] = failure
    else:
        result["energy_hartree"] = float(energy)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=ROOT / "data/broad_tzvp_192")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=6)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--molecule-seconds", type=float, default=60)
    args = parser.parse_args()
    torch_setup()
    lib.num_threads(1)
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
    model_class = EnsembleScreenedXC if checkpoint.get("architecture") == "ensemble_screened" else ScreenedStateXC
    model = model_class(checkpoint["width"], checkpoint["channels"], checkpoint.get("operator", "diffusive"),
                            quantum=checkpoint.get("quantum", False)).to(args.device).double()
    model.load_compatible(checkpoint["model"])
    model.requires_grad_(False)
    manifest = json.loads((args.data / "manifest.json").read_text())
    selected = sorted([row for row in manifest["molecules"] if row["split"] == args.split], key=lambda row: row["id"])[:args.limit]
    atoms = sorted({symbol for row in selected for symbol in row["symbols"]})
    report = {"created_utc": datetime.now(timezone.utc).isoformat(),
              "status": "Own fresh-SCF atomization screen; not GMTKN55, hidden suite, or matched challenge protocol",
              "device": args.device, "checkpoint": str(args.checkpoint),
              "basis": manifest["basis"], "grid_level": manifest["grid_level"],
              "atoms": {}, "molecules": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def record():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    for symbol in atoms:
        try:
            result = solve(model, args.device, f"{symbol} 0 0 0", 0, SPINS[symbol],
                           manifest["basis"], manifest["grid_level"], args.molecule_seconds)
        except Exception as exc:
            result = {"converged": False, "exception": f"{type(exc).__name__}: {exc}"}
        report["atoms"][symbol] = result
        record()
        print(json.dumps({"atom": symbol, **result}), flush=True)
    for row in selected:
        result = {"id": row["id"], "family": row["family"], "split": row["split"],
                  "pbe_input_error_kcal": (row["tae_pbe"] - row["tae@w1-f12"]) * KCAL}
        try:
            computed = solve(model, args.device, list(zip(row["symbols"], row["geometry"])),
                             int(row["molecular_charge"]), int(row["molecular_multiplicity"]) - 1,
                             manifest["basis"], manifest["grid_level"], args.molecule_seconds)
            result.update(computed)
            if "energy_hartree" in computed and all(report["atoms"][symbol]["converged"] for symbol in row["symbols"]):
                atom_sum = sum(report["atoms"][symbol]["energy_hartree"] for symbol in row["symbols"])
                result["tae_hartree"] = atom_sum - computed["energy_hartree"]
                result["error_kcal"] = (result["tae_hartree"] - row["tae@w1-f12"]) * KCAL
            elif not all(report["atoms"][symbol]["converged"] for symbol in row["symbols"]):
                result["atomic_reference_failure"] = True
        except Exception as exc:
            result.update({"converged": False, "exception": f"{type(exc).__name__}: {exc}"})
        report["molecules"].append(result)
        record()
        print(json.dumps(result), flush=True)
    rows = report["molecules"]
    report["summary"] = {"molecules": len(rows),
                         "molecular_failures": sum(not row["converged"] for row in rows),
                         "atomic_failures": sum(not row["converged"] for row in report["atoms"].values()),
                         "all_returned_tae_mae_kcal": float(np.mean([abs(row["error_kcal"]) for row in rows]))
                             if rows and all("error_kcal" in row for row in rows) else None,
                         "median_molecular_seconds": float(np.median([row["seconds"] for row in rows]))
                             if rows and all("seconds" in row for row in rows) else None}
    record()
    print(json.dumps({"summary": report["summary"]}), flush=True)


if __name__ == "__main__":
    main()
