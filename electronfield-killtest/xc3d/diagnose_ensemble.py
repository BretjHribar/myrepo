"""One new orbital-direction derivative diagnostic for analytic cancellation."""
from .runtime import configure, torch_setup
ROOT = configure()

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import numpy as np
from scipy.linalg import expm
import torch
from pyscf import dft, gto, lib
from .ensemble_screened import EnsembleScreenedXC
from .evaluate_scf import ScreenedCorrection
from .quantum_features import overlap_roots


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("research/ensemble_new_derivative_20261116.json"))
    args = parser.parse_args()
    device = torch_setup()
    lib.num_threads(1)
    started = time.perf_counter()
    with np.load(ROOT / "data/ensemble_tzvp_191/atom_C.npz") as data:
        dm = data["dm"]
        coords, weights = data["coords"].astype(float), data["weights"]
    mol = gto.M(atom="C 0 0 0", basis="def2-tzvp", spin=2, verbose=0)
    mf = dft.UKS(mol).density_fit()
    mf.grids.coords, mf.grids.weights = coords, weights
    model = EnsembleScreenedXC().to(device).double().requires_grad_(False)
    correction = ScreenedCorrection(model, mf, device)
    value, gradient = correction.value(dm)
    root, inverse = overlap_roots(mf.get_ovlp())
    projected = root @ dm @ root
    rng = np.random.default_rng(117)
    generator = rng.normal(size=dm.shape[-2:])
    generator = (generator-generator.T)/np.sqrt(dm.shape[-1])
    tangent = inverse @ (generator @ projected-projected @ generator) @ inverse
    analytic = float(np.sum(gradient*tangent))
    step = 1e-5
    plus, minus = expm(step*generator), expm(-step*generator)
    eplus = correction.value(inverse @ (plus @ projected @ plus.T) @ inverse)[0]
    eminus = correction.value(inverse @ (minus @ projected @ minus.T) @ inverse)[0]
    numeric = (eplus-eminus)/(2*step)
    error = abs(analytic-numeric)
    report = {"created_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "New analytic ensemble functional's full DM derivative; no unchanged benchmark rerun",
              "case": "spin-polarized carbon, occupied/virtual orbital rotation", "device": device,
              "correction_hartree": value, "autograd_directional": analytic,
              "finite_difference_directional": numeric, "absolute_error": error,
              "passed": error < 1e-5+1e-4*abs(numeric), "seconds": time.perf_counter()-started}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report), flush=True)
    if not report["passed"]:
        raise RuntimeError("Changed energy derivative disagrees with its orbital variation")


if __name__ == "__main__":
    main()
