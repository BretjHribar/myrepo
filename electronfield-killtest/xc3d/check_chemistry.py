"""One new-environment chemistry check, not a benchmark reproduction."""
from .runtime import configure, torch_setup
ROOT = configure()
DEVICE = torch_setup()

import json
import os
from pathlib import Path
import platform
import time
import numpy as np
import torch
from pyscf import gto, dft, lib
import pyscf


def main():
    lib.num_threads(1)
    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="cc-pvdz", verbose=0)
    report = {"python": platform.python_version(), "pyscf": pyscf.__version__,
              "torch": torch.__version__, "device": DEVICE,
              "cuda_runtime": torch.version.cuda,
              "cpu_affinity": sorted(os.sched_getaffinity(0)),
              "torch_threads": torch.get_num_threads(),
              "basis": "cc-pvdz", "molecule": "H2, 0.74 Angstrom", "calculations": {}}
    for xc in ("pbe", "r2scan"):
        mf = dft.RKS(mol)
        mf.xc = xc
        mf.grids.level = 1
        mf.conv_tol = 1e-10
        start = time.perf_counter()
        energy = mf.kernel()
        report["calculations"][f"cpu_{xc}"] = {
            "energy_hartree": float(energy), "converged": bool(mf.converged),
            "seconds": time.perf_counter() - start,
        }
    try:
        import cupy
        from gpu4pyscf.dft import rks
        mf = rks.RKS(mol)
        mf.xc = "pbe"
        mf.grids.level = 1
        mf.conv_tol = 1e-10
        cupy.cuda.Stream.null.synchronize()
        start = time.perf_counter()
        energy = mf.kernel()
        cupy.cuda.Stream.null.synchronize()
        report["calculations"]["gpu_pbe"] = {
            "energy_hartree": float(energy), "converged": bool(mf.converged),
            "seconds_including_first_call": time.perf_counter() - start,
            "cpu_energy_difference_hartree": float(energy) - report["calculations"]["cpu_pbe"]["energy_hartree"],
        }
    except Exception as exc:
        report["gpu4pyscf_error"] = f"{type(exc).__name__}: {exc}"
    output = Path("research/chemistry_environment_2026-11-16.json")
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
