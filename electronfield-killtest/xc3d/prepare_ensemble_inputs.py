"""Add analytic regional Coulomb and baseline-XC fields to existing inputs.

No SCF run or new energy labels. Dense integral tensors are temporary, and only
the small regional Coulomb matrix is cached for frozen-state parameter fitting.
"""
from .runtime import configure
ROOT = configure()

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import numpy as np
from pyscf import dft, gto, lib
from .ensemble_screened import whitened_integrals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/quantum_tzvp_191")
    parser.add_argument("--output", type=Path, default=ROOT / "data/ensemble_tzvp_191")
    args = parser.parse_args()
    lib.num_threads(1)
    manifest = json.loads((args.data / "manifest.json").read_text())
    report = {**manifest, "created_utc": datetime.now(timezone.utc).isoformat(),
              "ensemble_fields": "Regional analytic RI Coulomb metric and PBE grid energy density"}
    args.output.mkdir(parents=True, exist_ok=True)
    paths = sorted(args.data.glob("atom_*.npz")) + [args.data / f"mol_{row['id']}.npz" for row in manifest["molecules"]]
    for path in paths:
        started = time.perf_counter()
        output = args.output / path.name
        if output.exists():
            continue
        with np.load(path) as data:
            arrays = {key: data[key] for key in data.files}
        info = json.loads(str(arrays["metadata"]))
        is_atom = info["id"].startswith("atom_")
        atom = f"{info['symbols'][0]} 0 0 0" if is_atom else list(zip(info["symbols"], info["geometry"]))
        spin = int(info["spin"]) if is_atom else int(info["molecular_multiplicity"])-1
        mol = gto.M(atom=atom, unit="Bohr", basis=info["basis"], spin=spin,
                    charge=int(info.get("molecular_charge", 0)), verbose=0, max_memory=1800)
        integrals = whitened_integrals(mol, arrays["overlap_inverse_root"])
        projected = arrays["overlap_root"] @ arrays["dm"].sum(0) @ arrays["overlap_root"]
        fitted = np.array([np.einsum("ij,ijp->p", projected[a:b], integrals[a:b], optimize=True)
                           for a, b in arrays["atom_slices"]])
        arrays["hartree_matrix"] = fitted @ fitted.T
        rho = arrays["rho"].astype(float)
        eps = dft.numint.NumInt().eval_xc_eff("pbe", rho, deriv=0, spin=1)[0]
        arrays["pbe_density"] = eps*rho[:, 0].sum(0)
        np.savez_compressed(output, **arrays)
        print(json.dumps({"id": info["id"], "seconds": round(time.perf_counter()-started, 3)}), flush=True)
    (args.output / "manifest.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({"molecules": len(manifest["molecules"]), "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
