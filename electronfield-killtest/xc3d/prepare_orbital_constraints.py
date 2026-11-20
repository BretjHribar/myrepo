"""Cache exact one-orbital ensemble identities with consistent AO descriptors.

No new SCF calculations. Trial orbitals and three normalized tangent directions
are constructed in the cached basis, using fitting compositions and atoms only.
Their Hartree energies and slopes use the same analytic RI metric as the inputs.
"""
from .runtime import configure
ROOT = configure()

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from pyscf import df, dft, gto, lib


def generate(path, destination, orbitals):
    started = time.perf_counter()
    with np.load(path) as data:
        info = json.loads(str(data["metadata"]))
        coords, weights = data["coords"].astype(float), data["weights"]
        landmarks = data["landmarks"]
        dm, root = data["dm"], data["overlap_root"]
        inverse, overlap = data["overlap_inverse_root"], data["overlap"]
        slices = data["atom_slices"]
        density = data["rho"][:, 0].sum(0).astype(float)
    atom = (f"{info['symbols'][0]} 0 0 0" if info["id"].startswith("atom_")
            else list(zip(info["symbols"], info["geometry"])))
    spin = int(info["spin"]) if info["id"].startswith("atom_") else int(info["molecular_multiplicity"])-1
    mol = gto.M(atom=atom, unit="Bohr", basis=info["basis"], spin=spin,
                charge=int(info.get("molecular_charge", 0)), verbose=0, max_memory=1800)
    ao = dft.numint.eval_ao(mol, coords, deriv=1)
    auxiliary = df.addons.make_auxmol(mol)
    metric = auxiliary.intor("int2c2e")
    three_center = df.incore.aux_e2(mol, auxiliary, intor="int3c2e", aosym="s1")
    projected = root @ dm.sum(0) @ root
    rng = np.random.default_rng(int(hashlib.sha256(info["id"].encode()).hexdigest()[:8], 16))
    entries = []
    for index in range(orbitals):
        output = destination / f"{path.stem}_orbital{index}.npz"
        if output.exists():
            with np.load(output) as data:
                entries.append(json.loads(str(data["metadata"])))
            continue
        if index == 0:
            rhs = ao[0].T @ (weights * np.sqrt(density.clip(min=0)))
            coefficient = np.linalg.solve(overlap, rhs)
        else:
            coefficient = inverse @ (projected @ rng.normal(size=len(overlap)))
        coefficient /= np.sqrt(coefficient @ overlap @ coefficient)
        phi = ao @ coefficient
        directions = []
        probability_directions = []
        hartree_slopes = []
        fitted = np.einsum("ijp,i,j->p", three_center, coefficient, coefficient, optimize=True)
        fitted_potential = np.linalg.solve(metric, fitted)
        psi = root @ coefficient
        probability = np.array([np.sum(psi[a:b]**2) for a, b in slices])
        for _ in range(3):
            direction = inverse @ rng.normal(size=len(overlap))
            direction -= coefficient * (coefficient @ overlap @ direction)
            direction /= np.sqrt(direction @ overlap @ direction)
            directions.append(ao @ direction)
            tangent = root @ direction
            probability_directions.append([2*np.dot(psi[a:b], tangent[a:b]) for a, b in slices])
            variation = 2*np.einsum("ijp,i,j->p", three_center, coefficient, direction, optimize=True)
            hartree_slopes.append(float(fitted_potential @ variation))
        metadata = {"id": f"{info['id']}_orbital{index}", "source": str(path),
                    "hartree_normalized": float(fitted @ fitted_potential / 2),
                    "hartree_slopes": hartree_slopes, "fitting_shape": True,
                    "orbital_source": "AO projection of sqrt(input density)" if index == 0 else "occupied-subspace combination",
                    "hartree_approximation": "analytic RI in matching density-fit auxiliary basis"}
        np.savez_compressed(output, metadata=json.dumps(metadata), phi=phi.astype(np.float32),
                            directions=np.asarray(directions, dtype=np.float32),
                            probability=probability.astype(np.float32),
                            probability_directions=np.asarray(probability_directions, dtype=np.float32),
                            coords=coords.astype(np.float32), weights=weights, landmarks=landmarks)
        entries.append(metadata)
    print(json.dumps({"source": info["id"], "orbitals": len(entries),
                      "seconds": round(time.perf_counter()-started, 3)}), flush=True)
    return entries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/quantum_tzvp_191")
    parser.add_argument("--output", type=Path, default=ROOT / "data/orbital_one_electron")
    parser.add_argument("--molecules", type=int, default=16)
    parser.add_argument("--orbitals", type=int, default=2)
    args = parser.parse_args()
    lib.num_threads(1)
    manifest = json.loads((args.data / "manifest.json").read_text())
    selected = sorted([row for row in manifest["molecules"] if row["split"] == "train"],
                      key=lambda row: row["id"])[:args.molecules]
    paths = sorted(args.data.glob("atom_*.npz")) + [args.data / f"mol_{row['id']}.npz" for row in selected]
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "source": str(args.data),
              "identity": "E_xc=-E_H for N<=1 charge/spin ensembles in a normalized common orbital",
              "fitting_shapes_only": True, "entries": []}
    for path in paths:
        report["entries"].extend(generate(path, args.output, args.orbitals))
        (args.output / "manifest.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({"constraint_orbitals": len(report["entries"])}), flush=True)


if __name__ == "__main__":
    main()
