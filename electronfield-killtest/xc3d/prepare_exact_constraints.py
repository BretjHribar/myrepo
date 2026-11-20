"""Coulomb fields for exact N<=1 ensemble identities, without new SCF runs.

Normalized positive densities from fitting inputs define valid one-orbital
densities. Their kinetic descriptor is replaced by tau_W during constraint
training. Coulomb fitting supplies the Hartree energy and potential labels.
Only fitting compositions and isolated atoms supply shapes, not held-out inputs.
"""
from .runtime import configure
ROOT = configure()

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import numpy as np
from pyscf import df, gto, lib


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/broad_tzvp_192")
    parser.add_argument("--molecules", type=int, default=32)
    parser.add_argument("--output", type=Path, default=ROOT / "data/exact_one_electron")
    args = parser.parse_args()
    lib.num_threads(1)
    manifest = json.loads((args.data / "manifest.json").read_text())
    selected = sorted([row for row in manifest["molecules"] if row["split"] == "train"], key=lambda row: row["id"])[:args.molecules]
    paths = sorted(args.data.glob("atom_*.npz")) + [args.data / f"mol_{row['id']}.npz" for row in selected]
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"created_utc": datetime.now(timezone.utc).isoformat(),
              "identity": "E_xc[n_up,n_down]=-E_H[n] for N<=1 exact charge/spin ensembles",
              "hartree_approximation": "Coulomb fit in weigend auxiliary basis",
              "fitting_shapes_only": True, "entries": []}
    for path in paths:
        output = args.output / path.name
        if output.exists():
            with np.load(output) as cache:
                info = json.loads(str(cache["metadata"]))
            report["entries"].append(info)
            continue
        with np.load(path) as data:
            info = json.loads(str(data["metadata"]))
            coords, weights = data["coords"].astype(float), data["weights"]
            rho_total = data["rho"].astype(float).sum(0)
            landmarks = data["landmarks"]
        count = float(np.dot(weights, rho_total[0]))
        normalized = rho_total / count
        if info["id"].startswith("atom_"):
            atom = f"{info['symbols'][0]} 0 0 0"
        else:
            atom = list(zip(info["symbols"], info["geometry"]))
        mol = gto.M(atom=atom, unit="Bohr", basis=info["basis"],
                    charge=int(info.get("molecular_charge", 0)), spin=int(info.get("spin", 0)), verbose=0)
        auxiliary = df.addons.make_auxmol(mol, "weigend")
        metric = auxiliary.intor("int2c2e")
        potential = gto.intor_cross("int2c2e", auxiliary, gto.fakemol_for_charges(coords))
        rhs = potential @ (weights * normalized[0])
        coeff = np.linalg.solve(metric + np.eye(len(metric)) * 1e-12, rhs)
        hartree_potential = potential.T @ coeff
        hartree = float(np.dot(weights, normalized[0] * hartree_potential) / 2)
        metadata = {"id": info["id"], "source": str(path), "hartree_normalized": hartree,
                    "source_integrated_electrons": count, "fitting_shape": True}
        np.savez_compressed(output, metadata=json.dumps(metadata), rho_total=normalized.astype(np.float32),
                            hartree_potential=hartree_potential.astype(np.float32),
                            coords=coords.astype(np.float32), weights=weights, landmarks=landmarks)
        report["entries"].append(metadata)
        print(json.dumps({"id": metadata["id"], "one_electron_hartree": hartree}), flush=True)
        (args.output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"constraint_density_shapes": len(report["entries"])}), flush=True)


if __name__ == "__main__":
    main()
