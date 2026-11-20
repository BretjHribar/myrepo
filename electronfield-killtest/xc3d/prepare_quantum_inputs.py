"""Recover orbitals in one pass from cached fields, then cache new descriptors.

Builds the KS operator from existing densities and diagonalizes it once. It
does not rerun an SCF convergence study. Every new field/energy is evaluated on
the recovered determinant; the difference from its source is recorded.
"""
from .runtime import configure
ROOT = configure()

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import numpy as np
from pyscf import df, dft, gto, lib
from .quantum_features import overlap_roots, quantum_numpy


def transform(path, output):
    started = time.perf_counter()
    with np.load(path) as data:
        info = json.loads(str(data["metadata"]))
        coords = data["coords"].astype(float)
        weights = data["weights"]
        rho_old = data["rho"].astype(float)
        landmarks = data["landmarks"]
    if info["id"].startswith("atom_"):
        atom = f"{info['symbols'][0]} 0 0 0"
        spin = info["spin"]
    else:
        atom = list(zip(info["symbols"], info["geometry"]))
        spin = int(info["molecular_multiplicity"]) - 1
    mol = gto.M(atom=atom, unit="Bohr", basis=info["basis"], spin=spin,
                charge=int(info.get("molecular_charge", 0)), verbose=0, max_memory=1800)
    overlap = mol.intor("int1e_ovlp")
    root, inverse = overlap_roots(overlap)
    ao = dft.numint.eval_ao(mol, coords, deriv=1)
    auxiliary = df.addons.make_auxmol(mol)
    metric = auxiliary.intor("int2c2e")
    potential = gto.intor_cross("int2c2e", auxiliary, gto.fakemol_for_charges(coords))
    rhs = potential @ (weights * rho_old[:, 0].sum(0))
    coefficient = np.linalg.solve(metric, rhs)
    three_center = df.incore.aux_e2(mol, auxiliary, intor="int3c2e", aosym="s1")
    coulomb = np.einsum("ijp,p->ij", three_center, coefficient, optimize=True)
    hcore = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
    ni = dft.numint.NumInt()
    _, vxc, _, _ = ni.eval_xc_eff("pbe", rho_old, deriv=1, spin=1)
    matrices = []
    for sector in range(2):
        vmat = ao[0].T @ ((weights * vxc[sector, 0])[:, None] * ao[0])
        for axis in range(3):
            one_side = ao[axis+1].T @ ((weights * vxc[sector, axis+1])[:, None] * ao[0])
            vmat += one_side + one_side.T
        _, vectors = np.linalg.eigh(inverse.T @ (hcore + coulomb + vmat) @ inverse)
        occupied = inverse @ vectors[:, :mol.nelec[sector]]
        matrices.append(occupied @ occupied.T)
    dm = np.stack(matrices)
    rho = np.stack([ni.eval_rho(mol, ao, sector, xctype="GGA", hermi=1) for sector in dm])
    tau = np.stack([.5 * np.einsum("ij,dgi,dgj->g", sector, ao[1:4], ao[1:4], optimize=True) for sector in dm])
    density_fit_charge = np.einsum("ijp,ij->p", three_center, dm.sum(0), optimize=True)
    eh = float(density_fit_charge @ np.linalg.solve(metric, density_fit_charge) / 2)
    eps = ni.eval_xc_eff("pbe", rho, deriv=0, spin=1)[0]
    exc = float(np.dot(weights, eps * rho[:, 0].sum(0)))
    energy = float(np.sum(dm.sum(0) * hcore) + eh + exc + mol.energy_nuc())
    slices = mol.aoslice_by_atom()[:, 2:4].tolist()
    charge, coherence = quantum_numpy(dm, root, slices)
    relative_change = float(np.dot(weights, abs(rho[:, 0].sum(0) - rho_old[:, 0].sum(0))) / mol.nelectron)
    metadata = {**info, "energy_pbe_source": info["energy_pbe"], "energy_pbe": energy,
                "hartree_energy": eh, "reconstruction_density_l1_fraction": relative_change,
                "reconstruction_energy_change_hartree": energy - info["energy_pbe"],
                "descriptor": "Löwdin regional density-matrix charge and coherence",
                "orbital_source": "One KS-operator reconstruction from cached density; no SCF rerun",
                "descriptor_seconds": time.perf_counter() - started}
    np.savez_compressed(output, metadata=json.dumps(metadata), coords=coords.astype(np.float32),
                        weights=weights, landmarks=landmarks, rho=rho.astype(np.float32), tau=tau.astype(np.float32),
                        dm=dm, overlap=overlap, overlap_root=root, overlap_inverse_root=inverse,
                        atom_slices=np.array(slices), quantum_charge=charge.astype(np.float32),
                        quantum_coherence=coherence.astype(np.float32))
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/broad_tzvp_192")
    parser.add_argument("--output", type=Path, default=ROOT / "data/quantum_tzvp_191")
    args = parser.parse_args()
    lib.num_threads(1)
    old = json.loads((args.data / "manifest.json").read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "source": old["source"],
                "basis": old["basis"], "grid_level": old["grid_level"], "atoms": {}, "molecules": [], "failures": [],
                "selection_note": "Inherited Al/B-rich development panel; fixed composition splits retained",
                "descriptor": "Regional orbital coherence; single-pass reconstruction from saved fields"}
    for key in ("reactions", "evaluation_status", "missing_formulas"):
        if key in old:
            manifest[key] = old[key]
    if "reactions" in old:
        manifest["selection_note"] = old["evaluation_status"]
    paths = sorted(args.data.glob("atom_*.npz")) + [args.data / f"mol_{row['id']}.npz" for row in old["molecules"]]

    def record():
        (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    for path in paths:
        output = args.output / path.name
        try:
            if output.exists():
                with np.load(output) as data:
                    info = json.loads(str(data["metadata"]))
            else:
                info = transform(path, output)
            if path.name.startswith("atom_"):
                manifest["atoms"][info["symbols"][0]] = info
            else:
                info["tae_pbe"] = sum(manifest["atoms"][symbol]["energy_pbe"] for symbol in info["symbols"]) - info["energy_pbe"]
                with np.load(output) as data:
                    arrays = {name: data[name] for name in data.files if name != "metadata"}
                np.savez_compressed(output, metadata=json.dumps(info), **arrays)
                manifest["molecules"].append(info)
            print(json.dumps({"id": info["id"], "density_change": info["reconstruction_density_l1_fraction"],
                              "seconds": round(info["descriptor_seconds"], 2)}), flush=True)
        except Exception as exc:
            failure = {"id": path.stem, "exception": f"{type(exc).__name__}: {exc}"}
            manifest["failures"].append(failure)
            print(json.dumps(failure), flush=True)
        record()
    print(json.dumps({"molecules": len(manifest["molecules"]), "failures": len(manifest["failures"])}), flush=True)


if __name__ == "__main__":
    main()
