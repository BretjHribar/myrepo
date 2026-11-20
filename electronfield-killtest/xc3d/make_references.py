"""Generate our 24-anchor, finite-basis 3D electronic-response pilot.

FCI labels are exact within cc-pVDZ, not CBS or real-chemistry accuracy.
Whole families and electron sectors are assigned before calculation.
"""
from .runtime import configure
ROOT = configure()

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import numpy as np
from pyscf import ao2mo, dft, fci, gto, lib


def specifications():
    cases = [dict(id="H_atom", family="H_atom", split="train", atom="H 0 0 0", charge=0, spin=1),
             dict(id="He_atom", family="He_atom", split="train", atom="He 0 0 0", charge=0, spin=0)]
    for family, distances, charge, spin, split in (
        ("H2", [.7, 1., 1.5, 2., 3.], 0, 0, "train"),
        ("HeH_plus", [.7, 1.2, 2.2], 1, 0, "train"),
        ("H2_plus", [.8, 1.6, 3.], 1, 1, "val"),
        ("H3_linear_plus", [.8, 1.2, 1.8], 1, 0, "val"),
        ("H3_triangle_plus", [.8, 1.2, 1.8], 1, 0, "test"),
        ("H4_square_2plus", [1., 1.5, 2.3], 2, 0, "test"),
        ("H3_neutral", [1., 1.8], 0, 1, "test"),
    ):
        for distance in distances:
            r = distance
            if family.startswith("HeH"):
                atom = f"He 0 0 0; H 0 0 {r}"
            elif family.startswith("H2"):
                atom = f"H 0 0 0; H 0 0 {r}"
            elif family == "H3_triangle_plus":
                atom = f"H 0 0 0; H {r} 0 0; H {r/2} {r*np.sqrt(3)/2} 0"
            elif family == "H4_square_2plus":
                atom = f"H 0 0 0; H {r} 0 0; H 0 {r} 0; H {r} {r} 0"
            else:
                atom = f"H 0 0 {-r}; H 0 0 0; H 0 0 {r}"
            cases.append(dict(id=f"{family}_{r:.2f}", family=family, split=split,
                              atom=atom, charge=charge, spin=spin, distance_angstrom=r))
    return cases


def spin_dm(mf):
    dm = np.asarray(mf.make_rdm1())
    return np.stack((dm / 2, dm / 2)) if dm.ndim == 2 else dm


def grid_rho(mol, ao, dm):
    return np.stack([dft.numint.eval_rho(mol, ao, d, xctype="GGA", hermi=1) for d in dm])


def calculate(case, directory, perturbations=True, basis="cc-pvdz", grid_level=1):
    start = time.perf_counter()
    mol = gto.M(atom=case["atom"], charge=case["charge"], spin=case["spin"],
                basis=basis, verbose=0, max_memory=1800)
    overlap = mol.intor("int1e_ovlp")
    values, vectors = np.linalg.eigh(overlap)
    orth = (vectors / np.sqrt(values)) @ vectors.T
    hcore = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
    eri_orth = ao2mo.kernel(mol, orth)
    solver = fci.direct_spin1.FCI(mol)
    solver.conv_tol = 1e-11
    solver.max_cycle = 150
    grids = dft.gen_grid.Grids(mol)
    grids.level = grid_level
    grids.build()
    ao = dft.numint.eval_ao(mol, grids.coords, deriv=1)
    landmarks = mol.atom_coords()
    reference_dm = []
    baseline_dm = []
    focks = []
    energies = []
    baseline_energies = []
    converged = []
    potentials = [np.zeros(len(grids.weights))]
    ci = None
    previous_dm = None
    field_labels = [dict(mode="unperturbed", amplitude=0.)]
    for index in range(9 if perturbations else 1):
        if index == 1:
            n0 = grid_rho(mol, ao, baseline_dm[0])[:, 0].sum(0)
            if len(landmarks) > 1:
                pairs = ((landmarks[:, None] - landmarks[None, :])**2).sum(-1)
                left, right = np.unravel_index(pairs.argmax(), pairs.shape)
                direction = landmarks[right] - landmarks[left]
                direction /= np.linalg.norm(direction)
                center = (landmarks[left] + landmarks[right]) / 2
            else:
                direction = np.array([0., 0., 1.])
                center = landmarks[0]
            axis = (grids.coords - center) @ direction
            modes = [axis, np.tanh(axis / .8)]
            for mode_number, mode in enumerate(modes):
                mean = np.dot(grids.weights * n0, mode) / mol.nelectron
                mode = mode - mean
                norm = np.sqrt(np.dot(grids.weights * n0, mode**2) / mol.nelectron)
                mode /= max(norm, 1e-8)
                for amplitude in (.002, -.002, .004, -.004):
                    potentials.append(mode * amplitude)
                    field_labels.append(dict(mode=mode_number, amplitude=amplitude))
        external = np.einsum("gi,g,gj->ij", ao[0], grids.weights * potentials[index], ao[0], optimize=True)
        one_body = hcore + external
        h1 = orth.T @ one_body @ orth
        exact_e, ci = solver.kernel(h1, eri_orth, mol.nao_nr(), mol.nelec,
                                    ci0=ci, ecore=mol.energy_nuc())
        if not solver.converged:
            raise RuntimeError(f"FCI did not converge: {case['id']} state {index}")
        dm_orb = solver.make_rdm1s(ci, mol.nao_nr(), mol.nelec)
        reference_dm.append(np.stack([orth @ d @ orth.T for d in dm_orb]))
        mf = dft.UKS(mol) if mol.spin else dft.RKS(mol)
        mf.xc = "pbe"
        mf.grids = grids
        mf.get_hcore = lambda *args, h=one_body: h
        mf.conv_tol = 1e-10
        mf.max_cycle = 100
        mf.damp = .15
        base_e = mf.kernel(dm0=previous_dm)
        if not mf.converged:
            mf = mf.newton()
            base_e = mf.kernel()
        if not mf.converged:
            raise RuntimeError(f"Reference-input PBE did not converge: {case['id']} state {index}")
        previous_dm = mf.make_rdm1()
        dm = spin_dm(mf)
        veff = mf.get_veff(mol, previous_dm)
        fock = one_body + np.asarray(veff)
        if fock.ndim == 2:
            fock = np.stack((fock, fock))
        baseline_dm.append(dm)
        focks.append(fock)
        energies.append(float(exact_e))
        baseline_energies.append(float(base_e))
        converged.append(bool(mf.converged))
    metadata = {**case, "basis": basis, "grid_level": grid_level,
                "reference": "FCI within finite basis", "input_method": "PBE",
                "field_states": field_labels, "nelec": list(mol.nelec),
                "pbe_converged": converged, "fci_converged": True,
                "seconds": time.perf_counter() - start}
    rhos = np.stack([grid_rho(mol, ao, dm) for dm in baseline_dm])
    rhos_ref = np.stack([grid_rho(mol, ao, dm) for dm in reference_dm])
    np.savez_compressed(directory / f"{case['id']}.npz", metadata=json.dumps(metadata),
                        coords=grids.coords, weights=grids.weights, landmarks=landmarks,
                        ao=ao, orth=orth, dm=np.asarray(baseline_dm), dm_ref=np.asarray(reference_dm),
                        rho=rhos, rho_ref=rhos_ref, fock=np.asarray(focks),
                        energy=np.asarray(energies), energy_pbe=np.asarray(baseline_energies))
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=24)
    parser.add_argument("--no-perturbations", action="store_true")
    parser.add_argument("--basis", default="cc-pvdz")
    parser.add_argument("--grid-level", type=int, default=1)
    parser.add_argument("--output", type=Path, default=ROOT / "data/pilot_3d_ccpvdz")
    args = parser.parse_args()
    lib.num_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    spec = specifications()
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(),
                "schema": 1, "basis": args.basis, "grid_level": args.grid_level,
                "states_per_anchor": 1 if args.no_perturbations else 9,
                "specifications": spec, "results": []}
    (args.output / "specifications.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for case in spec[:args.limit]:
        path = args.output / f"{case['id']}.npz"
        if path.exists():
            with np.load(path) as cached:
                result = json.loads(str(cached["metadata"]))
            if result["basis"] != args.basis or len(result["field_states"]) != manifest["states_per_anchor"]:
                raise RuntimeError(f"Incompatible cached reference: {path}")
        else:
            result = calculate(case, args.output, not args.no_perturbations, args.basis, args.grid_level)
        manifest["results"].append(result)
        (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(json.dumps({"case": case["id"], "split": case["split"],
                          "seconds": round(result["seconds"], 2),
                          "states": len(result["field_states"])}), flush=True)


if __name__ == "__main__":
    main()
