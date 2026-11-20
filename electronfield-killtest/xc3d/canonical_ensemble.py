"""Our canonical local-ensemble correction: first energy-landscape experiment.

Unlike an always-negative correlation add-on, this can correct fractional
charge upward and fractional spin downward. Integer spin counts and local mean
densities are conserved jointly. The first implementation is restricted to
regions containing at most one electron per spin; it is not a general submitted
functional. It uses existing input fields and never recalculates FCI labels.
"""
from .runtime import configure
ROOT = configure()

import argparse
from itertools import combinations
import json
from pathlib import Path
import time
import warnings
import numpy as np
from scipy.optimize import linprog
from scipy.special import softmax
from pyscf import dft, df, gto, lib

KCAL = 627.509474


def occupations(count, nelec):
    options = []
    for number in nelec:
        rows = []
        for occupied in combinations(range(count), number):
            row = np.zeros(count)
            row[list(occupied)] = 1
            rows.append(row)
        options.append(rows)
    return np.array([np.stack((up, down)) for up in options[0] for down in options[1]])


def xc_energy(ni, rho, weights):
    epsilon = ni.eval_xc_eff("pbe", rho, deriv=0, spin=1)[0]
    return float(np.dot(weights, epsilon * rho[:, 0].sum(0)))


def calculate(path, localization_factor=1.):
    started = time.perf_counter()
    with np.load(path) as data:
        info = json.loads(str(data["metadata"]))
        rho = data["rho"][0]
        coords, weights, landmarks = data["coords"], data["weights"], data["landmarks"]
        baseline, exact = float(data["energy_pbe"][0]), float(data["energy"][0])
    mol = gto.M(atom=info["atom"], basis=info["basis"], charge=info["charge"], spin=info["spin"], verbose=0)
    count = len(landmarks)
    displacement = coords[:, None, :] - landmarks[None, :, :]
    partition = softmax(-np.sum(displacement**2, axis=-1) / 2, axis=-1)
    center = partition @ landmarks
    grad_partition = partition[:, :, None] * (landmarks[None, :, :] - center[:, None, :])
    local_rho = rho[None, :, :, :] * partition.T[:, None, None, :]
    local_rho[:, :, 1:4, :] += np.einsum("sg,gad->asdg", rho[:, 0], grad_partition)
    raw_mass = np.einsum("g,asg->as", weights, local_rho[:, :, 0])
    mass = raw_mass.copy()
    for spin, number in enumerate(info["nelec"]):
        if number:
            mass[:, spin] *= number / mass[:, spin].sum()
    if np.max(mass) > 1 + 1e-4:
        raise ValueError("Region exceeds single-orbital capacity; orbital-resolved extension required")
    mass = np.clip(mass, 0, 1)
    number_density = rho[:, 0].sum(0)
    local_density = partition * number_density[:, None]
    total_mass = raw_mass.sum(-1).clip(min=1e-10)

    # Coulomb fitting of regional real-space densities, not an empirical
    # on-site interaction scale. The sum of regional densities is the input.
    auxiliary = df.addons.make_auxmol(mol, "weigend")
    metric = auxiliary.intor("int2c2e")
    point_charges = gto.fakemol_for_charges(coords)
    potential = gto.intor_cross("int2c2e", auxiliary, point_charges)
    rhs = potential @ (weights[:, None] * local_density)
    coeff = np.linalg.solve(metric + np.eye(len(metric)) * 1e-12, rhs)
    coulomb = rhs.T @ coeff
    interaction = coulomb / (total_mass[:, None] * total_mass[None, :])
    ni = dft.numint.NumInt()
    site = np.zeros((count, 3))
    current_site = np.zeros(count)
    for region in range(count):
        normalized = local_rho[region].sum(0) / total_mass[region]
        # A one-electron region has zero Hartree-plus-XC self energy exactly.
        site[region, 1] = 0.
        twice = np.stack((normalized, normalized))
        site[region, 2] = 2 * interaction[region, region] + xc_energy(ni, twice, weights)
        current_site[region] = .5 * coulomb[region, region] + xc_energy(ni, local_rho[region], weights)
    states = occupations(count, info["nelec"])
    totals = states.sum(1)
    costs = np.zeros(len(states))
    for index, state in enumerate(totals):
        costs[index] = sum(site[region, int(state[region])] for region in range(count))
        costs[index] += sum(state[a] * state[b] * interaction[a, b] for a in range(count) for b in range(a))
    equality = states.reshape(len(states), -1).T
    target = mass.T.flatten()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Unrecognized options detected")
        optimum = linprog(costs, A_eq=equality, b_eq=target, bounds=(0, None),
                          method="highs", options={"threads": 1})
    if not optimum.success:
        raise RuntimeError(f"Canonical marginal solve failed: {optimum.message}")
    product_interaction = sum(total_mass[a] * total_mass[b] * interaction[a, b]
                              for a in range(count) for b in range(a))
    uncorrelated = current_site.sum() + product_interaction
    delta = float(optimum.fun - uncorrelated)
    # Explicit localization cost suppresses the separated-region approximation
    # when density bridges remain strong. No fitted reference energies enter.
    variance = np.einsum("ga,gad,gad->g", partition,
                         landmarks[None, :, :] - center[:, None, :],
                         landmarks[None, :, :] - center[:, None, :])
    localization = float(np.dot(weights, number_density * variance) / 8)
    gate = float(np.exp(-localization_factor * localization / (abs(delta) + 1e-10)))
    corrected = baseline + gate * delta
    return {"id": info["id"], "family": info["family"], "split": info["split"],
            "pbe_error_kcal": (baseline - exact) * KCAL,
            "corrected_error_kcal": (corrected - exact) * KCAL,
            "ungated_correction_hartree": delta, "localization_hartree": localization, "gate": gate,
            "mass_up_down": mass.tolist(), "canonical_states": len(states),
            "marginal_residual": float(np.max(np.abs(equality @ optimum.x - target))),
            "coulomb_fitted_hartree": float(coulomb.sum() / 2),
            "seconds": time.perf_counter() - started}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/pilot_3d_ccpvdz")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--localization-factor", type=float, default=1.)
    args = parser.parse_args()
    lib.num_threads(1)
    report = {"status": "Original canonical-ensemble mechanism screen; frozen input fields, not a general XC submission",
              "localization_factor": args.localization_factor, "cases": [], "failures": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for path in sorted(args.data.glob("*.npz")):
        try:
            row = calculate(path, args.localization_factor)
            report["cases"].append(row)
            print(json.dumps(row), flush=True)
        except Exception as exc:
            failure = {"id": path.stem, "exception": f"{type(exc).__name__}: {exc}"}
            report["failures"].append(failure)
            print(json.dumps(failure), flush=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    report["summary"] = {}
    for split in ("train", "val", "test"):
        rows = [row for row in report["cases"] if row["split"] == split]
        if rows:
            report["summary"][split] = {"cases": len(rows),
                                        "pbe_mae_kcal": float(np.mean([abs(row["pbe_error_kcal"]) for row in rows])),
                                        "corrected_mae_kcal": float(np.mean([abs(row["corrected_error_kcal"]) for row in rows]))}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"summary": report["summary"], "failures": report["failures"]}), flush=True)


if __name__ == "__main__":
    main()
