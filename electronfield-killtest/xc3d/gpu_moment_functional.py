"""First density-constrained universal-energy experiment with our spin state.

Defines F_M(q)=min_{Gamma: <Q>=q} Tr Gamma(T+W) in a fixed moment frame.
This is a finite-moment functional, not yet the full-grid XC functional. The
external nuclear potential is excluded from the constrained objective. A new
CUDA oracle checks the approximation and dual potential; it is not used as the
tensor state's optimization target.
"""
from .runtime import configure, torch_setup
ROOT = configure(gpu_only_compatible=True)

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import numpy as np
import torch
from .gpu_four_electron import prepare as prepare_four, bounded_reference, transform_eri
from .gpu_geminal import electronic_hamiltonian
from .gpu_density_frames import moment_operators
from .gpu_manybody_ops import molecular_words, dense_operator, expectation
from .gpu_spin_chain import SpinNumberState, norm, expect_product

KCAL = 627.509474


def one_body_words(matrices):
    count = matrices.shape[-1]
    words = [((i, spin, 1), (j, spin, 0), (-1, 0, 0), (-1, 0, 0))
             for i in range(count) for j in range(count) for spin in range(2)]
    return (torch.tensor(words, device=matrices.device, dtype=torch.long),
            matrices.flatten(-2).repeat_interleave(2, -1))


@torch.no_grad()
def prepare(device):
    destination = ROOT / "data/density_moment_functional_4e.pt"
    if destination.exists():
        return torch.load(destination, map_location=device, weights_only=True)
    data = prepare_four(device, 6)
    source = "H4_square_2plus_1.50"
    operator = torch.load(ROOT / "data/gpu_pair_state" / f"{source}.pt", map_location=device, weights_only=True)
    with np.load(ROOT / "data/pilot_3d_ccpvdz" / f"{source}.npz") as saved:
        dm, ao, orth, fock, weights, coords, landmarks = [torch.from_numpy(saved[key]).to(device) for key in
                                                       ("dm", "ao", "orth", "fock", "weights", "coords", "landmarks")]
    h = electronic_hamiltonian(dm[0], ao, operator["operator_eri"], fock[0], weights)
    full_h = orth.T @ h @ orth
    full_eri = transform_eri(operator["operator_eri"], orth)
    orbitals, status = bounded_reference(full_h, full_eri)
    selected = orbitals[:, :6]
    reconstructed = selected.T @ full_h @ selected
    discrepancy = float((reconstructed-data["h"]).abs().max())
    if discrepancy > 1e-9:
        raise RuntimeError("Active orbital gauge changed; density moments cannot be attached to the old Hamiltonian")
    orbital_map = orth @ selected
    density = 2*(ao[0] @ orbital_map[:, :2]).square().sum(-1)
    moments, frame = moment_operators(coords, weights, landmarks, ao[0], orbital_map, density)
    kinetic_ao = .5*sum(ao[axis].T @ (weights[:, None]*ao[axis]) for axis in (1, 2, 3))
    kinetic = orbital_map.T @ kinetic_ao @ orbital_map
    words, coefficient = molecular_words(kinetic, data["eri"])
    qw, qc = one_body_words(moments)
    q_matrices = torch.stack([dense_operator(qw, row, data["bits"], 6) for row in qc])
    spin_values, spin_vectors = torch.linalg.eigh(data["spin_matrix"])
    singlets = spin_vectors[:, spin_values.abs() < 1e-8]
    physical = singlets.T @ data["matrix"] @ singlets
    _, vectors = torch.linalg.eigh((physical+physical.T)/2)
    reference_state = singlets @ vectors[:, 0]
    target = torch.einsum("i,aij,j->a", reference_state, q_matrices, reference_state)
    universal = dense_operator(words, coefficient, data["bits"], 6)
    universal = singlets.T @ universal @ singlets
    projected_q = torch.einsum("ia,mij,jb->mab", singlets, q_matrices, singlets)
    data = {**data, "kinetic": kinetic, "moments": moments, "q_words": qw, "q_coefficients": qc,
            "universal_words": words, "universal_coefficient": coefficient,
            "universal_singlet": universal, "projected_moments": projected_q,
            "target": target, "frame": frame, "active_gauge_discrepancy": discrepancy,
            "kinetic_min_eigenvalue": float(torch.linalg.eigvalsh(kinetic).min())}
    torch.save({key: value.cpu() if isinstance(value, torch.Tensor) else
                {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in value.items()} if isinstance(value, dict)
                else value for key, value in data.items()}, destination)
    return data


@torch.no_grad()
def oracle(data, target, initial=None):
    h, q = data["universal_singlet"], data["projected_moments"]
    potential = torch.zeros_like(target) if initial is None else initial.clone()
    history = []
    for iteration in range(60):
        matrix = h+torch.einsum("a,aij->ij", potential, q)
        values, vectors = torch.linalg.eigh((matrix+matrix.T)/2)
        transitions = torch.einsum("i,aij,jk->ak", vectors[:, 0], q, vectors)
        density = transitions[:, 0]
        residual = density-target
        dual = values[0]-potential @ target
        error = float(residual.norm())
        history.append({"iteration": iteration+1, "density_moment_residual": error, "dual": float(dual)})
        if error < 1e-9:
            break
        gap = (values[1:]-values[0]).clamp_min(1e-8)
        response = -2*(transitions[:, 1:]/gap) @ transitions[:, 1:].T
        update = -torch.linalg.solve(response-torch.eye(len(target), device=h.device)*1e-8, residual)
        update = update*torch.clamp(.75/update.norm().clamp_min(1e-30), max=1.)
        accepted = False
        for scale in (1., .5, .25, .125, .0625, .03125):
            trial = potential+scale*update
            energy = torch.linalg.eigvalsh(h+torch.einsum("a,aij->ij", trial, q))[0]-trial @ target
            if float(energy) >= float(dual)-1e-12:
                potential = trial
                accepted = True
                break
        if not accepted:
            break
    return {"potential": potential, "functional": dual, "density": density,
            "residual": error, "history": history}


def observables(cores, data):
    operators = __import__("xc3d.gpu_manybody_ops", fromlist=["jw_products"]).jw_products(data["q_words"], 6)
    values = expect_product(cores, operators)/norm(cores)
    return data["q_coefficients"] @ values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outer", type=int, default=10)
    parser.add_argument("--inner", type=int, default=100)
    parser.add_argument("--output", type=Path, default=Path("research/density_moment_functional_gpu_20261116.json"))
    args = parser.parse_args()
    device = torch_setup()
    started = time.perf_counter()
    data = prepare(device)
    exact = oracle(data, data["target"])
    if exact["residual"] > 1e-6:
        raise RuntimeError(f"New constrained-search oracle did not satisfy its moments: {exact['residual']}")
    state = SpinNumberState(6, 4, channels=2, device=device)
    checkpoint = torch.load(Path("research/general_spin_state_4e_gpu_20261116.pt"), map_location=device, weights_only=True)
    state.load_state_dict(checkpoint["model"])
    potential = torch.zeros_like(data["target"])
    penalty = 1.
    optimizer = torch.optim.Adam(state.parameters(), lr=.01)
    previous = None
    history = []
    for outer in range(args.outer):
        for inner in range(args.inner):
            optimizer.zero_grad(set_to_none=True)
            cores = state.cores()
            energy = expectation(cores, data["universal_words"], data["universal_coefficient"])
            residual = observables(cores, data)-data["target"]
            loss = energy+potential @ residual+.5*penalty*residual.square().sum()
            loss.backward()
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("Nonfinite density-constrained state optimization")
            torch.nn.utils.clip_grad_norm_(state.parameters(), 10.)
            optimizer.step()
        with torch.no_grad():
            cores = state.cores()
            energy = expectation(cores, data["universal_words"], data["universal_coefficient"])
            residual = observables(cores, data)-data["target"]
            size = float(residual.norm())
            potential = potential+penalty*residual
            row = {"outer": outer+1, "universal_energy_hartree": float(energy),
                   "moment_residual": size, "oracle_difference_kcal": float((energy-exact["functional"])*KCAL),
                   "potential_difference_norm": float((potential-exact["potential"]).norm()), "penalty": penalty,
                   "seconds": time.perf_counter()-started}
            history.append(row)
            print(json.dumps(row), flush=True)
            if previous is not None and size > .7*previous:
                penalty = min(256., penalty*2)
            previous = size
    # A new local density perturbation checks the oracle's Legendre derivative;
    # this does not stand in for agreement of the approximate state's potential.
    direction = torch.arange(1, len(potential)+1, device=device, dtype=potential.dtype)
    direction = direction/direction.norm()
    step = 1e-5
    plus = oracle(data, data["target"]+step*direction, exact["potential"])
    minus = oracle(data, data["target"]-step*direction, exact["potential"])
    finite = (plus["functional"]-minus["functional"])/(2*step)
    analytic = -exact["potential"] @ direction
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "device": device,
              "scope": "First universal kinetic-plus-Coulomb functional with fixed spatial density moments, N=4 active-six case. Not a full-grid/CBS XC result.",
              "objective_excludes_external_potential": True,
              "moment_count": len(potential), "frame": {key: value.tolist() if isinstance(value, torch.Tensor) else value for key, value in data["frame"].items()},
              "oracle_functional_hartree": float(exact["functional"]), "oracle_moment_residual": exact["residual"],
              "state_history": history, "state_constraint_residual": history[-1]["moment_residual"],
              "state_functional_difference_kcal": history[-1]["oracle_difference_kcal"],
              "oracle_derivative": {"finite_difference": float(finite), "dual_prediction": float(analytic),
                                     "absolute_difference": float((finite-analytic).abs()),
                                     "perturbed_max_moment_residual": max(plus["residual"], minus["residual"])},
              "limitations": ["Moment constraints are not the full spatial density", "Moment frame is frozen in this local derivative experiment",
                               "Tensor-state constraints and potential require their own convergence evidence", "No broad-chemistry or hidden-suite claim"],
              "seconds": time.perf_counter()-started, "peak_cuda_bytes": torch.cuda.max_memory_allocated()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({key: report[key] for key in ("moment_count", "state_constraint_residual", "state_functional_difference_kcal", "oracle_derivative")}), flush=True)


if __name__ == "__main__":
    main()
