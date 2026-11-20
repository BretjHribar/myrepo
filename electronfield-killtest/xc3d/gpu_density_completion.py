"""Physical residual completion for density fibres and molecular states.

Hypothesis: let the universal Hamiltonian and density response generate missing
collective directions, instead of adding fitted chemistry-specific channels.
Uses ordinary orthogonal subspace/Ritz mathematics, not a novelty claim for
those primitives. The present dense occupation action is a small-basis research
implementation, not a scalable DFA or an amortized learned predictor.
"""
from .runtime import configure, torch_setup
ROOT = configure(gpu_only_compatible=True)

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import torch
from .gpu_spin_chain import SpinNumberState, amplitudes
from .gpu_moment_functional import prepare, oracle
from .gpu_manybody_ops import dense_operator

KCAL = 627.509474


@torch.no_grad()
def append_directions(basis, candidates, maximum):
    if candidates.ndim == 1:
        candidates = candidates[:, None]
    for direction in candidates.unbind(1):
        if basis.shape[1] >= maximum:
            break
        original_norm = direction.norm()
        for _ in range(2):
            direction = direction-basis @ (basis.T @ direction)
        size = direction.norm()
        if float(size) > max(1e-12, float(original_norm)*1e-10):
            basis = torch.cat((basis, (direction/size)[:, None]), 1)
    return basis


@torch.no_grad()
def complete_molecular(matrix, seed, maximum=64, tolerance=1e-7):
    """Reference-free residual stopping; no exact energy is an input."""
    basis = (seed/seed.norm())[:, None]
    history = []
    for iteration in range(maximum):
        projected = basis.T @ matrix @ basis
        values, vectors = torch.linalg.eigh((projected+projected.T)/2)
        state = basis @ vectors[:, 0]
        energy = values[0]
        residual = matrix @ state-energy*state
        error = float(residual.norm())
        history.append({"dimension": basis.shape[1], "energy_hartree": float(energy), "residual_norm": error})
        if error < tolerance or basis.shape[1] >= maximum:
            break
        updated = append_directions(basis, residual, maximum)
        if updated.shape[1] == basis.shape[1]:
            break
        basis = updated
    return {"state": state, "energy": energy, "residual": error, "history": history,
            "converged": error < tolerance, "dimension": basis.shape[1]}


@torch.no_grad()
def complete_density(matrix, moments, target, seed, maximum=96, tolerance=1e-7):
    """Adaptive constrained subspace; all accepted states meet the moments.

    Q_a|Psi> directions supply the complete local density response. The
    (T+W+lambda.Q)|Psi> residual supplies correlation outside that response.
    The reference energy/potential never supplies a direction or stopping rule.
    """
    state = seed/seed.norm()
    basis = state[:, None]
    basis = append_directions(basis, torch.einsum("aij,j->ia", moments, state), maximum)
    basis = append_directions(basis, matrix @ state, maximum)
    potential = torch.zeros_like(target)
    history = []
    for iteration in range(maximum):
        projected = {"universal_singlet": basis.T @ matrix @ basis,
                     "projected_moments": torch.einsum("ia,mij,jb->mab", basis, moments, basis)}
        dual = oracle(projected, target, potential)
        potential = dual["potential"]
        small = projected["universal_singlet"]+torch.einsum("a,aij->ij", potential, projected["projected_moments"])
        values, vectors = torch.linalg.eigh((small+small.T)/2)
        state = basis @ vectors[:, 0]
        action = matrix @ state+torch.einsum("a,aij,j->i", potential, moments, state)
        residual = action-values[0]*state
        error = float(residual.norm())
        density_error = float((torch.einsum("i,aij,j->a", state, moments, state)-target).norm())
        energy = state @ matrix @ state
        row = {"iteration": iteration+1, "dimension": basis.shape[1],
               "functional_hartree": float(energy), "density_residual": density_error,
               "stationarity_residual": error}
        history.append(row)
        print(json.dumps(row), flush=True)
        if density_error > 1e-6:
            return {"status": "projected_density_solve_failed", "state": state, "potential": potential,
                    "energy": energy, "residual": error, "density_residual": density_error, "history": history}
        if (error < tolerance and density_error < 1e-8) or basis.shape[1] >= maximum:
            break
        updated = append_directions(basis, residual, maximum)
        updated = append_directions(updated, torch.einsum("aij,j->ia", moments, state), maximum)
        if updated.shape[1] == basis.shape[1]:
            break
        basis = updated
    return {"status": "converged" if error < tolerance and density_error < 1e-8 else "capacity_limit",
            "state": state, "potential": potential, "energy": energy, "residual": error,
            "density_residual": density_error, "history": history}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("research/density_residual_completion_gpu_20261116.json"))
    args = parser.parse_args()
    device = torch_setup()
    started = time.perf_counter()
    data = prepare(device)
    matrix = dense_operator(data["universal_words"], data["universal_coefficient"], data["bits"], 6)
    moments = torch.stack([dense_operator(data["q_words"], row, data["bits"], 6) for row in data["q_coefficients"]])
    checkpoint = torch.load("research/feasible_density_functional_gpu_20261116.pt", map_location=device, weights_only=True)
    model = SpinNumberState(6, 4, channels=2, device=device)
    model.load_state_dict(checkpoint["model"])
    seed = amplitudes(model.cores(), data["configurations"]).detach()
    result = complete_density(matrix, moments, data["target"], seed)
    # Diagnostic reference is evaluated after the candidate has stopped.
    exact = oracle(data, data["target"])
    spin = result["state"] @ data["spin_matrix"] @ result["state"]
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "device": device,
              "scope": "Adaptive universal T+W constrained search, N=4 active-six, frozen spatial-moment frame",
              "status": result["status"], "reference_energy_fitted": False,
              "external_nuclear_potential_in_objective": False,
              "density_moment_residual": result["density_residual"], "stationarity_residual": result["residual"],
              "functional_error_kcal": float((result["energy"]-exact["functional"])*KCAL),
              "potential_difference_norm": float((result["potential"]-exact["potential"]).norm()),
              "spin_squared": float(spin), "history": result["history"],
              "seconds": time.perf_counter()-started, "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
              "limitations": ["Dense finite-basis action", "Incomplete spatial-density constraints",
                              "No full XC subtraction or self-consistency", "No broad hidden-suite evidence",
                              "Finite global subspaces do not guarantee fragment separability"]}
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    torch.save({"state": result["state"].cpu(), "potential": result["potential"].cpu(), "report": report}, args.output.with_suffix(".pt"))
    print(json.dumps({key: value for key, value in report.items() if key != "history"}), flush=True)


if __name__ == "__main__":
    main()
