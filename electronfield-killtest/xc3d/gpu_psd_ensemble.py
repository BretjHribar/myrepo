"""CUDA primal optimization of a positive fixed-trace ensemble.

ADMM/SVD/conic projection are established numerical primitives. The affine
constraints are checked explicitly; a finite penalty value is never accepted
as proof of density feasibility. The returned dual bound is for the supplied
reduced physical space, not automatically the full interacting functional.
"""
import torch


@torch.no_grad()
def affine_constraints(operators, target, relative_threshold=1e-12):
    flat = operators.flatten(1)
    left, singular, right = torch.linalg.svd(flat, full_matrices=False)
    selected = singular > singular[0]*relative_threshold
    left, singular, right = left[:, selected], singular[selected], right[selected]
    compatible = float((target-left @ (left.T @ target)).norm())
    if compatible > 1e-8:
        raise RuntimeError(f"Reduced density constraints have an incompatible target: {compatible}")
    return {"operators": right.reshape(-1, *operators.shape[1:]), "target": (left.T @ target)/singular,
            "left": left, "singular": singular, "compatibility_residual": compatible}


@torch.no_grad()
def optimize_ensemble(matrix, operators, target, feasible, maximum_iterations=12000,
                      feasibility_tolerance=1e-9, gap_tolerance=1e-7, callback=None):
    if matrix.device.type != "cuda":
        raise RuntimeError("Positive-ensemble optimization requires CUDA")
    size = len(matrix)
    identity = torch.eye(size, device=matrix.device, dtype=matrix.dtype)
    if float((operators[0]-identity).norm()) > 1e-10 or abs(float(target[0])-1) > 1e-10:
        raise ValueError("First affine constraint must be unit ensemble trace")
    affine = affine_constraints(operators, target)
    frame, requested = affine["operators"], affine["target"]
    matrix = (matrix+matrix.T)/2
    trace_free = matrix-identity*matrix.trace()/size
    penalty = max(1e-3, float(trace_free.norm())/(size**.5))
    z, scaled_dual = feasible.clone(), torch.zeros_like(feasible)
    best, best_energy = feasible.clone(), float((matrix*feasible).sum())
    history, latest = [], None
    for iteration in range(maximum_iterations+1):
        value = (z-scaled_dual-matrix/penalty)
        eigenvalues, eigenvectors = torch.linalg.eigh((value+value.T)/2)
        x = (eigenvectors*eigenvalues.clamp_min(0)) @ eigenvectors.T
        trial = x+scaled_dual
        alpha = torch.einsum("aij,ij->a", frame, trial)-requested
        previous = z
        z = trial-torch.einsum("a,aij->ij", alpha, frame)
        z = (z+z.T)/2
        scaled_dual = scaled_dual+x-z
        if iteration % 50 and iteration != maximum_iterations:
            continue
        population, vectors = torch.linalg.eigh(z)
        candidate = (vectors*population.clamp_min(0)) @ vectors.T
        candidate /= candidate.trace()
        residual = torch.einsum("aij,ij->a", operators, candidate)-target
        error = float(residual.norm())
        multipliers = affine["left"] @ (penalty*alpha/affine["singular"])
        slack = matrix+torch.einsum("a,aij->ij", multipliers, operators)
        minimum = torch.linalg.eigvalsh((slack+slack.T)/2)[0]
        lower = -multipliers @ target+minimum
        energy = (candidate*matrix).sum()
        gap = float(energy-lower)
        lagrangian_error = float((multipliers @ residual).abs())
        primal = float((x-z).norm())
        dual = float((penalty*(z-previous)).norm())
        if error < feasibility_tolerance and float(energy) < best_energy:
            best, best_energy = candidate, float(energy)
        row = {"iteration": iteration, "density_trace_constraint_residual": error,
               "primal_residual": primal, "dual_residual": dual, "primal_energy_hartree": float(energy),
               "reduced_space_dual_lower_hartree": float(lower), "reduced_space_gap_hartree": gap,
               "constraint_lagrangian_error_hartree": lagrangian_error}
        history.append(row)
        latest = {"ensemble": candidate, "potential": multipliers, "energy": energy,
                  "lower_bound": lower, "gap": gap, "density_residual": error,
                  "stationarity_slack": slack-minimum*identity, "history": history,
                  "affine_rank": len(frame), "affine_compatibility_residual": affine["compatibility_residual"]}
        if callback and (iteration % 500 == 0 or iteration == maximum_iterations):
            callback(row)
        if error < feasibility_tolerance and abs(gap) < gap_tolerance and lagrangian_error < gap_tolerance and primal < feasibility_tolerance:
            latest["converged"] = True
            return latest
    latest["converged"] = False
    latest["best_feasible_ensemble"] = best
    latest["best_feasible_energy_hartree"] = best_energy
    return latest
