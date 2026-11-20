"""CUDA ensemble duals with explicit density and number constraints.

Canonical statistics act inside an already fixed-N,S many-body sector.
One-body statistics minimize a binary-entropy regularized kinetic functional
over 0<=D<=2 with Tr(D)=N enforced by an explicit number potential. The final
fractional 1RDM is canonical-ensemble representable; no fluctuating-N state is
used as the physical input. Entropic dual values are lower bounds at zero T.
"""
import math
import torch
from .gpu_density_completion import append_directions


@torch.no_grad()
def evaluate(matrix, operators, target, potential, beta, statistics):
    values, vectors = torch.linalg.eigh(matrix+torch.einsum("a,aij->ij", potential, operators))
    rotated = torch.einsum("ia,mij,jb->mab", vectors, operators, vectors)
    if statistics == "canonical":
        shifted = values-values[0]
        population = torch.softmax(-beta*shifted, 0)
        free = values[0]-torch.logsumexp(-beta*shifted, 0)/beta
        derivative = -beta*population
    elif statistics == "onebody":
        population = 2*torch.sigmoid(-beta*values)
        free = -2*torch.nn.functional.softplus(-beta*values).sum()/beta
        derivative = -beta*population*(1-population/2)
    else:
        raise ValueError("Unknown ensemble statistics")
    density = torch.einsum("i,aii->a", population, rotated)
    difference = values[:, None]-values[None]
    pdifference = population[:, None]-population[None]
    nearby = (beta*difference).abs() < 1e-4
    divided = pdifference/torch.where(nearby, torch.ones_like(difference), difference)
    divided = torch.where(nearby, (derivative[:, None]+derivative[None])/2, divided)
    if statistics == "canonical":
        # Center the diagonal covariance before multiplying by beta. Avoid
        # cancellation of two O(beta) matrices near the zero-temperature limit.
        divided.fill_diagonal_(0.)
        centered = rotated.diagonal(dim1=-2, dim2=-1)-density[:, None]
        response = torch.einsum("aij,ij,bij->ab", rotated, divided, rotated)
        response -= beta*torch.einsum("i,ai,bi->ab", population, centered, centered)
    else:
        response = torch.einsum("aij,ij,bij->ab", rotated, divided, rotated)
    return {"dual": free-potential @ target, "density": density, "response": response,
            "values": values, "vectors": vectors, "population": population,
            "free": free, "energy": (population*values).sum()-potential @ density}


@torch.no_grad()
def ensemble_dual(matrix, operators, target, beta, statistics="canonical", initial=None, tolerance=1e-9):
    potential = torch.zeros_like(target) if initial is None else initial.clone()
    stages = [value for value in (.1, 1., 10., 100., 1000., 10000., 100000.) if value < beta]+[beta]
    # Set a physical trust radius from operator energies, rather than an
    # arbitrary sub-Hartree cap that cannot reach core-density potentials.
    spectrum = torch.linalg.eigvalsh(matrix)
    radius = max(1., float((spectrum[-1]-spectrum[0]).abs())/max(1., float(operators.flatten(1).norm(dim=1).max())))
    history = []
    for stage in stages:
        for iteration in range(80):
            point = evaluate(matrix, operators, target, potential, stage, statistics)
            residual = point["density"]-target
            error = float(residual.norm())
            if error < tolerance:
                break
            response = (point["response"]+point["response"].T)/2
            curvature, directions = torch.linalg.eigh(response)
            scale = curvature.abs().max().clamp_min(1e-12)
            inverse = 1/curvature.clamp_max(-max(1e-12, float(scale)*1e-12))
            update = -directions @ (inverse*(directions.T @ residual))
            update *= torch.clamp(radius/update.norm().clamp_min(1e-30), max=1.)
            accepted = False
            for power in range(30):
                trial = potential+update*(.5**power)
                check = evaluate(matrix, operators, target, trial, stage, statistics)
                if float(check["dual"]) >= float(point["dual"])-1e-12:
                    # At roundoff-sized dual changes, prefer density progress.
                    if float(check["dual"]-point["dual"]) > 1e-12 or float((check["density"]-target).norm()) < error:
                        potential = trial
                        accepted = True
                        break
            if not accepted:
                break
        final = evaluate(matrix, operators, target, potential, stage, statistics)
        history.append({"beta": stage, "iterations": iteration+1,
                        "density_residual": float((final["density"]-target).norm()),
                        "dual": float(final["dual"])})
    final.update(potential=potential, residual=float((final["density"]-target).norm()), history=history)
    return final


@torch.no_grad()
def ensemble_completion(matrix, operators, target, seed, beta, maximum=192, tolerance=1e-7):
    seeds = seed[:, None] if seed.ndim == 1 else seed
    basis = seed.new_zeros(len(matrix), 0)
    basis = append_directions(basis, seeds, maximum)
    initial = basis.clone()
    basis = append_directions(basis, torch.einsum("aij,jk->iak", operators, initial).flatten(1), maximum)
    basis = append_directions(basis, matrix @ initial, maximum)
    history, potential = [], None
    for iteration in range(maximum):
        reduced_matrix = basis.T @ matrix @ basis
        reduced_operators = torch.einsum("ia,mij,jb->mab", basis, operators, basis)
        result = ensemble_dual(reduced_matrix, reduced_operators, target, beta, initial=potential)
        potential = result["potential"]
        states = basis @ result["vectors"]
        density_operator = (states*result["population"]) @ states.T
        universal_energy = torch.trace(density_operator @ matrix)
        density_residual = float((torch.einsum("ij,aij->a", density_operator, operators)-target).norm())
        effective = matrix+torch.einsum("a,aij->ij", potential, operators)
        residual = (effective @ states-states*result["values"])*result["population"].sqrt()
        stationarity = float(residual.norm())
        row = {"iteration": iteration+1, "dimension": basis.shape[1], "density_residual": density_residual,
               "stationarity_residual": stationarity, "universal_energy_hartree": float(universal_energy)}
        history.append(row); print(row, flush=True)
        if density_residual > 1e-7:
            status = "ensemble_density_solve_failed"
            break
        if stationarity < tolerance or basis.shape[1] >= maximum:
            status = "converged" if stationarity < tolerance else "capacity_limit"
            break
        u, singular, _ = torch.linalg.svd(residual, full_matrices=False)
        candidates = u[:, singular > max(1e-12, float(singular[0])*1e-8)][:, :8]
        updated = append_directions(basis, candidates, maximum)
        active = states[:, result["population"] > 1e-10][:, :4]
        responses = torch.einsum("aij,jk->iak", operators, active).flatten(1)
        updated = append_directions(updated, responses, maximum)
        if updated.shape[1] == basis.shape[1]:
            status = "no_independent_completion"
            break
        basis = updated
    # Canonical entropy in this fixed sector is <=log(full dimension), so
    # free energy + this constant is an upper estimate of zero-T F_M once
    # the density constraint is met. The offset is independent of q.
    entropy_allowance = math.log(len(matrix))/beta
    entropy_term = (result["population"]*result["population"].clamp_min(1e-300).log()).sum()/beta
    return {"status": status, "density_operator": density_operator,
            "energy": universal_energy, "functional": universal_energy+entropy_term+entropy_allowance,
            "entropy_allowance": entropy_allowance, "potential": potential,
            "density_residual": density_residual, "stationarity_residual": stationarity,
            "history": history, "dual_history": result["history"]}
