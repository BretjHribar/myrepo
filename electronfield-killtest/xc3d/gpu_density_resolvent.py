"""Finite-space proximal density functionals with positive physical states.

Moreau regularization and accelerated projected gradients are established
mathematics. Here the auxiliary density is explicitly distinguished from the
positive fixed-N state density, and the external-energy counterterm is exact.
"""
import math
import torch


@torch.no_grad()
def capped_population(values, total, cap):
    if cap is None:
        ordered = values.sort(descending=True).values
        count = torch.arange(1, len(values)+1, device=values.device)
        offsets = (ordered.cumsum(0)-total)/count
        active = ordered > offsets
        last = torch.nonzero(active).flatten()[-1]
        return (values-offsets[last]).clamp_min(0)
    breaks = torch.cat((values, values-cap)).sort().values
    low, high = breaks[:-1], breaks[1:]
    middle = (low+high)/2
    difference = values[None]-middle[:, None]
    active = (difference > 0) & (difference < cap)
    saturated = difference >= cap
    counts = active.sum(-1)
    threshold = ((active*values[None]).sum(-1)+cap*saturated.sum(-1)-total)/counts.clamp_min(1)
    plateau = (counts == 0) & (cap*saturated.sum(-1) == total)
    threshold = torch.where(plateau, middle, threshold)
    allowed = ((counts > 0) | plateau) & (threshold >= low-1e-12) & (threshold <= high+1e-12)
    if not bool(allowed.any()):
        raise RuntimeError("Pauli/trace projection found no valid occupation threshold")
    selected = torch.nonzero(allowed).flatten()[0]
    return (values-threshold[selected]).clamp(0, cap)


@torch.no_grad()
def project_state(matrix, total=1., cap=None):
    values, vectors = torch.linalg.eigh((matrix+matrix.T)/2)
    population = capped_population(values, total, cap)
    return (vectors*population) @ vectors.T


@torch.no_grad()
def proximal_state(matrix, operators, auxiliary, epsilon, initial, total=1., cap=None,
                   maximum=4000, gap_tolerance=1e-11, callback=None, checkpoint=None, checkpoint_callback=None):
    if matrix.device.type != "cuda" or epsilon <= 0:
        raise ValueError("The density resolvent requires CUDA and positive epsilon")
    matrix = (matrix+matrix.T)/2
    gram = operators.flatten(1) @ operators.flatten(1).T
    lipschitz = max(1e-8, float(torch.linalg.eigvalsh((gram+gram.T)/2)[-1])/epsilon)
    def objective(value):
        moments = torch.einsum("aij,ij->a", operators, value)
        return (matrix*value).sum()+(moments-auxiliary).square().sum()/(2*epsilon)
    if checkpoint is None:
        state = project_state(initial, total, cap)
        point, acceleration = state.clone(), 1.
        energy = objective(state)
        history, start, gap, iteration = [], 0, float("inf"), -1
    else:
        if (checkpoint["epsilon"], checkpoint["total"], checkpoint["cap"], checkpoint["maximum"], checkpoint["gap_tolerance"]) != (epsilon, total, cap, maximum, gap_tolerance):
            raise ValueError("Proximal checkpoint settings differ from the original bounded solve")
        if not torch.equal(checkpoint["auxiliary"], auxiliary):
            raise ValueError("Proximal checkpoint belongs to a different auxiliary point")
        state, point, acceleration, energy = (checkpoint[key] for key in ("state", "point", "acceleration", "energy"))
        history, gap, iteration = checkpoint["history"], checkpoint["gap"], checkpoint["iteration"]
        start = maximum+1 if abs(gap) < gap_tolerance else iteration+1
    for iteration in range(start, maximum+1):
        moments = torch.einsum("aij,ij->a", operators, point)
        gradient = matrix+torch.einsum("a,aij->ij", (moments-auxiliary)/epsilon, operators)
        proposal = project_state(point-gradient/lipschitz, total, cap)
        proposed_energy = objective(proposal)
        if float(proposed_energy) > float(energy)+1e-12:
            point, acceleration = state, 1.
            moments = torch.einsum("aij,ij->a", operators, point)
            gradient = matrix+torch.einsum("a,aij->ij", (moments-auxiliary)/epsilon, operators)
            proposal = project_state(point-gradient/lipschitz, total, cap)
            proposed_energy = objective(proposal)
        next_acceleration = (1+math.sqrt(1+4*acceleration*acceleration))/2
        point = proposal+(acceleration-1)/next_acceleration*(proposal-state)
        state, energy, acceleration = proposal, proposed_energy, next_acceleration
        if iteration % 25 and iteration != maximum:
            continue
        moments = torch.einsum("aij,ij->a", operators, state)
        effective = matrix+torch.einsum("a,aij->ij", (moments-auxiliary)/epsilon, operators)
        values = torch.linalg.eigvalsh((effective+effective.T)/2)
        if cap is None:
            linear_minimum = total*values[0]
        else:
            full = int(total/cap)
            fraction = total-full*cap
            linear_minimum = cap*values[:full].sum()
            if fraction:
                linear_minimum += fraction*values[full]
        gap = float((effective*state).sum()-linear_minimum)
        row = {"iteration": iteration, "regularized_functional_hartree": float(energy),
               "reduced_space_frank_wolfe_gap_hartree": gap}
        history.append(row)
        if callback and (iteration % 500 == 0 or iteration == maximum): callback(row)
        if checkpoint_callback:
            checkpoint_callback({"state": state, "point": point, "acceleration": acceleration,
                                 "energy": energy, "history": history, "gap": gap, "iteration": iteration,
                                 "epsilon": epsilon, "total": total, "cap": cap, "maximum": maximum,
                                 "gap_tolerance": gap_tolerance, "auxiliary": auxiliary})
        if abs(gap) < gap_tolerance:
            break
    # A completed checkpoint can be recovered without taking another step.
    moments = torch.einsum("aij,ij->a", operators, state)
    effective = matrix+torch.einsum("a,aij->ij", (moments-auxiliary)/epsilon, operators)
    occupations = torch.linalg.eigvalsh((state+state.T)/2)
    physical = (float(occupations.min()) > -1e-10 and float((state.trace()-total).abs()) < 1e-9
                and (cap is None or float(occupations.max()) < cap+1e-10))
    return {"converged": physical and abs(gap) < gap_tolerance, "state": state, "moments": moments,
            "functional": energy, "gradient": (auxiliary-moments)/epsilon,
            "effective": effective, "gap": gap, "history": history, "iteration": iteration,
            "physical_state_valid": physical, "minimum_occupation": float(occupations.min()),
            "maximum_occupation": float(occupations.max()), "trace_error": float((state.trace()-total).abs())}
