"""Callable frozen-frame XC research model for even-N unpolarized densities.

Only T, W, the spatial moment frame and the input D enter the scalar/potential
evaluation. Nuclear external charges/energies are not used in this search.
It currently uses dense small-basis physical solves, not learned inference.
"""
import math
import torch
from .gpu_canonical_density import canonical_lift
from .gpu_ensemble_dual import ensemble_completion, ensemble_dual


@torch.no_grad()
def evaluate_xc(data, density, entropy_tolerance=1e-5, maximum=192, stationarity_tolerance=1e-7):
    if density.device.type != "cuda":
        raise RuntimeError("The model requires CUDA; CPU fallback is disabled")
    electrons, sites = data["case"]["electrons"], len(density)
    if electrons % 2 or data["case"]["spin"]:
        raise ValueError("The current functional interface supports even-N spin-unpolarized densities only")
    occupations, orbitals = torch.linalg.eigh((density+density.T)/2)
    if float(occupations.min()) < -1e-9 or float(occupations.max()) > 2+1e-9 or float((density.trace()-electrons).abs()) > 1e-8:
        raise ValueError("Input density is not a valid fixed-N spin-unpolarized 1RDM")
    lift = canonical_lift(orbitals, occupations, electrons, data["bits"])
    if lift["occupation_residual"] > 1e-8:
        raise RuntimeError("Input canonical ensemble lift did not reproduce the occupations")
    target = torch.einsum("ij,aij->a", density, data["moments"])
    beta = max(math.log(len(data["universal"])), 2*sites*math.log(2))/entropy_tolerance
    interacting = ensemble_completion(data["universal"], data["q_matrices"], target, lift["vectors"].T,
                                        beta, maximum=maximum, tolerance=stationarity_tolerance)
    operators = torch.cat((torch.eye(sites, device=density.device)[None], data["moments"]))
    targets = torch.cat((target.new_tensor([electrons]), target))
    spectrum = torch.linalg.eigvalsh(data["kinetic"])
    initial = torch.zeros_like(targets)
    initial[0] = -(spectrum[electrons//2-1]+spectrum[electrons//2])/2
    kinetic = ensemble_dual(data["kinetic"], operators, targets, beta, statistics="onebody", initial=initial)
    if interacting["density_residual"] > 1e-8 or kinetic["residual"] > 1e-8:
        raise RuntimeError(f"Density constraints failed: interacting={interacting['density_residual']}, kinetic={kinetic['residual']}")
    if interacting["stationarity_residual"] >= stationarity_tolerance:
        raise RuntimeError(f"XC derivative unavailable: state stationarity residual {interacting['stationarity_residual']}")
    hartree_matrix = torch.einsum("ijkl,kl->ij", data["eri"], density)
    hartree = .5*(density*hartree_matrix).sum()
    xc = interacting["functional"]-kinetic["dual"]-hartree
    potential = torch.einsum("a,aij->ij", -interacting["potential"]+kinetic["potential"][1:], data["moments"])-hartree_matrix
    return {"xc_energy": xc, "xc_potential": potential, "target": target,
            "interacting": interacting, "kinetic": kinetic,
            "canonical_lift_components": len(lift["weights"]), "beta": beta}
