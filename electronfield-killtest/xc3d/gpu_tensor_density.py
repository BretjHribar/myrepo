"""Spatial density matrices of physical tensor states, entirely on CUDA.

A first derivative of a literal one-body operator gives its symmetric real
transition density. This supplies density coordinates without full Fock links.
It is a fixed-state derivative, not a nuclear derivative or converged envelope.
"""
import torch
from .gpu_fermion_mpo import FermionMPO
from .gpu_purified_spin import purified_norm


def one_body_mpo(matrix):
    sites = len(matrix)
    labels = torch.cartesian_prod(torch.arange(sites, device=matrix.device),
                                  torch.arange(sites, device=matrix.device), torch.arange(2, device=matrix.device))
    words = torch.zeros(len(labels), 4, 3, dtype=torch.long, device=matrix.device)
    words[:, :, 0] = -1
    words[:, 0, 0], words[:, 1, 0] = labels[:, 0], labels[:, 1]
    words[:, 0, 1] = words[:, 1, 1] = labels[:, 2]
    words[:, 0, 2] = 1
    coefficient = matrix[labels[:, 0], labels[:, 1]]
    return FermionMPO(words, coefficient, sites)


def symmetric_transition_density(bra, ket):
    if (len(bra.cores), bra.electrons, bra.twice_spin) != (len(ket.cores), ket.electrons, ket.twice_spin):
        raise ValueError("Transition densities require matching physical sites and number/spin sectors")
    sites, device = len(bra.cores), bra.cores[0].device
    if device.type != "cuda": raise RuntimeError("Tensor density work requires CUDA")
    with torch.enable_grad():
        # All entries are nonzero so the operator graph contains every
        # derivative direction. Only the coefficient field is differentiated.
        field = torch.ones(sites, sites, device=device, requires_grad=True)
        operator = one_body_mpo((field+field.T)/2)
        value = operator.contract(ket.cores, bra=bra.cores)
        density, = torch.autograd.grad(value, field)
    return density


def density_matrix(state):
    density = symmetric_transition_density(state, state)
    return density/purified_norm(state.cores).detach()
