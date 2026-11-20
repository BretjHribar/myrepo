"""Locally purified, exact-number/total-spin physical states on CUDA.

Purification, CG coupling and tensor products are established primitives.
This state removes a global mixture-count restriction from our proposed
density-response construction; it is not a chemical-accuracy claim.
"""
import torch
from torch import nn
from .gpu_spin_chain import SpinNumberState


class PurifiedSpinState(SpinNumberState):
    def __init__(self, sites, electrons, twice_spin=0, channels=2, ancilla=2, device="cuda"):
        if not isinstance(ancilla, int) or ancilla < 1:
            raise ValueError("The local spinless ancilla dimension must be positive")
        super().__init__(sites, electrons, twice_spin, channels, device, local_multiplicity=ancilla)
        self.ancilla = ancilla
        values = []
        for original in self.reduced:
            parameter = .025*torch.randn(len(original), ancilla, device=original.device, dtype=original.dtype)
            parameter[:, 0] = original.detach()
            values.append(nn.Parameter(parameter))
        self.reduced = nn.ParameterList(values)

    def cores(self, reference=False):
        if reference:
            original = super().cores(reference=True)
            result = []
            for value in original:
                core = value.new_zeros(value.shape[0], 4, self.ancilla, value.shape[-1])
                core[:, :, 0, :] = value
                result.append(core)
            return result
        result = []
        for site, (values, names) in enumerate(zip(self.reduced, self._maps)):
            index, parameter, coefficient = [getattr(self, name) for name in names]
            flat = values.new_zeros(self.bond_dimensions[site]*4*self.bond_dimensions[site+1], self.ancilla)
            flat = flat.index_add(0, index, values[parameter]*coefficient[:, None])
            core = flat.reshape(self.bond_dimensions[site], 4, self.bond_dimensions[site+1], self.ancilla)
            core = core.permute(0, 1, 3, 2)
            result.append(core/core.norm().detach().clamp_min(1e-30))
        return result


def purified_norm(cores):
    environment = cores[0].new_ones(1, 1)
    for core in cores:
        environment = torch.einsum("ij,idar,jdas->rs", environment, core.conj(), core)
    return environment.squeeze().real


def purified_products(cores, operators):
    """Unnormalized batched products, tracing only local spinless ancillas."""
    environment = cores[0].new_ones(len(operators), 1, 1)
    for site, core in enumerate(cores):
        temporary = torch.einsum("bij,jcas->bicas", environment, core)
        temporary = torch.einsum("bdc,bicas->bidas", operators[:, site], temporary)
        environment = torch.einsum("idar,bidas->brs", core.conj(), temporary)
    return environment[:, 0, 0].real


def purification_amplitudes(cores, configurations, ancillas):
    """Diagnostic selected coefficients only; production never enumerates them."""
    if len(configurations) != len(ancillas):
        raise ValueError("Physical and ancilla configuration batches differ")
    state = cores[0].new_ones(len(configurations), 1)
    for site, core in enumerate(cores):
        selected = core[:, configurations[:, site], ancillas[:, site], :].permute(1, 0, 2)
        state = torch.einsum("bi,bij->bj", state, selected)
    return state[:, 0]


@torch.no_grad()
def compose_purifications(fragments):
    """Exact independent singlet product in the same local state family."""
    if not fragments or any(value.twice_spin != 0 for value in fragments):
        raise ValueError("Supplied independent fragments must be singlets")
    device = next(fragments[0].parameters()).device
    ancilla = fragments[0].ancilla
    if any(value.ancilla != ancilla or next(value.parameters()).device != device for value in fragments):
        raise ValueError("Fragments must share the local ancilla dimension and CUDA device")
    result = PurifiedSpinState(sum(value.sites for value in fragments),
                               sum(value.electrons for value in fragments),
                               channels=max(value.channels for value in fragments),
                               ancilla=ancilla, device=device)
    for value in result.reduced:
        value.zero_()
    start, offset = 0, 0
    for fragment in fragments:
        for site, edges in enumerate(fragment.edge_labels):
            lookup = {edge: index for index, edge in enumerate(result.edge_labels[start+site])}
            indices = []
            for number, spin, left, physical, next_number, next_spin, right in edges:
                target = (number+offset, spin, left, physical, next_number+offset, next_spin, right)
                if target not in lookup:
                    raise RuntimeError("The combined quantum sectors cannot retain this fragment product")
                indices.append(lookup[target])
            result.reduced[start+site][torch.tensor(indices, device=device)] = fragment.reduced[site]
        start += fragment.sites
        offset += fragment.electrons
    return result
