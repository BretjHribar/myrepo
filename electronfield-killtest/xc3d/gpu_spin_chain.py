"""Number- and total-spin-resolved tensor representation for our research state.

This is a numerical representation, not a claimed new tensor-network method.
Its purpose is to remove the N=2/3 limitation of our density-correlation models.
Reduced tensors carry (particle number, twice spin) labels; fixed Clebsch-Gordan
maps supply magnetic indices. All coefficients and contractions use CUDA.
"""
from collections import defaultdict
import torch
from torch import nn


def transitions(number, spin):
    yield 0, number, spin
    yield 2, number+2, spin
    yield 1, number+1, spin+1
    if spin:
        yield 1, number+1, spin-1


def quantum_bonds(sites, electrons, twice_spin, channels, local_multiplicity=1):
    # These integer tables describe tensor shapes/quantum labels only.
    forward = [{(0, 0): 1}]
    for _ in range(sites):
        table = defaultdict(int)
        for (number, spin), count in forward[-1].items():
            for _, new_number, new_spin in transitions(number, spin):
                if new_number <= electrons:
                    table[new_number, new_spin] += count*local_multiplicity
        forward.append(dict(table))
    if (electrons, twice_spin) not in forward[-1]:
        raise ValueError("The requested particle/spin sector is empty")
    backward = [None]*(sites+1)
    backward[-1] = {(electrons, twice_spin): 1}
    for site in range(sites-1, -1, -1):
        backward[site] = {}
        for number, spin in forward[site]:
            count = sum(backward[site+1].get((new_number, new_spin), 0)
                        for _, new_number, new_spin in transitions(number, spin))
            if count:
                backward[site][number, spin] = count*local_multiplicity
    return [{q: min(channels, count, backward[site][q]) for q, count in forward[site].items()
             if q in backward[site]} for site in range(sites+1)]


class SpinNumberState(nn.Module):
    def __init__(self, sites, electrons, twice_spin=0, channels=2, device="cuda", local_multiplicity=1):
        super().__init__()
        if torch.device(device).type != "cuda":
            raise RuntimeError("GPU-only numerical work")
        self.sites, self.electrons, self.twice_spin = sites, electrons, twice_spin
        self.channels = channels
        self.quantum_labels = quantum_bonds(sites, electrons, twice_spin, channels, local_multiplicity)
        basis = []
        for site, groups in enumerate(self.quantum_labels):
            labels = []
            for (number, spin), count in sorted(groups.items()):
                for channel in range(count):
                    magnetic = [twice_spin] if site == sites else range(-spin, spin+1, 2)
                    for m in magnetic:
                        labels.append((number, spin, channel, m))
            basis.append(labels)
        self.bond_dimensions = [len(labels) for labels in basis]
        self.bond_basis = basis
        paired = (electrons-twice_spin)//2
        reference = [2]*paired+[1]*twice_spin+[0]*(sites-paired-twice_spin)
        if len(reference) != sites:
            raise ValueError("Requested reference does not fit the orbital sites")
        self.reduced = nn.ParameterList()
        self._maps = []
        self.edge_labels = []
        ref_n, ref_s = 0, 0
        for site in range(sites):
            left, right = basis[site], basis[site+1]
            right_lookup = {label: i for i, label in enumerate(right)}
            edges, flat, parameter, numerators, denominators, signs = [], [], [], [], [], []
            for (number, spin), count in sorted(self.quantum_labels[site].items()):
                for physical, new_number, new_spin in transitions(number, spin):
                    right_count = self.quantum_labels[site+1].get((new_number, new_spin), 0)
                    for lc in range(count):
                        for rc in range(right_count):
                            edge = len(edges)
                            edges.append((number, spin, lc, physical, new_number, new_spin, rc))
                            for li, (_, js, ch, m) in enumerate(left):
                                if left[li][:3] != (number, spin, lc):
                                    continue
                                if physical in (0, 2):
                                    target = (new_number, new_spin, rc, m)
                                    if target in right_lookup:
                                        ri = right_lookup[target]
                                        flat.append((li*4+(0 if physical == 0 else 3))*len(right)+ri)
                                        parameter.append(edge); numerators.append(1); denominators.append(1); signs.append(1)
                                else:
                                    for local, magnetic in ((1, 1), (2, -1)):
                                        target = (new_number, new_spin, rc, m+magnetic)
                                        if target not in right_lookup:
                                            continue
                                        if new_spin == spin+1:
                                            numerator = spin+m+2 if magnetic == 1 else spin-m+2
                                            sign = 1
                                        else:
                                            numerator = spin-m if magnetic == 1 else spin+m
                                            sign = -1 if magnetic == 1 else 1
                                        if numerator == 0:
                                            continue
                                        ri = right_lookup[target]
                                        flat.append((li*4+local)*len(right)+ri)
                                        parameter.append(edge); numerators.append(numerator)
                                        denominators.append(2*(spin+1)); signs.append(sign)
            maps = []
            for suffix, data in (("index", flat), ("parameter", parameter)):
                name = f"_{suffix}_{site}"
                self.register_buffer(name, torch.tensor(data, dtype=torch.long, device=device))
                maps.append(name)
            coefficient = (torch.tensor(numerators, device=device)/torch.tensor(denominators, device=device)).sqrt()
            coefficient = coefficient*torch.tensor(signs, device=device)
            name = f"_coefficient_{site}"
            self.register_buffer(name, coefficient)
            maps.append(name)
            self._maps.append(maps)
            self.edge_labels.append(edges)
            tensor = .025*torch.randn(len(edges), device=device)
            ref_physical = reference[site]
            next_n = ref_n+ref_physical
            next_s = ref_s+(1 if ref_physical == 1 else 0)
            ref_edge = edges.index((ref_n, ref_s, 0, ref_physical, next_n, next_s, 0))
            tensor[ref_edge] = 1.
            self.reduced.append(nn.Parameter(tensor))
            ref_n, ref_s = next_n, next_s
        self.reference_occupations = [3 if value == 2 else 1 if value == 1 else 0 for value in reference]

    def cores(self, reference=False):
        result = []
        ref_n, ref_s = 0, 0
        for site, (values, names) in enumerate(zip(self.reduced, self._maps)):
            index, parameter, coefficient = [getattr(self, name) for name in names]
            if reference:
                # A pure number/spin path is also directly available as a product
                # state; no nonlinear optimizer must reconstruct the reference.
                core = values.new_zeros(self.bond_dimensions[site], 4, self.bond_dimensions[site+1])
                l = (ref_n, ref_s, 0, ref_s)
                occupation = self.reference_occupations[site]
                ref_n += 2 if occupation == 3 else 1 if occupation == 1 else 0
                ref_s += 1 if occupation == 1 else 0
                r = (ref_n, ref_s, 0, ref_s)
                def bond_index(location, target):
                    labels = []
                    for (n, s), count in sorted(self.quantum_labels[location].items()):
                        for channel in range(count):
                            for m in ([self.twice_spin] if location == self.sites else range(-s, s+1, 2)):
                                labels.append((n, s, channel, m))
                    return labels.index(target)
                core[bond_index(site, l), occupation, bond_index(site+1, r)] = 1.
            else:
                flat = values.new_zeros(self.bond_dimensions[site]*4*self.bond_dimensions[site+1])
                core = flat.scatter_add(0, index, values[parameter]*coefficient).reshape(self.bond_dimensions[site], 4, self.bond_dimensions[site+1])
                core = core/core.norm().detach().clamp_min(1e-20)
            result.append(core)
        return result


def norm(cores):
    environment = cores[0].new_ones(1, 1)
    for core in cores:
        environment = torch.einsum("ij,idk,jdl->kl", environment, core.conj(), core)
    return environment.squeeze().real


def expect_product(cores, operators):
    """Batched tensor-product operators, shape (batch,site,4,4)."""
    environment = cores[0].new_ones(len(operators), 1, 1)
    for site, core in enumerate(cores):
        temporary = torch.einsum("bij,jcr->bicr", environment, core)
        temporary = torch.einsum("bdc,bicr->bidr", operators[:, site], temporary)
        environment = torch.einsum("idk,bidr->bkr", core.conj(), temporary)
    return environment[:, 0, 0].real


def amplitudes(cores, configurations):
    """Diagnostic amplitudes on a supplied occupation list, without full 4^L storage."""
    state = cores[0].new_ones(len(configurations), 1)
    for site, core in enumerate(cores):
        selected = core[:, configurations[:, site], :].permute(1, 0, 2)
        state = (state[:, None] @ selected).squeeze(1)
    return state[:, 0]


@torch.no_grad()
def compose_singlets(fragments):
    """Embed a product in the same state family, with exact number offsets.

    Requires explicitly supplied, contiguous, independent singlet components.
    This supplies a composition operation, not automatic fragment discovery or
    a guarantee that an unrestricted finite-budget global solve is additive.
    """
    if not fragments or any(fragment.twice_spin != 0 for fragment in fragments):
        raise ValueError("This composition operation currently requires singlet fragments")
    device = next(fragments[0].parameters()).device
    if any(next(fragment.parameters()).device != device for fragment in fragments):
        raise ValueError("All fragment states must be on the same CUDA device")
    channels = max(fragment.channels for fragment in fragments)
    result = SpinNumberState(sum(fragment.sites for fragment in fragments),
                             sum(fragment.electrons for fragment in fragments),
                             channels=channels, device=device)
    for parameter in result.reduced:
        parameter.zero_()
    start, offset = 0, 0
    for fragment in fragments:
        for site, edges in enumerate(fragment.edge_labels):
            lookup = {edge: i for i, edge in enumerate(result.edge_labels[start+site])}
            indices = []
            for number, spin, left, physical, next_number, next_spin, right in edges:
                mapped = (number+offset, spin, left, physical, next_number+offset, next_spin, right)
                if mapped not in lookup:
                    raise RuntimeError("Combined state family cannot contain this fragment product")
                indices.append(lookup[mapped])
            index = torch.tensor(indices, device=device, dtype=torch.long)
            result.reduced[start+site][index] = fragment.reduced[site]
        start += fragment.sites
        offset += fragment.electrons
    return result
