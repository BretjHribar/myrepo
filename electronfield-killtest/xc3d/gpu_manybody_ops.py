"""Fermionic operator words and CUDA Hamiltonians for the new state engine."""
from itertools import combinations
import torch
from .gpu_spin_chain import norm, expect_product


def local_fermions(device):
    identity = torch.eye(4, device=device)
    parity = torch.diag(torch.tensor([1., -1., -1., 1.], device=device))
    up = torch.zeros(4, 4, device=device); up[0, 1] = 1; up[2, 3] = 1
    down = torch.zeros(4, 4, device=device); down[0, 2] = 1; down[1, 3] = -1
    return identity, parity, torch.stack((up, down, up.T, down.T))


def molecular_words(h, eri, threshold=1e-13):
    sites = len(h)
    words, hindex, gindex = [], [], []
    for i in range(sites):
        for j in range(sites):
            for spin in range(2):
                words.append(((i, spin, 1), (j, spin, 0), (-1, 0, 0), (-1, 0, 0)))
                hindex.append(i*sites+j)
    first = h.reshape(-1)[torch.tensor(hindex, device=h.device)]
    for i in range(sites):
        for j in range(sites):
            for k in range(sites):
                for l in range(sites):
                    for spin in range(2):
                        for other in range(2):
                            if spin == other and (i == k or j == l):
                                continue
                            words.append(((i, spin, 1), (k, other, 1), (l, other, 0), (j, spin, 0)))
                            gindex.append(((i*sites+j)*sites+k)*sites+l)
    second = .5*eri.reshape(-1)[torch.tensor(gindex, device=h.device)]
    coefficient = torch.cat((first, second))
    word = torch.tensor(words, dtype=torch.long, device=h.device)
    active = coefficient.abs() > threshold
    return word[active], coefficient[active]


def spin_words(sites, device):
    words = []
    for i in range(sites):
        for j in range(sites):
            words.append(((i, 0, 1), (i, 1, 0), (j, 1, 1), (j, 0, 0)))
            words.append(((i, 1, 1), (i, 0, 0), (j, 0, 1), (j, 1, 0)))
    return torch.tensor(words, device=device), torch.full((len(words),), .5, device=device)


def jw_products(words, sites):
    identity, parity, fermions = local_fermions(words.device)
    result = identity.expand(len(words), sites, 4, 4).clone()
    positions = torch.arange(sites, device=words.device)[None]
    for column in range(words.shape[1]):
        location, spin, creation = words[:, column].unbind(-1)
        before = (positions < location[:, None]) & (location[:, None] >= 0)
        on = positions == location[:, None]
        factor = torch.where(before[..., None, None], parity, identity)
        local = fermions[spin+2*creation]
        factor = torch.where(on[..., None, None], local[:, None], factor)
        result = result @ factor
    return result


def expectation(cores, words, coefficient, batch=256):
    total = coefficient.new_zeros(())
    for start in range(0, len(words), batch):
        values = expect_product(cores, jw_products(words[start:start+batch], len(cores)))
        total = total+(coefficient[start:start+batch]*values).sum()
    return total/norm(cores)


def fixed_sector(sites, up, down, device):
    # Integer occupation enumeration is metadata. Coefficients stay on CUDA.
    integers = sorted(sum(1 << (2*i) for i in a)+sum(1 << (2*i+1) for i in b)
                      for a in combinations(range(sites), up) for b in combinations(range(sites), down))
    bits = torch.tensor(integers, dtype=torch.long, device=device)
    configurations = (bits[:, None] >> (2*torch.arange(sites, device=device)[None])) & 3
    return bits, configurations


def dense_operator(words, coefficient, bits, sites, batch=256):
    """Reference-only occupation action, independent of the tensor contraction."""
    lookup = torch.full((1 << (2*sites),), -1, dtype=torch.long, device=bits.device)
    lookup[bits] = torch.arange(len(bits), device=bits.device)
    integers = torch.arange(len(lookup), device=bits.device)
    parity = torch.zeros_like(integers)
    for mode in range(2*sites):
        parity = parity ^ ((integers >> mode) & 1)
    matrix = coefficient.new_zeros(len(bits), len(bits))
    columns = torch.arange(len(bits), device=bits.device)[None]
    for start in range(0, len(words), batch):
        part = words[start:start+batch]
        values = bits[None].expand(len(part), -1)
        factor = coefficient.new_ones(values.shape)
        for slot in range(part.shape[1]-1, -1, -1):
            location, spin, creation = part[:, slot].unbind(-1)
            active = location >= 0
            mode = (2*location+spin).clamp_min(0)
            occupied = ((values >> mode[:, None]) & 1).bool()
            allowed = torch.where(creation[:, None].bool(), ~occupied, occupied) | (~active[:, None])
            lower = values & ((1 << mode[:, None])-1)
            sign = 1-2*parity[lower]
            factor = factor*allowed*torch.where(active[:, None], sign, 1)
            values = values ^ torch.where(active[:, None], 1 << mode[:, None], 0)
        rows = lookup[values]
        valid = (rows >= 0) & (factor != 0)
        contribution = factor*coefficient[start:start+len(part), None]
        flat = rows.clamp_min(0)*len(bits)+columns
        matrix.view(-1).scatter_add_(0, flat[valid], contribution[valid])
    return (matrix+matrix.T)/2
