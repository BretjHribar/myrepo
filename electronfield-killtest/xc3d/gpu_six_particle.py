"""Six-quasiparticle physical reference action for new accuracy tests.

Needed to assess an 18-electron peroxide in its complete minimal basis: that
case has six holes. This is a bounded reference/state-action capability, not a
speed experiment. Pair permutation identities act only on antisymmetric
alpha/beta tensors. General electron counts remain outside this backend.
"""
from itertools import combinations, permutations
import math
import torch
from .gpu_four_particle import FourParticleOperator


class SixParticleOperator(FourParticleOperator):
    def __init__(self, h, eri, electrons, nuclear=None):
        if h.device.type != "cuda" or eri.device.type != "cuda":
            raise RuntimeError("CUDA required")
        self.sites, self.electrons = len(h), electrons
        self.holes = electrons != 6
        if self.holes and 2*len(h)-electrons != 6:
            raise ValueError("This accuracy backend requires six particles or six holes")
        self.constant = h.new_zeros(()) if nuclear is None else nuclear.clone()
        if self.holes:
            density = 2*torch.eye(len(h), device=h.device, dtype=h.dtype)
            coulomb = torch.einsum("ijkl,kl->ij", eri, density)
            exchange = torch.einsum("ikjl,kl->ij", eri, density)
            self.constant += (density*(h+.5*coulomb-.25*exchange)).sum()
            self.h = -(h+coulomb-.5*exchange).T
        else:
            self.h = h
        self.pair = eri.permute(0, 2, 1, 3).contiguous().reshape(len(h)**2, len(h)**2)

    @staticmethod
    def project(value):
        first = (value-value.transpose(1, 2))/2
        first = (first-first.transpose(0, 1)-first.transpose(0, 2))/3
        second = (first-first.transpose(4, 5))/2
        return (second-second.transpose(3, 4)-second.transpose(3, 5))/3

    def one(self, value, axis):
        moved = value.movedim(axis, 0)
        return (self.h @ moved.reshape(self.sites, -1)).reshape_as(moved).movedim(0, axis)

    def two(self, value, first, second):
        order = [first, second]+[index for index in range(6) if index not in (first, second)]
        reverse = [order.index(index) for index in range(6)]
        moved = value.permute(order)
        return (self.pair @ moved.reshape(self.sites**2, -1)).reshape_as(moved).permute(reverse)

    def __call__(self, vector):
        value = vector.reshape((self.sites,)*6)
        first = self.one(value, 0)
        result = first-first.transpose(0, 1)-first.transpose(0, 2)
        first = self.one(value, 3)
        result = result+first-first.transpose(3, 4)-first.transpose(3, 5)
        first = self.two(value, 0, 1)
        result = result+first-first.transpose(1, 2)-first.transpose(0, 2)
        first = self.two(value, 3, 4)
        result = result+first-first.transpose(4, 5)-first.transpose(3, 5)
        first = self.two(value, 0, 3)
        first = first-first.transpose(0, 1)-first.transpose(0, 2)
        result = result+first-first.transpose(3, 4)-first.transpose(3, 5)+self.constant*value
        return self.project(result).flatten()

    def naive(self, vector):
        value = vector.reshape((self.sites,)*6)
        result = self.constant*value
        for axis in range(6):
            result = result+self.one(value, axis)
        for first, second in combinations(range(6), 2):
            result = result+self.two(value, first, second)
        return self.project(result).flatten()


class SixParticleFockAction:
    def __init__(self, operator, bits):
        self.operator, self.bits = operator, bits
        n = operator.sites
        quasi = (((1 << (2*n))-1) ^ bits) if operator.holes else bits
        sites = torch.arange(n, device=bits.device)
        up = ((quasi[:, None] >> (2*sites[None])) & 1).bool()
        down = ((quasi[:, None] >> (2*sites[None]+1)) & 1).bool()
        if not bool((up.sum(-1) == 3).all() & (down.sum(-1) == 3).all()):
            raise ValueError("Fock sector is not three-up/three-down quasiparticles")
        up = sites[None].expand(len(bits), -1)[up].reshape(len(bits), 3)
        down = sites[None].expand(len(bits), -1)[down].reshape(len(bits), 3)
        inversions = (up[:, :, None] > down[:, None]).sum((-1, -2))
        if operator.holes:
            inversions += (2*up).sum(-1)+(2*down+1).sum(-1)
        self.sign = 1-2*inversions.remainder(2)
        orders = list(permutations(range(3)))
        order_sign = [1-2*(sum(order[i] > order[j] for i in range(3) for j in range(i+1, 3)) % 2) for order in orders]
        indices, signs = [], []
        for first, sign_a in zip(orders, order_sign):
            for second, sign_b in zip(orders, order_sign):
                columns = [up[:, index] for index in first]+[down[:, index] for index in second]
                indices.append(sum(value*(n**(5-index)) for index, value in enumerate(columns)))
                signs.append(sign_a*sign_b)
        self.indices = torch.stack(indices)
        self.permutation_sign = torch.tensor(signs, device=bits.device)[:, None]

    def to_tensor(self, vector):
        result = vector.new_zeros(self.operator.sites**6)
        result[self.indices.flatten()] = (self.permutation_sign*(vector*self.sign/6)[None]).flatten()
        return result

    def to_fock(self, vector):
        return 6*self.sign*vector[self.indices[0]]

    def __call__(self, vector):
        return self.to_fock(self.operator(self.to_tensor(vector)))
