"""Our Coulomb-response coherent-orbital state, without dense occupation space.

Uses a variational superposition of spin-paired determinants. Nonorthogonal
determinants, cofactors and Ritz minimization are established primitives; the
research hypothesis is their density/Coulomb-response-conditioned state family
for a constrained correlation functional. No frontier method is reproduced.
Polynomial cofactors retain matrix elements even for orthogonal determinants;
an overlap-inverse regularization would incorrectly lose those couplings.

Accuracy status: the fixed-global determinant family fails fragment closure
and the N=6/10 rejection panel. Retain these kernels as local-state primitives;
do not promote or speed-tune that rejected global model.
"""
from itertools import combinations
import torch
from torch import nn


def determinant(value):
    n = value.shape[-1]
    if n == 0:
        return value.new_ones(value.shape[:-2])
    if n == 1:
        return value[..., 0, 0]
    if n == 2:
        return value[..., 0, 0]*value[..., 1, 1]-value[..., 0, 1]*value[..., 1, 0]
    if n == 3:
        return (value[..., 0, 0]*(value[..., 1, 1]*value[..., 2, 2]-value[..., 1, 2]*value[..., 2, 1])
                -value[..., 0, 1]*(value[..., 1, 0]*value[..., 2, 2]-value[..., 1, 2]*value[..., 2, 0])
                +value[..., 0, 2]*(value[..., 1, 0]*value[..., 2, 1]-value[..., 1, 1]*value[..., 2, 0]))
    return torch.linalg.det(value)


class CofactorKernel:
    def __init__(self, h, eri, pairs):
        if h.device.type != "cuda" or eri.device.type != "cuda":
            raise RuntimeError("CUDA is required")
        self.h, self.pairs = h, pairs
        size = len(h)
        with torch.no_grad():
            values, vectors = torch.linalg.eigh(eri.reshape(size*size, size*size))
            if float(values.min()) < -1e-9:
                raise RuntimeError("Coulomb matrix is not positive semidefinite")
            chosen = values > 1e-13
            self.factors = (vectors[:, chosen]*values[chosen].sqrt()).T.reshape(-1, size, size)
            self.factorization_error = float((self.factors.flatten(1).T @ self.factors.flatten(1)-eri.reshape(size*size, size*size)).abs().max())
        self.tables = []
        for rank in (1, 2):
            groups = list(combinations(range(pairs), rank))
            indices, signs = [], []
            for row in groups:
                for column in groups:
                    kept_rows = [i for i in range(pairs) if i not in row]
                    kept_columns = [i for i in range(pairs) if i not in column]
                    indices.append([i*pairs+j for i in kept_rows for j in kept_columns])
                    signs.append(1-2*((sum(row)+sum(column)) % 2))
            index = torch.tensor(indices, device=h.device, dtype=torch.long).reshape(len(groups)**2, max(0, pairs-rank)**2)
            sign = torch.tensor(signs, device=h.device)
            group = torch.tensor(groups, device=h.device, dtype=torch.long).reshape(-1, rank)
            self.tables.append((index, sign, group))

    def cofactor(self, overlaps, rank):
        index, sign, groups = self.tables[rank-1]
        count, order = len(groups), self.pairs-rank
        if count == 0:
            return overlaps.new_zeros(*overlaps.shape[:-2], 0, 0)
        minor = overlaps.flatten(-2)[..., index].reshape(*overlaps.shape[:-2], count*count, order, order)
        return (determinant(minor)*sign).reshape(*overlaps.shape[:-2], count, count)

    def matrices(self, orbitals):
        overlaps = torch.einsum("ria,sib->rsab", orbitals, orbitals)
        spin_overlap = determinant(overlaps)
        first = self.cofactor(overlaps, 1)
        transformed_h = torch.einsum("ria,ij,sjb->rsab", orbitals, self.h, orbitals)
        onebody = 2*spin_overlap*(first*transformed_h).sum((-1, -2))
        transformed_l = torch.einsum("ria,lij,sjb->rslab", orbitals, self.factors, orbitals)
        first_l = (transformed_l*first[:, :, None]).sum((-1, -2))
        opposite = first_l.square().sum(-1)
        second = self.cofactor(overlaps, 2)
        if self.pairs > 1:
            flat = transformed_l.flatten(-2)
            integrals = (flat.transpose(-1, -2) @ flat).reshape(*spin_overlap.shape, *((self.pairs,)*4))
            _, _, group = self.tables[1]
            a, c = group[:, 0], group[:, 1]
            b, d = a, c
            pair_integrals = integrals[..., a[:, None], b[None], c[:, None], d[None]]
            pair_integrals = pair_integrals-integrals[..., a[:, None], d[None], c[:, None], b[None]]
            same = 2*spin_overlap*(second*pair_integrals).sum((-1, -2))
        else:
            same = torch.zeros_like(spin_overlap)
        metric = spin_overlap.square()
        matrix = onebody+opposite+same
        return (matrix+matrix.T)/2, (metric+metric.T)/2


@torch.no_grad()
def ritz_coefficients(matrix, metric, threshold=1e-10, fast_path=True):
    # Select independent *physical* determinants; do not add an artificial
    # norm regularizer that would alter the variational Rayleigh quotient.
    chosen = None
    if fast_path:
        complete, info = torch.linalg.cholesky_ex(metric)
        if bool((info == 0) & (complete.diagonal().square().min() > threshold)):
            chosen = list(range(len(metric)))
    if chosen is None:
        chosen = [0]
        for index in range(1, len(metric)):
            block = metric[chosen][:, chosen]
            connection = metric[chosen, index]
            remainder = metric[index, index]-connection @ torch.linalg.solve(block, connection)
            if float(remainder) > threshold:
                chosen.append(index)
    index = torch.tensor(chosen, device=matrix.device)
    overlap = metric[index][:, index]
    hamiltonian = matrix[index][:, index]
    lower = torch.linalg.cholesky(overlap)
    transformed = torch.linalg.solve_triangular(lower, hamiltonian, upper=False)
    transformed = torch.linalg.solve_triangular(lower, transformed.T, upper=False).T
    _, vectors = torch.linalg.eigh((transformed+transformed.T)/2)
    coefficients = torch.linalg.solve_triangular(lower.T, vectors[:, :1], upper=True)[:, 0]
    full = matrix.new_zeros(len(metric)); full[index] = coefficients
    return full, chosen


class CoherentOrbitals(nn.Module):
    def __init__(self, h, eri, pairs, proposed_modes=6, displacement=.15, representation="rotation", fast_coefficients=True):
        super().__init__()
        self.pairs, self.sites = pairs, len(h)
        self.representation, self.fast_coefficients = representation, fast_coefficients
        self.kernel = CofactorKernel(h, eri, pairs)
        with torch.no_grad():
            density = h.new_zeros(len(h), len(h)); density[:pairs, :pairs] = 2*torch.eye(pairs, device=h.device)
            fock = h+torch.einsum("ijkl,kl->ij", eri, density)-.5*torch.einsum("ikjl,kl->ij", eri, density)
            energies = fock.diagonal()
            gap = energies[pairs:, None]-energies[None, :pairs]
            if float(gap.min()) <= 1e-6:
                raise RuntimeError("Response seed requires a gapped input determinant")
            coupling = eri[pairs:, :pairs, pairs:, :pairs].reshape(gap.numel(), gap.numel())
            inverse = gap.rsqrt().flatten()
            response = inverse[:, None]*coupling*inverse[None]
            values, vectors = torch.linalg.eigh((response+response.T)/2)
            boundary = values[-min(proposed_modes, len(values))]
            selected = values >= boundary-max(1e-10, float(values[-1])*1e-7)
            modes = vectors[:, selected].T.reshape(-1, len(h)-pairs, pairs)
            self.response_modes = len(modes)
            seed = torch.stack((displacement*modes, -displacement*modes), 1).flatten(0, 1)
        self.angles = nn.Parameter(seed)
        if representation == "thouless":
            with torch.no_grad():
                self.representation = "rotation"
                initial_orbitals = self.orbitals()[1:]
                self.angles.copy_(self.coordinates(initial_orbitals))
                self.representation = "thouless"

    def coordinates(self, orbitals):
        occupied = orbitals[:, :self.pairs]
        virtual = orbitals[:, self.pairs:]
        return torch.linalg.solve(occupied.transpose(-1, -2), virtual.transpose(-1, -2)).transpose(-1, -2)

    def orbitals(self):
        count = len(self.angles)
        if self.representation == "rotation":
            generator = self.angles.new_zeros(count, self.sites, self.sites)
            generator[:, self.pairs:, :self.pairs] = self.angles
            generator[:, :self.pairs, self.pairs:] = -self.angles.transpose(-1, -2)
            moving = torch.matrix_exp(generator)[:, :, :self.pairs]
        elif self.representation == "thouless":
            identity = torch.eye(self.pairs, device=self.angles.device, dtype=self.angles.dtype)
            raw = torch.cat((identity.expand(count, -1, -1), self.angles), 1)
            gram = identity+self.angles.transpose(-1, -2) @ self.angles
            if self.pairs == 2:
                # Exact small Cholesky normalization: no full-basis exponential
                # or eigendecomposition, with the same occupied subspace.
                first = gram[:, 0, 0].sqrt()
                cross = gram[:, 1, 0]/first
                second = (gram[:, 1, 1]-cross.square()).sqrt()
                column0 = raw[:, :, 0]/first[:, None]
                column1 = (raw[:, :, 1]-column0*cross[:, None])/second[:, None]
                moving = torch.stack((column0, column1), -1)
            else:
                lower = torch.linalg.cholesky(gram)
                moving = torch.linalg.solve_triangular(lower, raw.transpose(-1, -2), upper=False).transpose(-1, -2)
        else:
            raise ValueError("Unknown orbital representation")
        reference = torch.eye(self.sites, device=moving.device, dtype=moving.dtype)[:, :self.pairs]
        return torch.cat((reference[None], moving))

    def forward(self):
        orbitals = self.orbitals()
        matrix, metric = self.kernel.matrices(orbitals)
        coefficients, chosen = ritz_coefficients(matrix, metric, fast_path=self.fast_coefficients)
        energy = coefficients @ matrix @ coefficients/(coefficients @ metric @ coefficients)
        return energy, coefficients, orbitals, chosen

    @staticmethod
    def four_electron_vector(orbitals, coefficients):
        if orbitals.shape[-1] != 2:
            raise ValueError("This diagnostic coordinate tensor is specific to four electrons")
        first = orbitals[:, :, 0, None]*orbitals[:, None, :, 1]
        pair = (first-first.transpose(-1, -2))/(2**.5)
        vector = torch.einsum("a,aij,akl->ijkl", coefficients, pair, pair).flatten()
        return vector/vector.norm()
