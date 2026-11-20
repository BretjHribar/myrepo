"""Our second research candidate: Pauli-protected collective pair energies.

An analytic exchange term removes one-electron Hartree self-interaction.
Regional pair populations exclude same-spin self-pairs by density-matrix
coherence. A learned two-state energy saturates at small effective gaps.
The present exact-exchange implementation uses AO integrals: a mechanism
prototype whose large-system speed target is explicitly not established.
"""
import torch
from torch import nn
from .model import density_features, rho_from_dm


class PairStateXC(nn.Module):
    def __init__(self, width=32):
        super().__init__()
        self.width = width
        # Symmetric regional descriptors, pair population and separation.
        self.net = nn.Sequential(nn.Linear(15, width), nn.Tanh(),
                                 nn.Linear(width, width), nn.Tanh(), nn.Linear(width, 2))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def prepare(self, ao, template):
        regional_overlap = torch.einsum("ga,gi,gj->aij", template.weighted_partition, ao[0], ao[0])
        pair_distance = torch.cdist(template.landmarks, template.landmarks)
        return regional_overlap, pair_distance

    def forward(self, dm, ao, template, eri, prepared=None):
        regions, distance = self.prepare(ao, template) if prepared is None else prepared
        rho = rho_from_dm(dm, ao)
        n, features = density_features(rho)
        mass = n @ template.weighted_partition
        pooled = torch.einsum("...gf,...g,ga->...af", features, n, template.weighted_partition)
        pooled = pooled / (mass[..., :, None] + 1e-10)
        regional_spin = torch.einsum("...sij,aji->...sa", dm, regions)
        coherence = torch.einsum("...sij,ajk,...skl,bli->...sab", dm, regions, dm, regions)
        same_spin = ((regional_spin[..., :, :, None] * regional_spin[..., :, None, :]) - coherence).sum(-3).clamp_min(0)
        opposite_spin = (regional_spin[..., 0, :, None] * regional_spin[..., 1, None, :] +
                         regional_spin[..., 1, :, None] * regional_spin[..., 0, None, :]).clamp_min(0)
        total_pair = same_spin + opposite_spin
        # Exchange is exact for this finite AO basis, with no N=1 special case.
        exchange = -.5 * torch.einsum("...sij,...skl,ikjl->...", dm, dm, eri)
        first, second = pooled[..., :, None, :], pooled[..., None, :, :]
        combined = first + second
        difference = (first - second).square()
        shape = total_pair.shape
        geometric = distance.expand(shape)
        pair_features = torch.cat((combined, difference,
            torch.stack((same_spin, opposite_spin, torch.log1p(geometric),
                         mass[..., :, None].expand(shape) + mass[..., None, :].expand(shape),
                         (mass[..., :, None].expand(shape) - mass[..., None, :].expand(shape)).square()), -1)), -1)
        values = self.net(pair_features)
        # Bounded coupling and finite gap avoid perturbative divergence.
        coupling = .4 * torch.sigmoid(values[..., 0]) / (1 + geometric.square()).pow(1.5)
        gap = .005 + 2 * torch.sigmoid(values[..., 1])
        lowering = 2 * coupling.square() / (torch.sqrt(gap.square() + 4 * coupling.square()) + gap)
        correlation = -.5 * (total_pair * lowering).sum((-1, -2))
        return exchange + correlation
