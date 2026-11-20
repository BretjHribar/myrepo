"""Shared correlation-state candidate, replacing independent pair lowerings.

The occupied spectral subspace shares pair weight across spatial regions.
Disconnected blocks remain additive. No fitted total-electron-number gates.
This is our experimental ansatz, not a claim of N-representability.
"""
import torch
from .pair_state import PairStateXC
from .model import density_features, rho_from_dm


class CollectivePairXC(PairStateXC):
    def forward(self, dm, ao, template, eri, prepared=None):
        regions, distance = self.prepare(ao, template) if prepared is None else prepared
        n, features = density_features(rho_from_dm(dm, ao))
        mass = n @ template.weighted_partition
        pooled = torch.einsum("...gf,...g,ga->...af", features, n, template.weighted_partition)
        pooled = pooled / (mass[..., :, None] + 1e-10)
        regional_spin = torch.einsum("...sij,aji->...sa", dm, regions)
        coherence = torch.einsum("...sij,ajk,...skl,bli->...sab", dm, regions, dm, regions)
        same = ((regional_spin[..., :, :, None] * regional_spin[..., :, None, :]) - coherence).sum(-3).clamp_min(0)
        opposite = (regional_spin[..., 0, :, None] * regional_spin[..., 1, None, :] +
                    regional_spin[..., 1, :, None] * regional_spin[..., 0, None, :]).clamp_min(0)
        population = same + opposite
        exchange = -.5 * torch.einsum("...sij,...skl,ikjl->...", dm, dm, eri)
        left, right = pooled[..., :, None, :], pooled[..., None, :, :]
        shape = population.shape
        separation = distance.expand(shape)
        pair_features = torch.cat((left + right, (left - right).square(), torch.stack((
            same, opposite, torch.log1p(separation),
            mass[..., :, None].expand(shape) + mass[..., None, :].expand(shape),
            (mass[..., :, None].expand(shape) - mass[..., None, :].expand(shape)).square(),
        ), -1)), -1)
        values = self.net(pair_features)
        # Shared modes, rather than one independently lowered state per edge.
        amplitude = .4 * torch.sigmoid(values[..., 0])
        weight = torch.sqrt(population + 1e-12) - 1e-6
        coupling = amplitude * weight / (1 + separation.square()).pow(1.5)
        coupling = (coupling + coupling.transpose(-1, -2)) / 2
        diagonal_gap = .005 + 2 * torch.sigmoid(values[..., 1].diagonal(dim1=-2, dim2=-1))
        gap = torch.diag_embed(diagonal_gap)
        zero = torch.zeros_like(coupling)
        hamiltonian = torch.cat((torch.cat((zero, coupling), -1),
                                 torch.cat((coupling.transpose(-1, -2), gap), -1)), -2)
        count = mass.shape[-1]
        correlation = torch.linalg.eigvalsh(hamiltonian)[..., :count].sum(-1)
        return exchange + correlation
