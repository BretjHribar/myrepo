"""Our capacity-constrained correlation-hole transport experiment.

Moving pair weight off a region's diagonal adds it to symmetric off-diagonal
entries. Every row sum is conserved, and total outflow cannot exceed available
on-site pair weight. Orbital coherence selects static-correlation channels;
dispersion uses a separate damped, decaying term. These constraints do not prove
full many-electron N-representability. AO exchange is still explicit here.
"""
import torch
from torch import nn
from .pair_state import PairStateXC
from .model import density_features, rho_from_dm


class ConservedHoleXC(PairStateXC):
    def __init__(self, width=32):
        super().__init__(width)
        self.dynamic = nn.Sequential(nn.Linear(5, width), nn.Tanh(), nn.Linear(width, 1))
        nn.init.zeros_(self.dynamic[-1].weight)
        nn.init.constant_(self.dynamic[-1].bias, -2.)
        self.self_coulomb_scale = nn.Parameter(torch.tensor(2.17))

    def prepare(self, ao, template):
        regions, distance = super().prepare(ao, template)
        mids = (template.landmarks[:, None, :] + template.landmarks[None, :, :]) / 2
        kernel = torch.exp(-torch.cdist(template.coords, mids.flatten(0, 1)).square() / .5)
        kernel = kernel * template.weights[:, None]
        kernel = kernel / kernel.sum(0).clamp_min(1e-10)
        self_weights = template.partition.pow(4 / 3) * template.weights[:, None]
        return regions, distance, kernel, self_weights

    def forward(self, dm, ao, template, eri, prepared=None):
        regions, distance, bridge, self_weights = self.prepare(ao, template) if prepared is None else prepared
        n, features = density_features(rho_from_dm(dm, ao))
        n_positive = n.clamp_min(0)
        mass = (n_positive @ template.weighted_partition).clamp_min(1e-10)
        pooled = torch.einsum("...gf,...g,ga->...af", features, n_positive, template.weighted_partition)
        pooled = pooled / mass[..., :, None]
        regional_spin = torch.einsum("...sij,aji->...sa", dm, regions)
        coherence = torch.einsum("...sij,ajk,...skl,bli->...sab", dm, regions, dm, regions)
        same = ((regional_spin[..., :, :, None] * regional_spin[..., :, None, :]) - coherence).sum(-3).clamp_min(0)
        opposite = (regional_spin[..., 0, :, None] * regional_spin[..., 1, None, :] +
                    regional_spin[..., 1, :, None] * regional_spin[..., 0, None, :]).clamp_min(0)
        population = same + opposite
        capacity = population.diagonal(dim1=-2, dim2=-1)
        shape = population.shape
        eye = torch.eye(shape[-1], dtype=dm.dtype, device=dm.device)
        off_diagonal = 1 - eye
        geometric = distance.expand(shape)
        left, right = pooled[..., :, None, :], pooled[..., None, :, :]
        pair_features = torch.cat((left + right, (left - right).square(), torch.stack((
            same, opposite, torch.log1p(geometric),
            mass[..., :, None].expand(shape) + mass[..., None, :].expand(shape),
            (mass[..., :, None].expand(shape) - mass[..., None, :].expand(shape)).square(),
        ), -1)), -1)
        values = self.net(pair_features)
        # Density-derived Coulomb scale has no nuclear charge or atom identity.
        self_interaction = self.self_coulomb_scale.abs() * (n_positive.pow(4 / 3) @ self_weights) / mass.pow(4 / 3)
        self_interaction = self_interaction.clamp_min(1e-6)
        radius = self_interaction.reciprocal()
        mutual = torch.rsqrt(geometric.square() + .25 * (radius[..., :, None] + radius[..., None, :]).square())
        gain = (self_interaction[..., :, None] + self_interaction[..., None, :] - 2 * mutual).clamp_min(0)
        bridge_density = (n_positive @ bridge).reshape(shape).clamp_min(0)
        gap = 4 * torch.nn.functional.softplus(values[..., 0]) * torch.sqrt(bridge_density + 1e-10)
        fraction = (1 - gap / torch.sqrt(gap.square() + gain.square() + 1e-12)).clamp(0, 1)
        available = coherence.sum(-3).clamp_min(0) * off_diagonal
        proposal = available / available.sum(-1, keepdim=True).clamp_min(1e-12)
        symmetric_proposal = torch.sqrt(proposal * proposal.transpose(-1, -2) + 1e-24)
        raw_flow = (torch.sqrt(capacity[..., :, None] * capacity[..., None, :] + 1e-24) *
                    symmetric_proposal * fraction * off_diagonal)
        scale = (capacity / raw_flow.sum(-1).clamp_min(1e-12)).clamp(max=1)
        flow = raw_flow * torch.minimum(scale[..., :, None], scale[..., None, :])
        # Symmetry plus diagonal depletion preserves each pair marginal.
        static = -.25 * (flow * gain).sum((-1, -2))
        remaining = (capacity - flow.sum(-1)).clamp_min(0)
        coefficient = .1 * torch.sigmoid(self.dynamic(pooled).squeeze(-1))
        dynamic = -(remaining * self_interaction * coefficient).sum(-1)
        # Independent fragments retain only an asymptotic R^-6 coupling here.
        damp_radius = 2 * (radius[..., :, None] + radius[..., None, :])
        c6 = .5 * torch.sigmoid(values[..., 1]) / (
            self_interaction[..., :, None] * self_interaction[..., None, :]).square()
        dispersion = -.5 * (off_diagonal * population * c6 /
                              (geometric.pow(6) + damp_radius.pow(6) + 1e-10)).sum((-1, -2))
        exchange = -.5 * torch.einsum("...sij,...skl,ikjl->...", dm, dm, eri)
        return exchange + static + dynamic + dispersion
