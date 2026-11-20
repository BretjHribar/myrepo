"""Our scalar-energy model: local features plus collective spatial interactions.

No molecular identities, nuclear charges, or total-electron-number switches.
Coarse points are quadrature landmarks; their locations are held fixed while
taking electronic functional derivatives. First prototype, not a proven DFA.
"""
from __future__ import annotations

import torch
from torch import nn


def density_features(rho):
    # rho: (..., spin=2, density/xyz-gradient=4, grid)
    n = rho[..., 0, 0, :] + rho[..., 1, 0, :]
    positive = n.clamp_min(1e-12)
    polarization = (rho[..., 0, 0, :] - rho[..., 1, 0, :]) / positive
    grad = rho[..., 0, 1:4, :] + rho[..., 1, 1:4, :]
    spin_grad = rho[..., 0, 1:4, :] - rho[..., 1, 1:4, :]
    denom = positive.pow(8 / 3) + 1e-8
    reduced = (grad.square().sum(-2) / denom).clamp_min(0)
    spin_reduced = (spin_grad.square().sum(-2) / denom).clamp_min(0)
    root = positive.pow(1 / 3)
    return n, torch.stack((
        torch.log1p(positive), root / (1 + root), polarization.square(),
        torch.log1p(reduced) / 4, torch.log1p(spin_reduced) / 4,
    ), -1)


class SpatialTemplate:
    def __init__(self, coords, weights, landmarks):
        self.coords, self.weights, self.landmarks = coords, weights, landmarks
        distance = torch.cdist(coords, landmarks)
        self.partition = torch.softmax(-distance.square() / 2, dim=-1)
        self.weighted_partition = self.partition * weights[:, None]
        pair = torch.cdist(landmarks, landmarks)
        self.kernels = torch.stack([
            torch.exp(-pair.square() / (2 * scale**2)) for scale in (0.6, 1.8, 5.4)
        ] + [torch.rsqrt(1 + pair.square())])


class CollectiveXC(nn.Module):
    def __init__(self, width=32, layers=2, nonlocal_state=True):
        super().__init__()
        self.width, self.layers, self.nonlocal_state = width, layers, nonlocal_state
        self.local = nn.Sequential(nn.Linear(5, width), nn.Tanh(), nn.Linear(width, 1))
        self.encoder = nn.Sequential(nn.Linear(5, width), nn.Tanh())
        self.interactions = nn.ModuleList([
            nn.Sequential(nn.Linear(width * 5, width), nn.Tanh(),
                          nn.Linear(width, width), nn.Tanh()) for _ in range(layers)
        ])
        self.readout = nn.Linear(width, 1)
        nn.init.zeros_(self.local[-1].weight)
        nn.init.zeros_(self.local[-1].bias)
        nn.init.zeros_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)

    def forward(self, rho, template):
        n, features = density_features(rho)
        energy = .1 * (n * self.local(features).squeeze(-1) * template.weights).sum(-1)
        if not self.nonlocal_state:
            return energy
        mass = n @ template.weighted_partition
        pooled = torch.einsum("...gf,...g,ga->...af", features, n, template.weighted_partition)
        pooled = pooled / (mass[..., :, None] + 1e-10)
        hidden = self.encoder(pooled)
        for layer in self.interactions:
            messages = torch.einsum("kab,...bw,...b->...akw", template.kernels, hidden, mass)
            # Extensive sums, no molecule-global average or learned atom offsets.
            hidden = hidden + .25 * layer(torch.cat((hidden, messages.flatten(-2)), -1))
        return energy + .1 * (mass * self.readout(hidden).squeeze(-1)).sum(-1)


def rho_from_dm(dm, ao):
    """Spin density and its gradient; dm=(...,2,AO,AO), ao=(4,grid,AO)."""
    product = dm @ ao[0].T
    density = (product * ao[0].T).sum(-2)
    gradient = 2 * torch.einsum("...sag,dga->...sdg", product, ao[1:4])
    return torch.cat((density[..., :, None, :], gradient), -2)


class OccupiedProjector(torch.autograd.Function):
    """Integer-occupation projector with a stable occupied/virtual response.

    Equal-occupation degeneracies have zero divided difference. This avoids
    the irrelevant virtual/virtual singularity of a full eigenvector backward.
    Small occupied/virtual gaps are regularized and must be reported separately.
    This implements the first derivative, as used by the one-step pilot loss.
    """
    @staticmethod
    def forward(ctx, fock, occupations):
        eigenvalues, vectors = torch.linalg.eigh(fock)
        ctx.save_for_backward(eigenvalues, vectors, occupations)
        return (vectors * occupations[..., None, :]) @ vectors.transpose(-1, -2)

    @staticmethod
    def backward(ctx, gradient):
        energies, vectors, occupations = ctx.saved_tensors
        delta_e = energies[..., :, None] - energies[..., None, :]
        delta_f = occupations[..., :, None] - occupations[..., None, :]
        safe_gap = torch.where(delta_e >= 0, delta_e.clamp_min(1e-5), delta_e.clamp_max(-1e-5))
        divided = torch.where(delta_f == 0, torch.zeros_like(delta_f), delta_f / safe_gap)
        symmetric = (gradient + gradient.transpose(-1, -2)) / 2
        projected = vectors.transpose(-1, -2) @ symmetric @ vectors
        return vectors @ (projected * divided) @ vectors.transpose(-1, -2), None
