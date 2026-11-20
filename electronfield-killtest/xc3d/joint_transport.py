"""Our shared two-electron state, with density marginals and kinetic cost.

The positive probability table supplies a single pair's correlation budget.
It is an approximate coarse-state construction, not a real-wavefunction upper
bound and not yet a general-electron XC functional. No scalar residual head.
"""
import torch
from torch import nn


class JointTransport(nn.Module):
    def __init__(self, width=32, iterations=48):
        super().__init__()
        self.width, self.iterations = width, iterations
        self.node = nn.Sequential(nn.Linear(8, width), nn.Tanh(), nn.Linear(width, 1))
        self.pair = nn.Sequential(nn.Linear(19, width), nn.Tanh(), nn.Linear(width, width),
                                  nn.Tanh(), nn.Linear(width, 2))
        nn.init.zeros_(self.node[-1].weight)
        nn.init.constant_(self.node[-1].bias, -1.166434885)
        nn.init.zeros_(self.pair[-1].weight)
        nn.init.zeros_(self.pair[-1].bias)

    def forward(self, features, mass, edges, interaction, distance):
        p = mass[:, 0].clamp_min(1e-16)
        q = mass[:, 1].clamp_min(1e-16)
        p, q = p/p.sum(-1, keepdim=True), q/q.sum(-1, keepdim=True)
        pair_features = torch.cat((features[:, :, None]+features[:, None, :],
                                   (features[:, :, None]-features[:, None, :]).square(),
                                   torch.log1p(interaction.clamp_min(0))[..., None],
                                   torch.log1p(edges.sum(-3).clamp_min(0))[..., None],
                                   torch.log1p(distance)[..., None]), -1)
        raw = self.pair(pair_features)
        row_mean = (interaction*q[:, None]).sum(-1)
        col_mean = (interaction*p[:, :, None]).sum(-2)
        mean = (row_mean*p).sum(-1)
        centered = interaction-row_mean[:, :, None]-col_mean[:, None, :]+mean[:, None, None]
        inverse_temperature = .05*torch.exp(raw[..., 0].clamp(-6, 8))
        score = -centered*inverse_temperature+.05*torch.tanh(raw[..., 1])
        logp, logq = p.log(), q.log()
        left, right = torch.zeros_like(p), torch.zeros_like(q)
        for _ in range(self.iterations):
            left = logp-torch.logsumexp(score+right[:, None], -1)
            right = logq-torch.logsumexp(score+left[:, :, None], -2)
        probability = torch.exp(score+left[:, :, None]+right[:, None])
        # Finite Sinkhorn iterations may drift for sharp kernels. Round into
        # the transport polytope: scale rows/columns downward, then fill the
        # two nonnegative deficit vectors with a rank-one correction. This
        # makes the density constraints exact without CPU optimization.
        row_scale = torch.minimum(p/probability.sum(-1).clamp_min(1e-300), torch.ones_like(p))
        probability = probability*row_scale[:, :, None]
        col_scale = torch.minimum(q/probability.sum(-2).clamp_min(1e-300), torch.ones_like(q))
        probability = probability*col_scale[:, None, :]
        row_deficit = (p-probability.sum(-1)).clamp_min(0)
        col_deficit = (q-probability.sum(-2)).clamp_min(0)
        deficit = row_deficit.sum(-1).clamp_min(1e-300)
        probability = probability+row_deficit[:, :, None]*col_deficit[:, None, :]/deficit[:, None, None]
        amplitude = probability.clamp_min(1e-32).sqrt()
        kappa = .05+3.95*torch.sigmoid(self.node(features).squeeze(-1))
        scale = (kappa[:, :, None]*kappa[:, None, :]).sqrt()
        scaled_edges = edges*scale[:, None]
        diagonal_mask = torch.eye(p.shape[-1], dtype=p.dtype, device=p.device)
        scaled_edges = scaled_edges*(1-diagonal_mask)
        up_hopping = -.5*scaled_edges[:, 0]/(p[:, :, None]*p[:, None, :]).sqrt()
        down_hopping = -.5*scaled_edges[:, 1]/(q[:, :, None]*q[:, None, :]).sqrt()
        up = up_hopping+torch.diag_embed(.5*scaled_edges[:, 0].sum(-1)/p)
        down = down_hopping+torch.diag_embed(.5*scaled_edges[:, 1].sum(-1)/q)
        kinetic = (amplitude*(up@amplitude+amplitude@down.transpose(-1, -2))).sum((-1, -2))
        coulomb = (interaction*probability).sum((-1, -2))
        uncorrelated = (interaction*p[:, :, None]*q[:, None, :]).sum((-1, -2))
        candidate = kinetic+coulomb-uncorrelated
        correction = torch.minimum(candidate, torch.zeros_like(candidate))
        residual = torch.maximum((probability.sum(-1)-p).abs().amax(-1),
                                 (probability.sum(-2)-q).abs().amax(-1))
        return correction, {"marginal_residual": residual, "kinetic_cost": kinetic,
                            "coulomb_change": coulomb-uncorrelated, "candidate_positive": candidate > 0,
                            "kappa_min": kappa.amin(-1), "kappa_max": kappa.amax(-1)}
