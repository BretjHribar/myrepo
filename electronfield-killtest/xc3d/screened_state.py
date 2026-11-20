"""Our variational latent-screening functional for broader chemistry.

Local grid features include the Pauli kinetic-energy descriptor. Spatial
regions share a positive-definite response operator and a minimized latent
state, instead of accumulating independent pair-energy reductions. The latent
minimum is not a variational many-electron wavefunction upper bound.
"""
import math
import torch
from torch import nn
from .model import density_features


def pauli_features(rho, tau):
    n, features = density_features(rho)
    positive = n.clamp_min(1e-12)
    gradient = rho[..., 0, 1:4, :] + rho[..., 1, 1:4, :]
    weizsaecker = gradient.square().sum(-2) / (8 * positive)
    uniform = .3 * (3 * math.pi**2)**(2 / 3) * positive.pow(5 / 3)
    alpha = (tau.sum(-2) - weizsaecker).clamp_min(0) / (uniform + 1e-8)
    return n, torch.cat((features, torch.log1p(alpha)[..., :, None] / 4), -1)


class ScreenedStateXC(nn.Module):
    def __init__(self, width=32, channels=8, operator="diffusive", quantum=False):
        super().__init__()
        self.width, self.channels = width, channels
        self.operator = operator
        self.quantum_enabled = quantum
        self.local = nn.Sequential(nn.Linear(6, width), nn.Tanh(),
                                   nn.Linear(width, width), nn.Tanh(), nn.Linear(width, 1))
        self.encode = nn.Sequential(nn.Linear(10, width), nn.Tanh())
        self.source = nn.Linear(width, channels)
        self.stiffness = nn.Linear(width, 1)
        self.radial_logits = nn.Parameter(torch.zeros(3))
        self.bridge_strength = nn.Parameter(torch.tensor(0.))
        self.coulomb_strength = nn.Parameter(torch.tensor(-1.))
        self.context = nn.Sequential(nn.Linear(width + channels, 8), nn.Tanh())
        self.context_energy = nn.Sequential(nn.Linear(14, width), nn.Tanh(), nn.Linear(width, 1))
        nn.init.zeros_(self.local[-1].weight)
        nn.init.zeros_(self.local[-1].bias)
        nn.init.normal_(self.source.weight, std=.02)
        nn.init.zeros_(self.source.bias)
        nn.init.zeros_(self.context_energy[-1].weight)
        nn.init.zeros_(self.context_energy[-1].bias)

    def load_compatible(self, state):
        state = dict(state)
        old = state.get("encode.0.weight")
        if old is not None and old.shape[1] < 10:
            state["encode.0.weight"] = torch.nn.functional.pad(old, (0, 10-old.shape[1]))
        result = self.load_state_dict(state, strict=False)
        allowed = {"coulomb_strength"} | {key for key in self.state_dict()
                                           if key.startswith(("context.", "context_energy."))}
        if result.unexpected_keys or set(result.missing_keys) - allowed:
            raise RuntimeError(f"Unexpected checkpoint incompatibility: {result}")
        return result

    def prepare(self, template):
        distance = torch.cdist(template.landmarks, template.landmarks)
        radial = torch.stack([torch.exp(-distance.square() / (2 * length**2)) for length in (.8, 2., 5.)])
        eye = torch.eye(len(distance), device=distance.device, dtype=distance.dtype)
        count = len(distance)
        pairs = set()
        if count > 1:
            nearest = (distance + eye * 1e6).argsort(-1)[:, :min(2, count-1)].cpu().tolist()
            for left, neighbors in enumerate(nearest):
                for right in neighbors:
                    pairs.add(tuple(sorted((left, right))))
        ordered = sorted(pairs)
        if ordered:
            mids = torch.stack([(template.landmarks[a] + template.landmarks[b]) / 2 for a, b in ordered])
            kernel = torch.exp(-torch.cdist(template.coords, mids).square() / .5)
            weighted = kernel * template.weights[:, None]
            weighted = weighted / weighted.sum(0).clamp_min(1e-8)
            incidence = torch.zeros(count, len(ordered), device=distance.device, dtype=distance.dtype)
            for edge, (a, b) in enumerate(ordered):
                value = torch.exp(-distance[a, b].square() / 32)
                incidence[a, edge] = incidence[b, edge] = value
            incidence = incidence / incidence.sum(-1, keepdim=True).clamp_min(1e-8)
        else:
            weighted = template.coords.new_zeros(len(template.coords), 0)
            incidence = distance.new_zeros(count, 0)
        coulomb = torch.rsqrt(1 + distance.square())
        return radial, eye, weighted, incidence, coulomb

    def forward(self, rho, tau, template, prepared=None, quantum=None, suppression=None):
        n, features = pauli_features(rho, tau)
        weights64 = getattr(template, "weights64", template.weights).double()
        grid_gate = 1. if suppression is None else 1-torch.einsum("ga,...a->...g", template.partition.to(rho.dtype), suppression)
        local = .1 * (n.double() * self.local(features).squeeze(-1).double() * weights64 * grid_gate).sum(-1)
        weighted_partition = template.weighted_partition.to(rho.dtype)
        # Valence-weighted pooling keeps dense atomic cores from drowning out
        # the bond region. It is a density measure, not an element embedding.
        valence = n / (1 + n.clamp_min(0) / .2).square()
        mass = (valence @ weighted_partition).clamp_min(0)
        pooled = torch.einsum("...gf,...g,ga->...af", features, valence, weighted_partition)
        pooled = pooled / (mass[..., :, None] + 1e-8)
        charge = (n @ weighted_partition).clamp_min(1e-8)
        spin_density = rho[..., 0, 0, :] - rho[..., 1, 0, :]
        regional_spin = (spin_density @ weighted_partition) / charge
        occupation = torch.zeros_like(charge)
        purity = torch.zeros_like(charge)
        coherence = None
        if self.quantum_enabled:
            if quantum is None:
                raise ValueError("This functional requires regional density-matrix descriptors")
            charge = quantum["charge"].clamp_min(1e-8)
            shared = quantum["coherence"]
            eta = shared.sum(-1) / charge
            # Smooth at paired integer occupation: this descriptor has no
            # artificial first-order penalty normal to an idempotent RKS DM.
            occupation = eta - eta.square() / 4
            purity = shared.diagonal(dim1=-2, dim2=-1) / charge.square()
            coherence = shared / (charge[..., :, None] * charge[..., None, :])
        # Regional charge is an electronic observable, not a nuclear-charge
        # embedding or a molecule-global electron-number switch.
        coarse = torch.cat((pooled, torch.log1p(charge)[..., :, None], regional_spin.square()[..., :, None],
                            occupation[..., :, None], purity[..., :, None]), -1)
        hidden = self.encode(coarse)
        root = torch.sqrt(mass + 1e-8)
        source = .2 * root[..., :, None] * torch.tanh(self.source(hidden))
        if suppression is not None:
            source = source * (1-suppression[..., :, None])
        stiffness = .1 + torch.nn.functional.softplus(self.stiffness(hidden).squeeze(-1))
        radial, identity, bridge_weights, incidence, coulomb = self.prepare(template) if prepared is None else prepared
        if bridge_weights.shape[-1]:
            bridge_density = (n @ bridge_weights).clamp_min(0)
            bridge_signal = torch.log1p(bridge_density / .02) @ incidence.T
            # Electron density connecting regions raises the effective gap;
            # a weak density bridge permits stronger collective relaxation.
            stiffness = stiffness + torch.nn.functional.softplus(self.bridge_strength) * bridge_signal
        radial = torch.einsum("k,kab->ab", torch.softmax(self.radial_logits, 0), radial)
        similarity = torch.sigmoid(hidden @ hidden.transpose(-1, -2) / math.sqrt(self.width))
        if self.operator == "coherent":
            tail = coulomb if coherence is None else coulomb * coherence
            radial = radial + torch.nn.functional.softplus(self.coulomb_strength) * tail
        edges = radial * similarity * root[..., :, None] * root[..., None, :] * (1 - identity)
        if self.operator == "coherent":
            # A graph Laplacian cannot change a uniform source mode. This
            # bounded attractive operator can, and retains a Coulomb tail.
            row_scale = torch.sqrt(1 + edges.sum(-1))
            normalized = edges / (row_scale[..., :, None] * row_scale[..., None, :])
            diagonal = torch.sqrt(stiffness)
            response_operator = diagonal[..., :, None] * (identity - .95 * normalized) * diagonal[..., None, :]
        else:
            response_operator = torch.diag_embed(stiffness + edges.sum(-1)) - edges
        latent = torch.linalg.solve(response_operator, source)
        collective = -.05 * (source * latent).double().sum((-1, -2))
        node_context = self.context(torch.cat((hidden, latent), -1))
        grid_context = torch.einsum("ga,...ac->...gc", template.partition.to(rho.dtype), node_context)
        contextual = self.context_energy(torch.cat((features, grid_context), -1)).squeeze(-1)
        context_energy = .1 * (n.double() * contextual.double() * weights64 * grid_gate).sum(-1)
        return local + collective + context_energy


def tau_from_dm(dm, ao):
    return .5 * torch.einsum("...sij,dgi,dgj->...sg", dm, ao[1:4], ao[1:4])
