"""Exact ensemble energies and directional potentials for training our model."""
import json
import numpy as np
import torch
from pyscf import dft
from .model import SpatialTemplate


class ConstraintShape:
    def __init__(self, path, device, model):
        with np.load(path) as data:
            self.metadata = json.loads(str(data["metadata"]))
            self.rho = data["rho_total"].astype(float)
            self.vh = data["hartree_potential"].astype(float)
            self.coords = data["coords"].astype(float)
            self.weights = data["weights"]
            self.template = SpatialTemplate(*[torch.tensor(data[key], dtype=torch.float32, device=device)
                                               for key in ("coords", "weights", "landmarks")])
            self.template.weights64 = torch.tensor(self.weights, dtype=torch.float64, device=device)
        self.prepared = model.prepare(self.template)
        self.device = device

    def loss(self, model, rng, directional=True):
        occupation = float(rng.choice([.25, .5, .75, 1.]))
        spin_up = float(rng.choice([0., .25, .5, .75, 1.]))
        spin = np.array([spin_up, 1-spin_up])
        rho = occupation * spin[:, None, None] * self.rho[None]
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        length = float(rng.choice([.5, 1., 2.]))
        argument = (self.coords - self.coords.mean(0)) @ direction / length
        modulation = np.tanh(argument)
        modulation -= np.dot(self.weights * self.rho[0], modulation)
        derivative_modulation = (1 - np.tanh(argument)**2)[:, None] * direction[None, :] / length
        variation = occupation * self.rho * modulation[None]
        variation[1:4] += occupation * self.rho[0][None] * derivative_modulation.T
        spin_variation = spin[:, None, None] * variation[None]
        epsilon, vxc, _, _ = dft.numint.NumInt().eval_xc_eff("pbe", rho, deriv=1, spin=1)
        exc = float(np.dot(self.weights, epsilon * rho[:, 0].sum(0)))
        target = -occupation**2 * self.metadata["hartree_normalized"] - exc
        xc_slope = float(np.sum(self.weights[None, None] * vxc * spin_variation))
        target_slope = -float(np.dot(self.weights, occupation * self.vh * variation[0])) - xc_slope
        fields = torch.tensor(rho, dtype=torch.float32, device=self.device)
        change = torch.tensor(spin_variation, dtype=torch.float32, device=self.device)
        amplitude = torch.zeros((), dtype=torch.float32, device=self.device, requires_grad=directional)
        fields = fields + amplitude * change
        n = fields[:, 0].sum(0).clamp_min(1e-12)
        gradient = fields[:, 1:4].sum(0)
        tau_total = gradient.square().sum(0) / (8 * n)
        tau = torch.tensor(spin, dtype=torch.float32, device=self.device)[:, None] * tau_total
        predicted = model(fields, tau, self.template, self.prepared)
        energy_loss = ((predicted - target) / .02).square()
        derivative_loss = predicted.new_zeros(())
        if directional:
            slope = torch.autograd.grad(predicted, amplitude, create_graph=True)[0]
            derivative_loss = ((slope - target_slope) / .02).square()
        return energy_loss, derivative_loss


class OrbitalConstraint:
    """Energy and orbital-direction labels with consistent regional coherence."""
    def __init__(self, path, device, model):
        with np.load(path) as data:
            self.metadata = json.loads(str(data["metadata"]))
            self.phi = data["phi"].astype(float)
            self.directions = data["directions"].astype(float)
            self.probability = data["probability"]
            self.probability_directions = data["probability_directions"]
            self.weights = data["weights"]
            self.template = SpatialTemplate(*[torch.tensor(data[key], dtype=torch.float32, device=device)
                                               for key in ("coords", "weights", "landmarks")])
            self.template.weights64 = torch.tensor(self.weights, dtype=torch.float64, device=device)
        self.prepared = model.prepare(self.template)
        self.device = device

    def loss(self, model, rng, directional=True):
        occupation = float(rng.choice([.25, .5, .75, 1.]))
        spin_up = float(rng.choice([0., .25, .5, .75, 1.]))
        spin = np.array([spin_up, 1-spin_up])
        index = int(rng.integers(len(self.directions)))
        phi, delta = self.phi, self.directions[index]
        total = np.vstack((phi[0]**2, 2*phi[0]*phi[1:4])) * occupation
        variation = np.vstack((2*phi[0]*delta[0], 2*(delta[0]*phi[1:4]+phi[0]*delta[1:4]))) * occupation
        rho = spin[:, None, None] * total[None]
        change = spin[:, None, None] * variation[None]
        eps, vxc, _, _ = dft.numint.NumInt().eval_xc_eff("pbe", rho, deriv=1, spin=1)
        exc = float(np.dot(self.weights, eps * total[0]))
        target = -occupation**2 * self.metadata["hartree_normalized"] - exc
        target_slope = (-occupation**2 * self.metadata["hartree_slopes"][index]
                        - float(np.sum(self.weights[None, None] * vxc * change)))
        def tensor(array):
            return torch.tensor(array, dtype=torch.float32, device=self.device)
        amplitude = torch.zeros((), dtype=torch.float32, device=self.device, requires_grad=directional)
        fields = tensor(rho) + amplitude * tensor(change)
        tau_total = occupation * .5*np.sum(phi[1:4]**2, axis=0)
        tau_variation = occupation * np.sum(phi[1:4]*delta[1:4], axis=0)
        tau = tensor(spin[:, None]) * (tensor(tau_total) + amplitude*tensor(tau_variation))
        probability = tensor(self.probability) + amplitude*tensor(self.probability_directions[index])
        quantum = {"charge": occupation*probability,
                   "coherence": occupation**2*probability[:, None]*probability[None, :]}
        predicted = model(fields, tau, self.template, self.prepared, quantum=quantum)
        energy_loss = ((predicted-target)/.02).square()
        derivative_loss = predicted.new_zeros(())
        if directional:
            slope = torch.autograd.grad(predicted, amplitude, create_graph=True)[0]
            derivative_loss = ((slope-target_slope)/.02).square()
        return energy_loss, derivative_loss
