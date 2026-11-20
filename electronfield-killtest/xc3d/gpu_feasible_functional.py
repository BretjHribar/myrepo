"""Density-constrained state search with explicit feasibility restoration.

Replaces the failed penalty experiment, rather than increasing its iterations.
The small dense operator action is a finite-space research implementation;
neither the reference energy nor the exact dual potential drives optimization.
Constrained search and SQP are established mathematics, not novelty claims.
"""
from .runtime import configure, torch_setup
ROOT = configure(gpu_only_compatible=True)

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import torch
from .gpu_moment_functional import prepare, oracle
from .gpu_spin_chain import SpinNumberState, amplitudes
from .gpu_manybody_ops import dense_operator

KCAL = 627.509474


class FeasibleState:
    def __init__(self, state, matrix, moments, configurations, target):
        self.state, self.matrix, self.moments = state, matrix, moments
        self.configurations, self.target = configurations, target
        self.parameters = list(state.parameters())

    def point(self):
        return torch.cat([p.detach().flatten() for p in self.parameters])

    @torch.no_grad()
    def assign(self, point):
        start = 0
        for parameter in self.parameters:
            parameter.copy_(point[start:start+parameter.numel()].reshape_as(parameter))
            start += parameter.numel()

    def values(self):
        vector = amplitudes(self.state.cores(), self.configurations)
        vector = vector/vector.norm()
        energy = vector @ self.matrix @ vector
        moments = torch.einsum("i,aij,j->a", vector, self.moments, vector)
        return torch.cat((energy[None], moments))

    def derivatives(self, energy=True):
        values = self.values()
        selected = values if energy else values[1:]
        rows = []
        for index, scalar in enumerate(selected):
            gradients = torch.autograd.grad(scalar, self.parameters, retain_graph=index+1 < len(selected))
            rows.append(torch.cat([g.flatten() for g in gradients]))
        jacobian = torch.stack(rows).detach()
        return values.detach(), jacobian

    @staticmethod
    def solve(matrix, rhs):
        # Discard only numerical gauge directions. A lost density direction is
        # exposed by the residual and cannot be converted into a passing score.
        values, vectors = torch.linalg.eigh((matrix+matrix.T)/2)
        inverse = torch.where(values > values[-1]*1e-11, 1/values.clamp_min(1e-30), 0.)
        return vectors @ (inverse*(vectors.T @ rhs))

    def restore(self, point, tolerance=1e-9, iterations=18):
        self.assign(point)
        for iteration in range(iterations):
            values, jacobian = self.derivatives(energy=False)
            residual = values[1:]-self.target
            error = float(residual.norm())
            if error <= tolerance:
                return self.point(), True, error, iteration
            update = -jacobian.T @ self.solve(jacobian @ jacobian.T, residual)
            update *= torch.clamp(.4/update.norm().clamp_min(1e-30), max=1.)
            original = self.point()
            accepted = False
            for scale in (1., .5, .25, .125, .0625, .03125):
                self.assign(original+scale*update)
                with torch.no_grad():
                    trial = float((self.values()[1:]-self.target).norm())
                if trial < error:
                    accepted = True
                    break
            if not accepted:
                self.assign(original)
                break
        with torch.no_grad():
            error = float((self.values()[1:]-self.target).norm())
        return self.point(), error <= tolerance, error, iterations

    def minimize(self, steps=80):
        point, feasible, residual, _ = self.restore(self.point())
        history = []
        if not feasible:
            return {"status": "initial_density_restoration_failed", "residual": residual, "history": history}
        inverse = torch.eye(len(point), device=point.device)
        previous = None
        for step in range(steps):
            values, derivatives = self.derivatives()
            gradient, jacobian = derivatives[0], derivatives[1:]
            potential = -self.solve(jacobian @ jacobian.T, jacobian @ gradient)
            lagrangian_gradient = gradient+jacobian.T @ potential
            if previous is not None:
                old_point, old_gradient, old_jacobian = previous
                displacement = point-old_point
                difference = gradient-old_gradient+(jacobian-old_jacobian).T @ potential
                curvature = displacement @ difference
                if float(curvature) > 1e-12:
                    rho = 1/curvature
                    unit = torch.eye(len(point), device=point.device)
                    transform = unit-rho*displacement[:, None]*difference[None]
                    inverse = transform @ inverse @ transform.T+rho*displacement[:, None]*displacement[None]
            metric_q = jacobian @ inverse @ jacobian.T
            metric_potential = -self.solve(metric_q, jacobian @ inverse @ gradient)
            direction = -inverse @ (gradient+jacobian.T @ metric_potential)
            direction *= torch.clamp(.3/direction.norm().clamp_min(1e-30), max=1.)
            row = {"step": step, "universal_energy_hartree": float(values[0]),
                   "moment_residual": float((values[1:]-self.target).norm()),
                   "stationarity_residual": float(lagrangian_gradient.norm())}
            if step % 10 == 0:
                history.append(row)
                print(json.dumps(row), flush=True)
            if float(lagrangian_gradient.norm()) < 1e-7:
                break
            original = point.clone()
            accepted = False
            for scale in (1., .5, .25, .125, .0625, .03125, .015625):
                trial, feasible, _, _ = self.restore(original+scale*direction)
                with torch.no_grad():
                    trial_energy = self.values()[0]
                if feasible and float(trial_energy) < float(values[0])-1e-13:
                    previous = original, gradient, jacobian
                    point = trial
                    accepted = True
                    break
            if not accepted:
                self.assign(original)
                break
        values, derivatives = self.derivatives()
        gradient, jacobian = derivatives[0], derivatives[1:]
        potential = -self.solve(jacobian @ jacobian.T, jacobian @ gradient)
        residual = float((values[1:]-self.target).norm())
        return {"status": "feasible" if residual < 1e-8 else "density_constraint_failed",
                "residual": residual, "energy": values[0], "potential": potential,
                "stationarity_residual": float((gradient+jacobian.T @ potential).norm()),
                "steps": step+1, "history": history}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--output", type=Path, default=Path("research/feasible_density_functional_gpu_20261116.json"))
    args = parser.parse_args()
    device = torch_setup()
    started = time.perf_counter()
    data = prepare(device)
    matrix = dense_operator(data["universal_words"], data["universal_coefficient"], data["bits"], 6)
    moments = torch.stack([dense_operator(data["q_words"], row, data["bits"], 6)
                           for row in data["q_coefficients"]])
    state = SpinNumberState(6, 4, channels=2, device=device)
    checkpoint = torch.load("research/general_spin_state_4e_gpu_20261116.pt", map_location=device, weights_only=True)
    state.load_state_dict(checkpoint["model"])
    solver = FeasibleState(state, matrix, moments, data["configurations"], data["target"])
    result = solver.minimize(args.steps)
    # Compute reference diagnostics only after the model solve has ended.
    exact = oracle(data, data["target"])
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "device": device,
              "scope": "N=4, six-orbital fixed-spatial-moment constrained search; no broad/full-density XC claim",
              "reference_energy_fitted": False, "external_potential_in_objective": False,
              "status": result["status"], "density_moment_residual": result["residual"],
              "history": result["history"], "oracle_residual": exact["residual"],
              "seconds": time.perf_counter()-started, "peak_cuda_bytes": torch.cuda.max_memory_allocated()}
    if "energy" in result:
        report.update(functional_error_kcal=float((result["energy"]-exact["functional"])*KCAL),
                      potential_difference_norm=float((result["potential"]-exact["potential"]).norm()),
                      stationarity_residual=result["stationarity_residual"], steps=result["steps"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    torch.save({"model": state.state_dict(), "target": data["target"].cpu(), "report": report}, args.output.with_suffix(".pt"))
    print(json.dumps({key: value for key, value in report.items() if key != "history"}), flush=True)


if __name__ == "__main__":
    main()
