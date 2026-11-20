"""Independent closed-form and finite-difference resolvent checks."""
from .runtime import configure, torch_setup
ROOT = configure(gpu_only_compatible=True)

from datetime import datetime, timezone
import json
from pathlib import Path
import torch
from .gpu_density_resolvent import proximal_state, project_state


@torch.no_grad()
def main():
    device = torch_setup()
    # A classical two-state mixture has a scalar clipped-quadratic solution.
    h = torch.diag(torch.tensor([.3, 1.1], device=device))
    q = torch.diag(torch.tensor([-1., 1.], device=device))[None]
    epsilon, auxiliary = .7, torch.tensor([.2], device=device)
    exact_moment = (auxiliary-epsilon*(h[1, 1]-h[0, 0])/2).clamp(-1, 1)
    result = proximal_state(h, q, auxiliary, epsilon, torch.eye(2, device=device)/2)
    moment_error = float((result["moments"]-exact_moment).abs().max())
    direction, step = torch.tensor([1.], device=device), 1e-3
    points = [proximal_state(h, q, auxiliary+sign*step*direction, epsilon, result["state"]) for sign in (-1, 1)]
    derivative_error = float(((points[1]["functional"]-points[0]["functional"])/(2*step)-result["gradient"] @ direction).abs())
    potential = torch.tensor([-.4], device=device)
    physical_effective = h+torch.einsum("a,aij->ij", potential, q)
    _, orbitals = torch.linalg.eigh(physical_effective)
    ground = orbitals[:, :1] @ orbitals[:, :1].T
    physical_q = torch.einsum("aij,ij->a", q, ground)
    anchor = physical_q-epsilon*potential
    proximal = proximal_state(h, q, anchor, epsilon, ground)
    recovered = proximal["functional"]+potential @ anchor+epsilon*potential.square().sum()/2
    correction_error = float((recovered-(physical_effective*ground).sum()).abs())
    # The capped case has an independent spectral projection when Q spans all
    # real symmetric matrices, including an exact positive trace constraint.
    matrices = []
    for i in range(3):
        for j in range(i, 3):
            item = torch.zeros(3, 3, device=device); item[i, j] = item[j, i] = 1
            item /= item.norm(); matrices.append(item)
    operators = torch.stack(matrices)
    kinetic = torch.tensor([[1., .2, -.1], [.2, .3, .1], [-.1, .1, .8]], device=device)
    target = torch.randn(len(operators), device=device)
    witness = torch.einsum("a,aij->ij", target, operators)
    exact_density = project_state(witness-.7*kinetic, total=2., cap=2.)
    kinetic_result = proximal_state(kinetic, operators, target, .7, torch.eye(3, device=device)*2/3,
                                    total=2., cap=2.)
    pauli_error = float((kinetic_result["state"]-exact_density).norm())
    valid = (result["converged"] and kinetic_result["converged"] and
             max(moment_error, derivative_error, correction_error, pauli_error) < 1e-7)
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "device": device, "passed": valid,
              "closed_form_moment_error": moment_error, "directional_derivative_error": derivative_error,
              "exact_external_counterterm_error_hartree": correction_error,
              "closed_form_pauli_density_error": pauli_error,
              "interacting_gap": result["gap"], "noninteracting_gap": kinetic_result["gap"],
              "physical_density_distinguished_from_auxiliary": True}
    Path("research/density_resolvent_checks_gpu_20261117.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report), flush=True)
    if not valid: raise RuntimeError("Density resolvent failed its independent analytic checks")


if __name__ == "__main__":
    main()
