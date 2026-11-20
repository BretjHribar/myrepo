"""Implicit spatial-density XC search with no molecular external potential.

Accuracy research backend for six-electron singlets in a complete finite
orbital basis. Ground-state Legendre searches and numerical linear algebra
are established primitives, not novelty claims. The input contains only T,
W, a fixed spatial-density frame, occupation metadata and the input density.
It is intentionally separate from the molecular-h-selected block initializer.
"""
import math
import torch
from .gpu_canonical_density import slater_vector
from .gpu_six_particle import SixParticleOperator, SixParticleFockAction
from .gpu_four_particle import incremental_ritz
from .gpu_ensemble_dual import ensemble_dual
from .intrinsic_density_frame import one_body_density
from .gpu_fock_blocks import SingletProjector


def evaluate_xc(universal, density, maximum_updates=40, density_tolerance=1e-7,
                state_tolerance=1e-8, maximum_ritz_dimension=256, entropy_allowance=1e-5,
                callback=None):
    if set(universal) != {"kinetic", "eri", "operators", "bits", "electrons"}:
        raise ValueError("The universal interface accepts only T, W, Q, bits and N; external potentials/labels are excluded")
    if density.device.type != "cuda" or universal["electrons"] != 6:
        raise ValueError("This research backend currently requires CUDA and six-electron singlet densities")
    kinetic, eri, operators, bits = (universal[key] for key in ("kinetic", "eri", "operators", "bits"))
    with torch.no_grad():
        if density.dtype != torch.float64 or float((density-density.T).norm()) > 1e-9 or not bool(torch.isfinite(density).all()):
            raise ValueError("The current interface requires a finite real symmetric float64 density")
        if any(universal[key].device != density.device for key in ("kinetic", "eri", "operators", "bits")):
            raise ValueError("All universal inputs must share the input density's CUDA device")
        occupations, natural = torch.linalg.eigh((density+density.T)/2)
        if float(occupations.min()) < -1e-9 or float(occupations.max()) > 2+1e-9 or float((density.trace()-6).abs()) > 1e-8:
            raise ValueError("Invalid fixed-number ensemble input density")
        identity = torch.eye(len(density), device=density.device)
        if float((operators[0]-identity/identity.new_tensor(len(identity)).sqrt()).norm()) > 1e-8:
            raise ValueError("The first constraint must be the normalized fixed-number operator")
        target = torch.einsum("aij,ij->a", operators, density)
        beta = 2*len(density)*math.log(2)/entropy_allowance
        effective = (natural*((2/occupations.clamp(1e-14, 2-1e-14)-1).log()/beta)) @ natural.T
        kinetic_initial = torch.einsum("aij,ij->a", operators, effective-kinetic)
        noninteracting = ensemble_dual(kinetic, operators, target, beta, statistics="onebody", initial=kinetic_initial)
        if noninteracting["residual"] > 1e-8:
            raise RuntimeError(f"Noninteracting density search failed: {noninteracting['residual']}")
        constrained = operators[1:]  # Remove the fixed-N potential gauge.
        coulomb = torch.einsum("ijkl,kl->ij", eri, density)
        exchange = torch.einsum("ikjl,kl->ij", eri, density)
        guess = noninteracting["potential"][1:]-torch.einsum("aij,ij->a", constrained, coulomb-.5*exchange)
        # A positive Gaussian occupation covariance only sets dual coordinates;
        # it does not replace the physical interacting objective or response.
        rotated = natural.T @ constrained @ natural
        covariance_weights = (occupations[:, None]*(1-occupations[None]/2)
                              +occupations[None]*(1-occupations[:, None]/2))
        covariance = torch.einsum("aij,ij,bij->ab", rotated, covariance_weights, rotated)
        values, vectors = torch.linalg.eigh((covariance+covariance.T)/2)
        coordinate_map = (vectors*values.clamp_min(values.max()*1e-8).rsqrt()) @ vectors.T
        coordinates = torch.linalg.solve(coordinate_map, guess)
        generator = torch.Generator(device=density.device).manual_seed(17112029)
        generic, _ = torch.linalg.qr(torch.randn(len(density), 3, device=density.device, generator=generator))
        state = slater_vector(natural, torch.arange(len(density)-3, len(density), device=density.device), bits)
        state += .01*slater_vector(generic, torch.arange(3, device=density.device), bits)
        state /= state.norm()
        operator = SixParticleOperator(kinetic, eri, 6)
        action = SixParticleFockAction(operator, bits)
        singlet = SingletProjector(bits, len(density), 3)
        state = singlet(state); state /= state.norm()
        initial_spin = float(singlet.spin_squared(state).norm())
        if initial_spin > 1e-9:
            raise RuntimeError(f"Universal seed failed its singlet projector: {initial_spin}")
    parameter = torch.nn.Parameter(coordinates)
    optimizer = torch.optim.LBFGS([parameter], lr=1., max_iter=1, max_eval=8, history_size=12,
                                  tolerance_grad=1e-10, tolerance_change=1e-13, line_search_fn="strong_wolfe")
    latest, history = {}, []
    evaluations = 0

    @torch.no_grad()
    def closure():
        nonlocal state, evaluations, latest
        potential = coordinate_map @ parameter
        if latest and torch.equal(potential, latest["potential"]):
            parameter.grad = coordinate_map.T @ (target[1:]-latest["moments"])
            return latest["objective"]
        operator.h = kinetic+torch.einsum("a,aij->ij", potential, constrained)
        solution = incremental_ritz(lambda value: singlet(action(singlet(value))), state,
                                     maximum=maximum_ritz_dimension, tolerance=state_tolerance,
                                     log_every=25, projector=singlet)
        if not solution["converged"]:
            spin_error = float(singlet.spin_squared(solution["state"]).norm())
            raise RuntimeError(f"Interacting potential solve failed its state residual: {solution['residual']}; S2-state norm: {spin_error}")
        state = solution["state"]
        output_density = one_body_density(state, bits, len(density))
        moments = torch.einsum("aij,ij->a", constrained, output_density)
        dual = solution["energy"]-potential @ target[1:]
        parameter.grad = coordinate_map.T @ (target[1:]-moments)
        latest = {"potential": potential.clone(), "moments": moments, "density": output_density,
                  "dual": dual, "objective": -dual, "state_residual": solution["residual"], "state": state}
        evaluations += 1
        return latest["objective"]

    closure()
    for update in range(maximum_updates+1):
        with torch.no_grad():
            error = float((latest["moments"]-target[1:]).norm())
            row = {"update": update, "universal_functional_hartree": float(latest["dual"]),
                   "density_residual": error, "state_residual": latest["state_residual"],
                   "ground_state_evaluations": evaluations}
            history.append(row)
            if callback:
                callback(row)
            if error < density_tolerance or update == maximum_updates:
                break
        optimizer.step(closure)
        closure()  # Cached when the accepted line-search point is current.
    with torch.no_grad():
        spin_error = float(singlet.spin_squared(latest["state"]).norm())
        valid = error < density_tolerance and latest["state_residual"] < state_tolerance and spin_error < 1e-8
        hartree = .5*(density*coulomb).sum()
        xc = latest["dual"]-noninteracting["dual"]-hartree
        potential = (torch.einsum("a,aij->ij", noninteracting["potential"][1:]-latest["potential"], constrained)
                     -coulomb)
        return {"converged": valid, "xc_energy": xc if valid else None, "xc_potential": potential if valid else None,
                "rejected_raw_xc_energy": None if valid else xc, "universal_functional": latest["dual"],
                "interacting_potential": latest["potential"], "interacting_state": latest["state"],
                "interacting_density": latest["density"], "density_residual": error,
                "state_residual": latest["state_residual"], "noninteracting_residual": noninteracting["residual"],
                "spin_squared_state_norm": spin_error,
                "ground_state_evaluations": evaluations, "history": history, "entropy_allowance": entropy_allowance}
