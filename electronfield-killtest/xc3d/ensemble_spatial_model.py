"""Feasible canonical-ensemble universal search in physical response spaces.

This accuracy backend is an implicit finite-basis diagnostic. It does not
claim new constrained-search mathematics, a learned predictor or a scalable
many-fragment state. It never accepts h, nuclear charges or reference labels.
"""
import math
import torch
from .gpu_canonical_density import canonical_lift, slater_vector
from .gpu_fock_blocks import SingletProjector
from .gpu_density_links import DensityLinks
from .gpu_density_completion import append_directions
from .gpu_ensemble_dual import ensemble_dual
from .gpu_charge_spin_accuracy import PhysicalAction
from .gpu_four_particle import incremental_ritz
from .gpu_psd_ensemble import optimize_ensemble


@torch.no_grad()
def leading_completion(basis, sources, count, maximum):
    candidates = sources.reshape(len(basis), -1)
    for _ in range(2):
        candidates = candidates-basis @ (basis.T @ candidates)
    gram = candidates.T @ candidates
    values, vectors = torch.linalg.eigh((gram+gram.T)/2)
    values, vectors = values.flip(0), vectors.flip(1)
    # Gram eigenvalues at roundoff are not physical source directions. Using
    # a cutoff below machine precision promoted many numerical null vectors.
    numerical_cutoff = max(1e-24, float(values[0])*64*torch.finfo(values.dtype).eps*candidates.shape[1])
    eligible = values > numerical_cutoff
    available = int(eligible.sum())
    if not available:
        return basis, 0, False
    boundary = values[min(count, available)-1]
    boundary_tolerance = max(1e-24, float(boundary)*1e-8, numerical_cutoff/64)
    selected = eligible & (values >= boundary-boundary_tolerance)
    if basis.shape[1]+int(selected.sum()) > maximum:
        return basis, 0, True
    directions = (candidates @ vectors[:, selected])/values[selected].sqrt()
    updated = append_directions(basis, directions, maximum)
    return updated, updated.shape[1]-basis.shape[1], False


@torch.no_grad()
def evaluate_xc(universal, density, protocol, callback=None, checkpoint=None, checkpoint_callback=None):
    if protocol.get("status", "").startswith("closed"):
        raise RuntimeError("This hard-density ensemble protocol is closed after its bounded convergence failure")
    if set(universal) != {"kinetic", "eri", "operators", "bits", "electrons"}:
        raise ValueError("Universal search accepts only T, W, Q, occupation metadata and N")
    kinetic, eri, operators, bits, number = (universal[key] for key in ("kinetic", "eri", "operators", "bits", "electrons"))
    if density.device.type != "cuda" or density.dtype != torch.float64 or number % 2:
        raise ValueError("CUDA float64 and an even-N spin-unpolarized ensemble are required")
    if min(number, 2*len(density)-number) not in (4, 6):
        raise ValueError("This matrix-free Hamiltonian backend currently covers four/six particles or holes")
    if float((density-density.T).norm()) > 1e-9 or not bool(torch.isfinite(density).all()):
        raise ValueError("Input density must be finite and real symmetric")
    occupations, natural = torch.linalg.eigh(density)
    if float(occupations.min()) < -1e-9 or float(occupations.max()) > 2+1e-9 or float((density.trace()-number).abs()) > 1e-8:
        raise ValueError("Invalid ensemble occupations or total number")
    identity = torch.eye(len(density), device=density.device)
    if float((operators[0]-identity/identity.new_tensor(len(identity)).sqrt()).norm()) > 1e-8:
        raise ValueError("First constraint must be normalized fixed number")
    target = torch.einsum("aij,ij->a", operators, density)
    moments, requested = operators[1:], target[1:]
    lift = canonical_lift(natural, occupations, number, bits)
    if lift["occupation_residual"] > 1e-8:
        raise RuntimeError("Canonical initialization did not retain the input occupations")
    singlet = SingletProjector(bits, len(density), number//2)
    links = DensityLinks(bits, len(density))
    physical = PhysicalAction({"h": kinetic, "eri": eri, "bits": bits,
                               "nuclear": kinetic.new_zeros(()), "case": {"electrons": number}})
    def action(vector):
        return singlet(physical(singlet(vector)))
    maximum = protocol["maximum_response_space_dimension"]
    if checkpoint is None:
        basis = append_directions(kinetic.new_zeros(len(bits), 0), lift["vectors"].T, maximum)
        initial_count = basis.shape[1]
        images = torch.stack([action(vector) for vector in basis.unbind(1)], 1)
        basis = append_directions(basis, images, maximum)
        leading = lift["weights"].argsort(descending=True)[:protocol["maximum_density_response_states"]]
        source_states = lift["vectors"][leading].T*lift["weights"][leading].sqrt()
        response = links.response_images(moments, source_states).permute(1, 0, 2).reshape(len(bits), -1)
        basis, _, initial_capacity = leading_completion(basis, response, protocol["density_response_directions"], maximum)
        del response
        feasible = None
    else:
        basis, feasible = checkpoint["basis"], checkpoint["reduced_ensemble"]
        initial_count = checkpoint["initial_canonical_components"]
        images = checkpoint.get("images", kinetic.new_zeros(len(bits), 0))
        initial_capacity = False
    beta = math.log(len(bits))/protocol["interacting_entropy_allowance_hartree"]
    potential = requested.new_zeros(len(requested))
    history, dual_history = [], []
    global_check = None
    status = "initial_capacity_limit" if initial_capacity else "running"
    for update in range(protocol["maximum_response_updates"]+1):
        if images.shape[1] < basis.shape[1]:
            images = torch.cat((images, torch.stack([action(vector) for vector in basis[:, images.shape[1]:].unbind(1)], 1)), 1)
        matrix = basis.T @ images
        hermiticity = float((matrix-matrix.T).norm())
        matrix = (matrix+matrix.T)/2
        generators = links.project(basis)
        reduced = links.reduced(moments, generators)
        support = basis.T @ lift["vectors"].T
        source_retention = float((basis @ support-lift["vectors"].T).norm())
        if source_retention > 1e-8 or hermiticity > 1e-7:
            raise RuntimeError(f"Physical response space failed retention/Hermiticity: {source_retention}, {hermiticity}")
        if feasible is None:
            feasible = (support*lift["weights"]) @ support.T
        else:
            embedded = matrix.new_zeros(matrix.shape)
            embedded[:len(feasible), :len(feasible)] = feasible
            feasible = embedded
        feasible_error = float((torch.einsum("ij,aij->a", feasible, reduced)-requested).norm())
        if feasible_error > protocol["density_residual_tolerance"]:
            raise RuntimeError(f"Known canonical ensemble lost density feasibility: {feasible_error}")
        if protocol.get("optimizer") == "positive_primal":
            affine = torch.cat((torch.eye(len(matrix), device=matrix.device)[None], reduced))
            affine_target = torch.cat((requested.new_ones(1), requested))
            def inner_progress(row):
                if callback: callback({"phase": "positive_reduced_ensemble", "update": update, **row})
            primal = optimize_ensemble(matrix, affine, affine_target, feasible,
                                        maximum_iterations=protocol["maximum_primal_iterations"],
                                        feasibility_tolerance=protocol["primal_feasibility_tolerance"],
                                        gap_tolerance=protocol["primal_dual_gap_tolerance_hartree"], callback=inner_progress)
            if not primal["converged"]:
                feasible = primal.get("best_feasible_ensemble", feasible)
                status = "positive_primal_limit_kept_feasible"
                row = {"update": update, "dimension": len(matrix), "status": status,
                       "density_residual": primal["density_residual"], "reduced_primal_dual_gap_hartree": primal["gap"],
                       "universal_energy_hartree": float((feasible*matrix).sum())}
                history.append(row); dual_history.append(primal["history"])
                if callback: callback(row)
                break
            population, rotation = torch.linalg.eigh(primal["ensemble"])
            small_potential = primal["potential"][1:]
            values, effective_vectors = torch.linalg.eigh(matrix+torch.einsum("a,aij->ij", small_potential, reduced))
            solved = {"population": population.clamp_min(0), "vectors": rotation,
                      "potential": small_potential, "values": values, "effective_vectors": effective_vectors,
                      "residual": primal["density_residual"], "history": primal["history"]}
        else:
            solved = ensemble_dual(matrix, reduced, requested, beta, initial=potential,
                                   tolerance=protocol["inner_dual_tolerance"])
            solved["effective_vectors"] = solved["vectors"]
        dual_history.append(solved["history"])
        if solved["residual"] > protocol["density_residual_tolerance"]:
            status = "density_dual_failed_kept_feasible"
            row = {"update": update, "dimension": basis.shape[1], "density_residual": feasible_error,
                   "rejected_dual_density_residual": solved["residual"], "source_retention_error": source_retention,
                   "universal_energy_hartree": float((feasible*matrix).sum()), "status": status}
            history.append(row)
            if callback: callback(row)
            break
        potential = solved["potential"]
        feasible = (solved["vectors"]*solved["population"]) @ solved["vectors"].T
        if checkpoint_callback:
            checkpoint_callback({"basis": basis, "images": images, "reduced_ensemble": feasible,
                                 "initial_canonical_components": initial_count,
                                 "interacting_potential": potential, "history": history})
        root = solved["vectors"]*solved["population"].sqrt()
        external_dual = torch.einsum("a,aij->ij", potential, moments)
        effective_images = images+links.image(external_dual, basis)
        small_effective = basis.T @ effective_images
        residual = (effective_images-basis @ small_effective) @ root
        stationarity = float(residual.norm())
        row = {"update": update, "dimension": basis.shape[1], "density_residual": solved["residual"],
               "weighted_stationarity_residual": stationarity, "source_retention_error": source_retention,
               "universal_energy_hartree": float((feasible*matrix).sum()),
               "ensemble_purity": float(feasible.square().sum()), "status": "density_feasible"}
        history.append(row)
        if callback: callback(row)
        if stationarity < protocol["stationarity_residual_tolerance_hartree"]:
            generator = torch.Generator(device=density.device).manual_seed(protocol["global_check_seed"])
            orbitals, _ = torch.linalg.qr(torch.randn(len(density), number//2, device=density.device, generator=generator), mode="reduced")
            seed = basis @ solved["effective_vectors"][:, 0]+.01*slater_vector(orbitals, torch.arange(number//2, device=density.device), bits)
            ground = incremental_ritz(lambda value: singlet(action(value)+links.image(external_dual, value)), seed,
                                       maximum=protocol["global_check_ritz_dimension"],
                                       tolerance=protocol["global_check_residual_tolerance"], projector=singlet)
            gap = float(ground["energy"]-solved["values"][0])
            global_check = {"converged": ground["converged"], "residual": ground["residual"],
                            "full_minus_projected_lowest_hartree": gap}
            if ground["converged"] and abs(gap) < protocol["global_check_energy_tolerance_hartree"]:
                status = "stationary_with_lowest_state_check"
                break
            status = "global_lowest_state_check_failed"
            break
        if update == protocol["maximum_response_updates"]:
            status = "response_update_limit_kept_feasible"
            break
        old_count = basis.shape[1]
        basis, added, capacity = leading_completion(basis, residual, protocol["stationarity_response_directions"], maximum)
        if capacity:
            status = "response_space_capacity_kept_feasible"
            break
        active = solved["population"].argsort(descending=True)[:protocol["maximum_density_response_states"]]
        states = (basis[:, :old_count] @ solved["vectors"][:, active])*solved["population"][active].sqrt()
        source = links.response_images(moments, states).permute(1, 0, 2).reshape(len(bits), -1)
        basis, density_added, capacity = leading_completion(basis, source, protocol["density_response_directions"], maximum)
        del source, residual, effective_images
        row["added_stationarity_directions"], row["added_density_directions"] = added, density_added
        if capacity:
            status = "response_space_capacity_kept_feasible"
            # The last accepted ensemble remains in the unchanged old prefix.
            basis = basis[:, :old_count]
            break
        if basis.shape[1] == old_count:
            status = "no_independent_response_kept_feasible"
            break
    # If completion stopped before projecting an expanded space, retain the
    # last feasible prefix rather than attach an ensemble to the wrong basis.
    basis = basis[:, :len(feasible)]
    matrix = matrix[:len(feasible), :len(feasible)]
    generators = generators[:, :len(feasible), :len(feasible)]
    population = torch.linalg.eigvalsh((feasible+feasible.T)/2).clamp_min(0)
    entropy_term = (population*population.clamp_min(1e-300).log()).sum()/beta
    universal_energy = (feasible*matrix).sum()
    upper_functional = (universal_energy if protocol.get("optimizer") == "positive_primal" else
                        universal_energy+entropy_term+protocol["interacting_entropy_allowance_hartree"])
    output_density = links.density(feasible, generators)
    final_density_error = float((torch.einsum("aij,ij->a", operators, output_density)-target).norm())
    spin_norm = max(float(singlet.spin_squared(vector).norm()) for vector in basis.unbind(1))
    kinetic_beta = 2*len(density)*math.log(2)/protocol["kinetic_entropy_allowance_hartree"]
    effective = (natural*((2/occupations.clamp(1e-14, 2-1e-14)-1).log()/kinetic_beta)) @ natural.T
    kinetic_initial = torch.einsum("aij,ij->a", operators, effective-kinetic)
    noninteracting = ensemble_dual(kinetic, operators, target, kinetic_beta, statistics="onebody", initial=kinetic_initial)
    hartree_matrix = torch.einsum("ijkl,kl->ij", eri, density)
    hartree = .5*(hartree_matrix*density).sum()
    physical_valid = final_density_error < protocol["density_residual_tolerance"] and spin_norm < 1e-8 and float((feasible.trace()-1).abs()) < 1e-8
    accepted = physical_valid and noninteracting["residual"] < 1e-8 and status == "stationary_with_lowest_state_check"
    xc_upper = upper_functional-noninteracting["dual"]-hartree
    xc_potential = torch.einsum("a,aij->ij", noninteracting["potential"][1:]-potential, moments)-hartree_matrix if accepted else None
    return {"status": status, "converged": accepted, "feasible_ensemble_retained": physical_valid,
            "xc_energy": xc_upper if accepted else None, "xc_potential": xc_potential,
            "variational_xc_upper_estimate": xc_upper, "universal_upper_functional": upper_functional,
            "universal_energy": universal_energy, "density_residual": final_density_error,
            "spin_squared_basis_norm": spin_norm, "ensemble_trace_error": float((feasible.trace()-1).abs()),
            "minimum_reduced_ensemble_eigenvalue": float(torch.linalg.eigvalsh(feasible).min()),
            "noninteracting_density_residual": noninteracting["residual"], "global_lowest_state_check": global_check,
            "basis": basis, "reduced_ensemble": feasible, "interacting_potential": potential,
            "density": output_density, "history": history, "dual_history": dual_history,
            "initial_canonical_components": initial_count, "occupation_link_bytes": links.storage_bytes(),
            "entropy_allowance_hartree": protocol["interacting_entropy_allowance_hartree"],
            "limits": ["Implicit fixed-basis ensemble search", "Global Fock vectors and supplied spatial frame",
                       "No learned predictor or many-fragment closure claim", "Potential requires separate density-path validation"]}
