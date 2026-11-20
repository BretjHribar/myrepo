"""Literal physical forward search with changing intermediate N,S sectors.

Two-site Ritz search, spin-complete Schmidt retraction and canonical tensor
gauges are established numerical methods. They support the research state;
they are not our claimed novelty, a learned XC model or a global-minimum
certificate. A fixed resource cap is declared before a chemistry run.
Rejected energy-increasing retractions remain explicit in its record.
"""
import torch
from .gpu_two_site_tensor import pair_problem, split_pair
from .gpu_four_particle import incremental_ritz
from .gpu_tensor_gauge import canonical_at
from .physical_tensor_forward import payload, restore, physical_checks
from .physical_ritz_seed import mixed_physical_start


@torch.no_grad()
def pair_minimum(operator, state, center, protocol, problem_builder=None):
    canonical, _ = canonical_at(state, center)
    builder = pair_problem if problem_builder is None else problem_builder
    problem = builder(operator, canonical, center, protocol["maximum_environment_bytes"])
    start_policy = protocol.get("ritz_start_policy", "retained_only")
    if start_policy == "retained_only":
        start, start_details = problem["seed"], {"policy": "original_retained_only"}
    elif start_policy == "fixed_generic_physical_component":
        start, start_details = mixed_physical_start(problem["seed"], protocol["ritz_probe_seed"]+center)
    else:
        raise ValueError("Unknown physical Krylov start policy")
    minimum = incremental_ritz(problem["action"], start,
        maximum=min(protocol["ritz_maximum_dimension"], problem["basis"].dimension),
        tolerance=protocol["ritz_residual_tolerance_hartree"], log_every=protocol.get("ritz_log_every", 100))
    minimum["start_details"] = start_details
    if not minimum["converged"]:
        raise RuntimeError(f"Two-site physical eigenproblem exhausted its declared capacity at center {center}: {minimum['residual']}")
    old_energy = problem["seed"] @ problem["action"](problem["seed"])
    gap = float(old_energy-minimum["energy"])
    if gap < -protocol["physical_identity_tolerance_hartree"]:
        raise RuntimeError("A two-site variational minimum lies above its retained input")
    return canonical, problem, minimum, float(old_energy), gap


@torch.no_grad()
def update_pair(operator, state, center, protocol, problem_builder=None):
    canonical, problem, minimum, old_energy, gap = pair_minimum(operator, state, center, protocol, problem_builder)
    candidate, split = split_pair(canonical, center, problem["basis"], minimum["state"]*problem["basis"].inverse_norm,
                                 maximum_bond_dimension=protocol["maximum_bond_dimension"])
    candidate = candidate.normalized()
    actual = float(operator.expectation(candidate.cores))
    accepted = actual <= old_energy+protocol["physical_identity_tolerance_hartree"]
    if not accepted: candidate = canonical
    return candidate, {"center": center, "physical_dimension": problem["basis"].dimension,
        "pair_minimum_hartree": float(minimum["energy"]), "untruncated_variational_gap_hartree": gap,
        "local_ritz_residual_hartree": minimum["residual"], "local_ritz_dimension": minimum["dimension"],
        "ritz_start_details": minimum["start_details"],
        "old_electronic_energy_hartree": old_energy, "retracted_electronic_energy_hartree": actual,
        "electronic_energy_hartree": actual if accepted else old_energy,
        "accepted": accepted, "energy_lowering_hartree": old_energy-actual if accepted else 0.,
        "rejection_reason": None if accepted else "Spin-complete bounded retraction raises literal physical energy",
        "split": split, "outer_metric_error": problem["canonical_error"]}


@torch.no_grad()
def solve_two_site(operator, initial, protocol, callback=None, checkpoint_callback=None, checkpoint=None, problem_builder=None):
    sites = len(initial.cores)
    if checkpoint is None:
        state = initial.normalized()
        density, checks = physical_checks(state, protocol["physical_tolerance"])
        if not checks["passed"]: raise RuntimeError("Initial two-site state failed its physical checks")
        energy = float(operator.expectation(state.cores))
        run = {"stage": "sweep", "sweep": 0, "position": 0, "state": payload(state), "energy": energy,
               "sweep_start_energy": energy, "sweep_start_density": density, "history": [], "sweep_history": [], "audit": []}
    else:
        run = checkpoint; state = restore(run["state"])
    def save():
        run["state"] = payload(state)
        if checkpoint_callback: checkpoint_callback(run)
    def emit(row):
        if callback: callback(row)
    while run["stage"] != "completed":
        if run["stage"] == "sweep":
            order = list(range(sites-1)) if run["sweep"] % 2 == 0 else list(range(sites-2, -1, -1))
            for position in range(run["position"], len(order)):
                center = order[position]
                state, row = update_pair(operator, state, center, protocol, problem_builder)
                if row["electronic_energy_hartree"] > run["energy"]+protocol["physical_identity_tolerance_hartree"]:
                    raise RuntimeError("Two-site update does not retain its preceding physical energy")
                row.update({"sweep": run["sweep"], "position": position})
                run["energy"] = row["electronic_energy_hartree"]
                run["position"] = position+1; run["history"].append(row)
                save(); emit({"phase": "two_site_update", **row})
            density, checks = physical_checks(state, protocol["physical_tolerance"])
            if not checks["passed"]: raise RuntimeError("Two-site sweep failed number/spin/occupation checks")
            energy_change = run["sweep_start_energy"]-run["energy"]
            density_change = float((density-run["sweep_start_density"]).norm())
            row = {"sweep": run["sweep"], "electronic_energy_hartree": run["energy"],
                   "energy_lowering_hartree": energy_change, "density_change": density_change, "physical_checks": checks,
                   "maximum_bond_dimension": max(len(bond) for bond in state.labels)}
            run["sweep_history"].append(row); emit({"phase": "two_site_sweep_completed", **row})
            stable = (energy_change < protocol["sweep_energy_tolerance_hartree"] and density_change < protocol["sweep_density_tolerance"])
            if stable or run["sweep"]+1 >= protocol["maximum_sweeps"]:
                run["stage"], run["position"], run["audit"] = "audit", 0, []
            else:
                run["sweep"] += 1; run["position"] = 0
                run["sweep_start_energy"], run["sweep_start_density"] = run["energy"], density
            save()
        else:
            # Canonical transformations below preserve the same frozen
            # physical state. None of these audit minima is installed.
            for center in range(run["position"], sites-1):
                _, problem, minimum, old_energy, gap = pair_minimum(operator, state, center, protocol, problem_builder)
                if abs(old_energy-run["energy"]) > protocol["physical_identity_tolerance_hartree"]:
                    raise RuntimeError("Frozen physical state changed during its common-context audit")
                row = {"center": center, "gap_hartree": gap, "physical_dimension": problem["basis"].dimension,
                       "local_ritz_residual_hartree": minimum["residual"], "local_minimum_hartree": float(minimum["energy"])}
                run["audit"].append(row); run["position"] = center+1
                save(); emit({"phase": "two_site_common_context_audit", **row})
            maximum_gap = max(row["gap_hartree"] for row in run["audit"])
            if maximum_gap < protocol["common_context_gap_tolerance_hartree"] or run["sweep"]+1 >= protocol["maximum_sweeps"]:
                run["stage"] = "completed"
                run["coordinate_stationarity_passed"] = maximum_gap < protocol["common_context_gap_tolerance_hartree"]
            else:
                run["sweep"] += 1; run["stage"], run["position"] = "sweep", 0
                run["sweep_start_energy"] = run["energy"]
                run["sweep_start_density"] = physical_checks(state, protocol["physical_tolerance"])[0]
            save()
    density, checks = physical_checks(state, protocol["physical_tolerance"])
    return {"state": state, "density": density, "electronic_energy_hartree": run["energy"],
            "physical_checks": checks, "coordinate_stationarity_passed": run["coordinate_stationarity_passed"],
            "maximum_common_context_gap_hartree": max(row["gap_hartree"] for row in run["audit"]),
            "history": run["history"], "sweep_history": run["sweep_history"], "audit": run["audit"],
            "full_hilbert_residual_certified": False, "global_variational_minimum_certified": False}
