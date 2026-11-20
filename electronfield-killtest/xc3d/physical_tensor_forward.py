"""Joint auxiliary/physical forward search by exact auxiliary elimination.

Minimizing x in Tr Gamma H0+||q-x||^2/(2 eps)+v.x+eps||v||^2/2
recovers Tr Gamma(H0+v.Q) for every physical Gamma. Thus the forward
calculation can update physical tensors with the literal Hamiltonian.
The universal conditional evaluator remains a separate T,W,Q interface.

Local variational updates and their final common-context audit certify
coordinate stationarity in the supplied tensor sectors, not a global
minimum, a full Hilbert residual, or chemical accuracy. Tensor algorithms
and this square-completion identity are established numerical mathematics.
"""
import torch
from .gpu_local_tensor import physical_local_space, pullback_operator
from .gpu_spin_response import SpinChain
from .gpu_purified_spin import purified_norm
from .gpu_tensor_density import density_matrix
from .gpu_manybody_ops import spin_words
from .gpu_fermion_mpo import FermionMPO


def payload(state):
    return {"cores": state.cores, "labels": state.labels, "electrons": state.electrons, "twice_spin": state.twice_spin}


def restore(value):
    return SpinChain(value["cores"], value["labels"], value["electrons"], value["twice_spin"])


@torch.no_grad()
def local_physical_problem(operator, state, center, protocol):
    space = physical_local_space(state, center, maximum_embedding_bytes=protocol["maximum_embedding_bytes"],
                                  relative_null_tolerance=protocol["local_metric_null_tolerance"])
    rank = space["physical_dimension"]
    if rank*rank*8 > protocol["maximum_local_matrix_bytes"]:
        raise RuntimeError("Physical local Hamiltonian exceeds the declared matrix budget")
    matrix = pullback_operator(operator, state, center, space["basis"], space["transform"],
                               maximum_environment_bytes=protocol["maximum_environment_bytes"])
    coefficient = space["inverse"] @ space["basis"].coordinates(state.cores[center])
    density = coefficient @ coefficient.T
    density /= density.trace()
    values, vectors = torch.linalg.eigh(matrix)
    old_energy = (density*matrix).sum()
    gap = float(old_energy-values[0])
    if gap < -protocol["physical_identity_tolerance_hartree"]:
        raise RuntimeError("Physical local state lies below its literal variational eigenvalue")
    return space, matrix, density, values, vectors, gap


@torch.no_grad()
def update_site(operator, state, center, protocol):
    space, matrix, density, values, vectors, gap = local_physical_problem(operator, state, center, protocol)
    # A numerical ground-state degeneracy is resolved by retaining the old
    # state's largest weight in that subspace; no random/label selection.
    selected = values-values[0] <= protocol["ground_degeneracy_tolerance_hartree"]
    ground = vectors[:, selected]
    weights, rotation = torch.linalg.eigh(ground.T @ density @ ground)
    vector = ground @ rotation[:, -1]
    core = space["basis"].core((space["transform"] @ vector)[:, None])
    cores = list(state.cores); cores[center] = core
    updated = SpinChain(cores, state.labels, state.electrons, state.twice_spin).normalized()
    expected = vector @ matrix @ vector
    actual = operator.expectation(updated.cores)
    error = float((actual-expected).abs())
    if error > protocol["physical_identity_tolerance_hartree"]:
        raise RuntimeError(f"Local update differs from its literal global Hamiltonian expectation: {error}")
    return updated, {"center": center, "physical_dimension": space["physical_dimension"],
                     "local_variational_gap_hartree": gap, "electronic_energy_hartree": float(actual),
                     "local_global_energy_identity_error_hartree": error,
                     "metric_error": space["literal_metric_orthogonality_error"],
                     "input_retention_error": space["input_retention_error"]}


@torch.no_grad()
def physical_checks(state, tolerance=1e-7):
    density = density_matrix(state)
    occupations = torch.linalg.eigvalsh(density)
    words, coefficient = spin_words(len(state.cores), density.device)
    transverse = FermionMPO(words, coefficient, len(state.cores)).expectation(state.cores)
    spin = transverse+(state.twice_spin/2)**2
    expected_spin = (state.twice_spin/2)*(state.twice_spin/2+1)
    number_error = float((density.trace()-state.electrons).abs())
    spin_error = float((spin-expected_spin).abs())
    norm_error = float((purified_norm(state.cores)-1).abs())
    valid = (max(number_error, spin_error, norm_error) < tolerance and float(occupations.min()) > -tolerance
             and float(occupations.max()) < 2+tolerance)
    return density, {"passed": valid, "number_error": number_error, "spin_squared": float(spin),
                     "target_spin_squared": expected_spin, "spin_error": spin_error, "norm_error": norm_error,
                     "minimum_spatial_occupation": float(occupations.min()), "maximum_spatial_occupation": float(occupations.max())}


@torch.no_grad()
def solve_forward(operator, initial, protocol, callback=None, checkpoint_callback=None, checkpoint=None):
    sites = len(initial.cores)
    if checkpoint is None:
        state = initial.normalized()
        energy = float(operator.expectation(state.cores))
        density, checks = physical_checks(state, protocol["physical_tolerance"])
        if not checks["passed"]: raise RuntimeError("Initial physical tensor state failed its domain checks")
        run = {"stage": "sweep", "sweep": 0, "position": 0, "state": payload(state), "energy": energy,
               "sweep_start_energy": energy, "sweep_start_density": density, "history": [], "sweep_history": [],
               "audit": [], "maximum_update_drop": 0.}
    else:
        run = checkpoint
        state = restore(run["state"])
    def save():
        run["state"] = payload(state)
        if checkpoint_callback: checkpoint_callback(run)
    while run["stage"] != "completed":
        if run["stage"] == "sweep":
            order = list(range(sites)) if run["sweep"] % 2 == 0 else list(range(sites-1, -1, -1))
            for position in range(run["position"], sites):
                center = order[position]
                previous = run["energy"]
                state, row = update_site(operator, state, center, protocol)
                if row["electronic_energy_hartree"] > previous+protocol["physical_identity_tolerance_hartree"]:
                    raise RuntimeError("Physical tensor energy increased in an exact variational local update")
                row.update({"sweep": run["sweep"], "position": position})
                run["energy"] = row["electronic_energy_hartree"]
                run["maximum_update_drop"] = max(run["maximum_update_drop"], previous-run["energy"])
                run["position"] = position+1; run["history"].append(row)
                save()
                if callback: callback({"phase": "physical_site_update", **row})
            density, checks = physical_checks(state, protocol["physical_tolerance"])
            if not checks["passed"]: raise RuntimeError(f"Physical sweep failed number/spin/Pauli checks: {checks}")
            density_change = float((density-run["sweep_start_density"]).norm())
            energy_change = run["sweep_start_energy"]-run["energy"]
            row = {"sweep": run["sweep"], "electronic_energy_hartree": run["energy"],
                   "energy_lowering_hartree": energy_change, "maximum_site_lowering_hartree": run["maximum_update_drop"],
                   "density_change": density_change, "physical_checks": checks}
            run["sweep_history"].append(row)
            if callback: callback({"phase": "physical_sweep_completed", **row})
            stationary = (energy_change < protocol["sweep_energy_tolerance_hartree"]
                           and run["maximum_update_drop"] < protocol["sweep_energy_tolerance_hartree"]
                           and density_change < protocol["sweep_density_tolerance"])
            if stationary or run["sweep"]+1 >= protocol["maximum_sweeps"]:
                run["stage"], run["position"], run["audit"] = "audit", 0, []
                run["capacity_reached"] = not stationary
            else:
                run["sweep"] += 1; run["position"] = 0
                run["sweep_start_energy"], run["sweep_start_density"] = run["energy"], density
                run["maximum_update_drop"] = 0.
            save()
        else:
            # Every audit uses the same final state; no tensor is changed.
            for center in range(run["position"], sites):
                space, _, _, values, _, gap = local_physical_problem(operator, state, center, protocol)
                row = {"center": center, "gap_hartree": gap, "physical_dimension": space["physical_dimension"],
                       "local_minimum_hartree": float(values[0])}
                run["audit"].append(row); run["position"] = center+1
                save()
                if callback: callback({"phase": "common_context_audit", **row})
            maximum_gap = max(row["gap_hartree"] for row in run["audit"])
            if maximum_gap < protocol["common_context_gap_tolerance_hartree"] or run["sweep"]+1 >= protocol["maximum_sweeps"]:
                run["stage"] = "completed"
                run["coordinate_stationarity_passed"] = maximum_gap < protocol["common_context_gap_tolerance_hartree"]
            else:
                run["sweep"] += 1; run["stage"], run["position"] = "sweep", 0
                run["sweep_start_energy"] = run["energy"]
                run["sweep_start_density"] = density_matrix(state)
                run["maximum_update_drop"] = 0.
            save()
    density, checks = physical_checks(state, protocol["physical_tolerance"])
    return {"state": state, "density": density, "physical_checks": checks, "electronic_energy_hartree": run["energy"],
            "coordinate_stationarity_passed": run["coordinate_stationarity_passed"], "history": run["history"],
            "sweep_history": run["sweep_history"], "audit": run["audit"],
            "maximum_common_context_gap_hartree": max(row["gap_hartree"] for row in run["audit"]),
            "full_hilbert_residual_certified": False, "global_variational_minimum_certified": False}
