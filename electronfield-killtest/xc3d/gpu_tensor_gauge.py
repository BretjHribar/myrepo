"""Exact singlet bond rebasing, with complete number/spin multiplets.

This changes tensor coordinates, not the represented physical state or
Hamiltonian. Direct physical factors resolve numerical null support. No
physical rank/seed/tolerance sweep or energy-error selection is involved.
"""
import torch
from .gpu_local_tensor import LocalSpinBasis, norm_environments
from .gpu_spin_response import SpinChain, overlap
from .gpu_purified_spin import purified_norm


@torch.no_grad()
def rebase_bond(state, cut, side, null_tolerance=1e-14):
    if state.twice_spin != 0: raise ValueError("This scalar-irrep gauge currently requires a total singlet")
    if not 0 < cut < len(state.cores) or side not in ("left", "right"):
        raise ValueError("Invalid bond or gauge direction")
    old = state.labels[cut]
    site = cut-1 if side == "left" else cut
    core = state.cores[site]
    left_norm, right_norm = norm_environments(state.cores, site)
    outer = left_norm if side == "left" else right_norm
    if float((outer-torch.eye(len(outer), device=outer.device)).norm()) > 1e-8:
        raise RuntimeError("Direct canonical rebasing requires the outer prefix/suffix to be canonical")
    basis = LocalSpinBasis(state.labels[site], state.labels[site+1], core.device)
    coordinates = basis.coordinates(core)
    groups = {}
    for index, (ln, ls, lc, physical, rn, rs, rc) in enumerate(basis.edges):
        quantum = (rn, rs) if side == "left" else (ln, ls)
        group = groups.setdefault(quantum, {"paths": {}, "channels": set(), "entries": []})
        for ancilla in range(core.shape[2]):
            path = (ln, ls, lc, physical, ancilla) if side == "left" else (physical, rn, rs, rc, ancilla)
            group["paths"].setdefault(path, len(group["paths"]))
            group["channels"].add(rc if side == "left" else lc)
            group["entries"].append((index, ancilla, path, rc if side == "left" else lc))
    labels, changes, discarded, decomposed = [], [], core.new_zeros(()), {}
    lookup = {label: index for index, label in enumerate(old)}
    for (number, spin), group in sorted(groups.items()):
        channels = sorted(group["channels"])
        channel_index = {value: index for index, value in enumerate(channels)}
        block = core.new_zeros(len(group["paths"]), len(channels)) if side == "left" else core.new_zeros(len(channels), len(group["paths"]))
        weights = core.new_ones(len(group["paths"]))
        if side == "right":
            for path, index in group["paths"].items():
                weights[index] = core.new_tensor((path[2]+1)/(spin+1)).sqrt()
        for index, ancilla, path, channel in group["entries"]:
            path_index = group["paths"][path]
            if side == "left": block[path_index, channel_index[channel]] = coordinates[index, ancilla]
            else: block[channel_index[channel], path_index] = coordinates[index, ancilla]*weights[path_index]
        u, singular, vh = torch.linalg.svd(block, full_matrices=False)
        retained = singular > singular.max()*null_tolerance
        if not bool(retained.any()): continue
        discarded += singular[~retained].square().sum()*(spin+1)
        if side == "left":
            values = u[:, retained]
            change = singular[retained, None]*vh[retained]
        else:
            values = vh[retained]/weights[None]
            change = u[:, retained]*singular[retained]
        offset = len(labels)
        labels.extend((number, spin, channel, magnetic) for channel in range(int(retained.sum())) for magnetic in range(-spin, spin+1, 2))
        changes.append((number, spin, channels, offset, change))
        decomposed[number, spin] = group["paths"], values
    if not labels: raise RuntimeError("Bond rebasing removed the entire physical support")
    left_change = core.new_zeros(len(old), len(labels))
    right_change = core.new_zeros(len(labels), len(old))
    for number, spin, channels, offset, change in changes:
        for mindex, magnetic in enumerate(range(-spin, spin+1, 2)):
            source = torch.tensor([lookup[number, spin, channel, magnetic] for channel in channels], device=core.device)
            count = change.shape[0] if side == "left" else change.shape[1]
            target = offset+torch.arange(count, device=core.device)*(spin+1)+mindex
            if side == "left": right_change[target[:, None], source[None]] = change
            else: left_change[source[:, None], target[None]] = change
    cores = list(state.cores)
    bonds = list(state.labels); bonds[cut] = labels
    new_basis = LocalSpinBasis(bonds[site], bonds[site+1], core.device)
    new_coordinates = core.new_zeros(new_basis.dimension, core.shape[2])
    for index, (ln, ls, lc, physical, rn, rs, rc) in enumerate(new_basis.edges):
        paths, values = decomposed[(rn, rs) if side == "left" else (ln, ls)]
        for ancilla in range(core.shape[2]):
            path = (ln, ls, lc, physical, ancilla) if side == "left" else (physical, rn, rs, rc, ancilla)
            new_coordinates[index, ancilla] = values[paths[path], rc] if side == "left" else values[lc, paths[path]]
    if side == "left":
        cores[cut-1] = new_basis.core(new_coordinates)
        cores[cut] = torch.einsum("sl,ldar->sdar", right_change, cores[cut])
    else:
        cores[cut-1] = torch.einsum("ldar,rs->ldas", cores[cut-1], left_change)
        cores[cut] = new_basis.core(new_coordinates)
    result = SpinChain(cores, bonds, state.electrons, state.twice_spin)
    norm_old, norm_new = purified_norm(state.cores), purified_norm(result.cores)
    cross = overlap(state, result)
    squared_difference = norm_old+norm_new-2*cross
    error = float(squared_difference.abs()/norm_old.clamp_min(1e-30))
    if error > 1e-10: raise RuntimeError(f"Physical state changed during bond rebasing: squared error {error}")
    return result, {"cut": cut, "side": side, "old_bond_dimension": len(old), "new_bond_dimension": len(labels),
                    "relative_squared_state_difference_identity": float(squared_difference/norm_old),
                    "discarded_boundary_squared_norm": float(discarded)}


@torch.no_grad()
def canonical_at(state, center, callback=None):
    if not 0 <= center < len(state.cores): raise ValueError("Invalid tensor center")
    history = []
    for cut in range(1, center+1):
        state, row = rebase_bond(state, cut, "left")
        history.append(row)
        if callback: callback(row)
    for cut in range(len(state.cores)-1, center, -1):
        state, row = rebase_bond(state, cut, "right")
        history.append(row)
        if callback: callback(row)
    return state, history
