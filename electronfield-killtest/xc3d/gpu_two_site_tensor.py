"""Two-site physical intertwiners with unrestricted intermediate N,S sectors.

Two half-spin CG maps give a complete local physical section. No seed
middle-bond charge/spin restriction is inherited. This supplies physical
response directions for our model; CG/MPO/tensor algebra is not a novelty
claim. All scientific coefficients and contractions remain CUDA float64.
"""
from types import SimpleNamespace
import torch
from .gpu_spin_chain import transitions
from .gpu_local_tensor import LocalSpinBasis, norm_environments, left_step, right_step, local_operator_image
from .gpu_four_particle import incremental_ritz
from .gpu_spin_response import SpinChain, overlap
from .gpu_purified_spin import purified_norm


def _half_factors(spin, magnetic, physical, new_spin):
    if physical in (0, 2):
        return [(0 if physical == 0 else 3, magnetic, 1, 1, 1)]
    result = []
    for local, change in ((1, 1), (2, -1)):
        if new_spin == spin+1:
            numerator, sign = (spin+magnetic+2 if change == 1 else spin-magnetic+2), 1
        else:
            numerator, sign = (spin-magnetic if change == 1 else spin+magnetic), (-1 if change == 1 else 1)
        if numerator:
            result.append((local, magnetic+change, numerator, 2*(spin+1), sign))
    return result


class TwoSiteSpinBasis:
    def __init__(self, left, right, device):
        self.left, self.right = left, right
        lookup = {label: index for index, label in enumerate(right)}
        right_channels = {}
        for number, spin, channel, _ in right: right_channels.setdefault((number, spin), set()).add(channel)
        paths, index, parameter, numerator, denominator, signs = [], [], [], [], [], []
        for ln, ls, lc in sorted(set(label[:3] for label in left)):
            for first, mn, ms in transitions(ln, ls):
                for second, rn, rs in transitions(mn, ms):
                    for rc in sorted(right_channels.get((rn, rs), [])):
                        entry = len(paths); paths.append((ln, ls, lc, first, mn, ms, second, rn, rs, rc))
                        for li, label in enumerate(left):
                            if label[:3] != (ln, ls, lc): continue
                            for d1, mm, n1, den1, sign1 in _half_factors(ls, label[3], first, ms):
                                for d2, rm, n2, den2, sign2 in _half_factors(ms, mm, second, rs):
                                    if (rn, rs, rc, rm) not in lookup: continue
                                    ri = lookup[rn, rs, rc, rm]
                                    index.append((li*16+d1*4+d2)*len(right)+ri); parameter.append(entry)
                                    numerator.append(n1*n2); denominator.append(den1*den2); signs.append(sign1*sign2)
        self.paths = paths
        self.index = torch.tensor(index, device=device)
        self.parameter = torch.tensor(parameter, device=device)
        self.coefficient = (torch.tensor(numerator, device=device)/torch.tensor(denominator, device=device)).sqrt()*torch.tensor(signs, device=device)
        self.norm_squared = self.coefficient.new_zeros(len(paths)).index_add(0, self.parameter, self.coefficient.square())
        if not paths or bool((self.norm_squared <= 0).any()): raise RuntimeError("The two-site physical section is empty")
        self.inverse_norm = self.norm_squared.rsqrt()

    @property
    def dimension(self): return len(self.paths)

    def vectors(self, values):
        result = values.new_zeros(len(self.left)*16*len(self.right), values.shape[1])
        result.index_add_(0, self.index, self.coefficient[:, None]*values[self.parameter])
        return result.reshape(len(self.left), 16, len(self.right), values.shape[1])

    def adjoint(self, values):
        flat = values.flatten(0, 2)
        return flat.new_zeros(self.dimension, flat.shape[1]).index_add(0, self.parameter, self.coefficient[:, None]*flat[self.index])

    def coordinates(self, merged):
        flat = merged.reshape(len(self.left), 16, len(self.right), 1)
        values = self.adjoint(flat)/self.norm_squared[:, None]
        error = float((self.vectors(values)-flat).norm()/flat.norm().clamp_min(1e-30))
        if error > 1e-9: raise RuntimeError(f"Merged tensor is outside its complete two-site section: {error}")
        return values[:, 0]

    def groups(self, parameters):
        groups = {}
        for index, (ln, ls, lc, p1, mn, ms, p2, rn, rs, rc) in enumerate(self.paths):
            group = groups.setdefault((mn, ms), {"rows": {}, "columns": {}, "entries": []})
            row, column = (ln, ls, lc, p1), (p2, rn, rs, rc)
            group["rows"].setdefault(row, len(group["rows"])); group["columns"].setdefault(column, len(group["columns"]))
            group["entries"].append((index, group["rows"][row], group["columns"][column]))
        for quantum, group in groups.items():
            matrix = parameters.new_zeros(len(group["rows"]), len(group["columns"]))
            entries = torch.tensor(group["entries"], device=parameters.device)
            matrix[entries[:, 1], entries[:, 2]] = parameters[entries[:, 0]]
            degeneracy = torch.tensor([column[2]+1 for column in group["columns"]], device=parameters.device)
            weights = (degeneracy/(quantum[1]+1)).sqrt()
            group["matrix"], group["weights"] = matrix, weights
        return groups


@torch.no_grad()
def merge_operator(mpo, center):
    """Exact two-site MPO contraction; all sparse products execute on CUDA."""
    first, second = mpo.blocks[center], mpo.blocks[center+1]
    operators, blocks = [], []
    for op1, input1, output1, matrix1 in first:
        rows1, cols1 = matrix1.indices()
        global_first = torch.sparse_coo_tensor(torch.stack((output1[rows1], input1[cols1])), matrix1.values(),
                                               (mpo.bond_dimensions[center+1], mpo.bond_dimensions[center]), device=matrix1.device).coalesce()
        for op2, input2, output2, matrix2 in second:
            rows2, cols2 = matrix2.indices()
            global_second = torch.sparse_coo_tensor(torch.stack((output2[rows2], input2[cols2])), matrix2.values(),
                                                    (mpo.bond_dimensions[center+2], mpo.bond_dimensions[center+1]), device=matrix2.device).coalesce()
            joined = torch.sparse.mm(global_second, global_first).coalesce()
            if not joined._nnz(): continue
            active = joined.values() != 0
            if not bool(active.any()): continue
            row, col = joined.indices()[:, active]
            output, out_inverse = torch.unique(row, return_inverse=True)
            input_, in_inverse = torch.unique(col, return_inverse=True)
            compact = torch.sparse_coo_tensor(torch.stack((out_inverse, in_inverse)), joined.values()[active],
                                               (len(output), len(input_)), device=matrix1.device).coalesce()
            blocks.append((len(operators), input_, output, compact))
            operators.append(torch.kron(mpo.operators[op1], mpo.operators[op2]))
    return SimpleNamespace(operators=torch.stack(operators), blocks=[blocks])


@torch.no_grad()
def pair_problem(mpo, state, center, maximum_environment_bytes=1_500_000_000):
    if state.twice_spin or any(core.shape[2] != 1 for core in state.cores):
        raise ValueError("The current sector-enrichment forward model requires a pure singlet")
    if not 0 <= center < len(state.cores)-1: raise ValueError("Invalid two-site center")
    left = state.cores[0].new_ones(1, 1, 1)
    for site in range(center):
        required = mpo.bond_dimensions[site+1]*state.cores[site].shape[-1]**2*8
        if required > maximum_environment_bytes: raise RuntimeError("Two-site left environment exceeds its declared cap")
        left = left_step(mpo, site, left, state.cores[site])
    right = state.cores[0].new_ones(1, 1, 1)
    for site in range(len(state.cores)-1, center+1, -1):
        required = mpo.bond_dimensions[site]*state.cores[site].shape[0]**2*8
        if required > maximum_environment_bytes: raise RuntimeError("Two-site right environment exceeds its declared cap")
        right = right_step(mpo, site, right, state.cores[site])
    if max(left.numel(), right.numel())*8 > maximum_environment_bytes: raise RuntimeError("Two-site operator environment exceeds its declared cap")
    norm_left, _ = norm_environments(state.cores, center)
    _, norm_right = norm_environments(state.cores, center+1)
    canonical_error = max(float((norm_left-torch.eye(len(norm_left), device=norm_left.device)).norm()),
                          float((norm_right-torch.eye(len(norm_right), device=norm_right.device)).norm()))
    if canonical_error > 1e-8: raise RuntimeError("Two-site outer physical environments must be canonical")
    basis = TwoSiteSpinBasis(state.labels[center], state.labels[center+2], state.cores[0].device)
    # Exact CG orthogonality is checked independently of the physical H.
    cg = torch.sparse_coo_tensor(torch.stack((basis.index, basis.parameter)), basis.coefficient,
                                 (len(basis.left)*16*len(basis.right), basis.dimension), device=norm_left.device).coalesce()
    gram = torch.sparse.mm(cg.T, cg).coalesce()
    row, column = gram.indices()
    offdiag = row != column
    if bool(offdiag.any()) and float(gram.values()[offdiag].abs().max()) > 1e-10:
        raise RuntimeError("Two-site CG paths are not orthogonal")
    merged = merge_operator(mpo, center)
    def action(value):
        vectors = basis.vectors((basis.inverse_norm*value)[:, None])
        image = local_operator_image(merged, 0, left, right, vectors)
        return basis.inverse_norm*basis.adjoint(image)[:, 0]
    raw = torch.einsum("ldm,mer->lder", state.cores[center][:, :, 0, :], state.cores[center+1][:, :, 0, :])
    parameters = basis.coordinates(raw)
    seed = parameters*basis.norm_squared.sqrt()
    if abs(float(seed.square().sum())-1.) > 1e-8: raise RuntimeError("Two-site section lost physical normalization")
    return {"basis": basis, "action": action, "seed": seed, "canonical_error": canonical_error,
            "merged_operator_blocks": len(merged.blocks[0])}


@torch.no_grad()
def split_pair(state, center, basis, parameters, relative_null_tolerance=1e-13,
               maximum_bond_dimension=None, degeneracy_tolerance=1e-8):
    groups = basis.groups(parameters)
    labels, decomposed = [], {}
    discarded = parameters.new_zeros(())
    choices = []
    for quantum, group in sorted(groups.items()):
        matrix = group["matrix"]*group["weights"][None]
        u, singular, vh = torch.linalg.svd(matrix, full_matrices=False)
        group["svd"] = u, singular, vh
        group["retained"] = []
        start = 0
        while start < len(singular) and bool(singular[start] > singular.max()*relative_null_tolerance):
            stop = start+1
            while stop < len(singular) and bool((singular[stop]-singular[start]).abs() <= degeneracy_tolerance*singular[start]):
                stop += 1
            choices.append((quantum, start, stop, (quantum[1]+1)*(stop-start), singular[start:stop].square().mean()))
            start = stop
    if not choices: raise RuntimeError("Two-site splitting removed the complete state")
    order = torch.argsort(torch.stack([entry[4] for entry in choices]), descending=True).tolist()
    used = 0
    for entry in order:
        quantum, start, stop, cost, _ = choices[entry]
        if maximum_bond_dimension is not None and used+cost > maximum_bond_dimension: continue
        groups[quantum]["retained"].extend(range(start, stop)); used += cost
    for quantum, group in sorted(groups.items()):
        u, singular, vh = group["svd"]
        retained = torch.zeros(len(singular), dtype=torch.bool, device=parameters.device)
        retained[group["retained"]] = True
        discarded += (quantum[1]+1)*singular[~retained].square().sum()
        if not bool(retained.any()): continue
        labels.extend((*quantum, channel, magnetic) for channel in range(int(retained.sum())) for magnetic in range(-quantum[1], quantum[1]+1, 2))
        decomposed[quantum] = group, u[:, retained], singular[retained, None]*vh[retained]/group["weights"][None]
    left_basis = LocalSpinBasis(state.labels[center], labels, parameters.device)
    right_basis = LocalSpinBasis(labels, state.labels[center+2], parameters.device)
    left_values = parameters.new_zeros(left_basis.dimension, 1)
    right_values = parameters.new_zeros(right_basis.dimension, 1)
    for index, (ln, ls, lc, physical, rn, rs, rc) in enumerate(left_basis.edges):
        group, left, _ = decomposed[rn, rs]
        left_values[index, 0] = left[group["rows"][ln, ls, lc, physical], rc]
    for index, (ln, ls, lc, physical, rn, rs, rc) in enumerate(right_basis.edges):
        group, _, right = decomposed[ln, ls]
        right_values[index, 0] = right[lc, group["columns"][physical, rn, rs, rc]]
    cores = list(state.cores); cores[center], cores[center+1] = left_basis.core(left_values), right_basis.core(right_values)
    bonds = list(state.labels); bonds[center+1] = labels
    if not labels: raise RuntimeError("The declared bond cap cannot retain a complete physical spin multiplet")
    return SpinChain(cores, bonds, state.electrons, state.twice_spin), {"middle_bond_dimension": len(labels),
        "discarded_squared_norm": float(discarded), "maximum_bond_dimension": maximum_bond_dimension}


@torch.no_grad()
def enrich_left_support(state, center, basis, target_parameters, maximum_bond_dimension,
                        maximum_added_magnetic_states, relative_null_tolerance=1e-11,
                        degeneracy_tolerance=1e-8):
    """Retain the old state and add physical residual Schmidt directions.

    The left prefix, including the first site of the pair, must be
    canonical. The target is normally the two-site Hamiltonian residual;
    no reference energy enters this selection. Full spin multiplets and
    degenerate singular blocks are retained together. The old right core
    is embedded with exact zero coefficients on all new channels.
    """
    old = state.labels[center+1]
    if len(old) > maximum_bond_dimension: raise RuntimeError("Existing bond exceeds the declared cap")
    budget = min(maximum_added_magnetic_states, maximum_bond_dimension-len(old))
    groups = basis.groups(target_parameters)
    old_basis = LocalSpinBasis(state.labels[center], old, target_parameters.device)
    old_values = old_basis.coordinates(state.cores[center])[:, 0]
    old_channels = {}
    for number, spin, channel, _ in old:
        old_channels.setdefault((number, spin), set()).add(channel)
    old_channels = {quantum: sorted(channels) for quantum, channels in old_channels.items()}
    entries, largest, canonical_error = [], target_parameters.new_zeros(()), 0.
    for quantum, group in groups.items():
        channels = old_channels.get(quantum, [])
        channel_lookup = {channel: index for index, channel in enumerate(channels)}
        retained = target_parameters.new_zeros(len(group["rows"]), len(channels))
        for index, (ln, ls, lc, physical, rn, rs, rc) in enumerate(old_basis.edges):
            if (rn, rs) == quantum:
                retained[group["rows"][ln, ls, lc, physical], channel_lookup[rc]] = old_values[index]
        if channels:
            error = float((retained.T @ retained-torch.eye(len(channels), device=retained.device)).norm())
            canonical_error = max(canonical_error, error)
        residual = group["matrix"]*group["weights"][None]
        for _ in range(2): residual -= retained @ (retained.T @ residual)
        u, singular, _ = torch.linalg.svd(residual, full_matrices=False)
        if singular.numel(): largest = torch.maximum(largest, singular[0])
        group.update({"old_channels": channels, "old_left": retained, "new_left": u, "singular": singular})
    if canonical_error > 1e-8: raise RuntimeError(f"Enrichment needs a canonical existing left support: {canonical_error}")
    threshold = largest*relative_null_tolerance
    for quantum, group in groups.items():
        singular = group["singular"]
        start = 0
        while start < len(singular) and bool(singular[start] > threshold):
            stop = start+1
            while stop < len(singular) and bool(singular[stop] > threshold) and bool((singular[stop]-singular[start]).abs() <= degeneracy_tolerance*singular[start]):
                stop += 1
            weight = (quantum[1]+1)*singular[start:stop].square().sum()
            entries.append({"quantum": quantum, "start": start, "stop": stop,
                            "cost": (quantum[1]+1)*(stop-start), "weight": weight})
            start = stop
    order = torch.argsort(torch.stack([entry["weight"] for entry in entries]), descending=True).tolist() if entries else []
    selected = {quantum: [] for quantum in groups}
    used, captured = 0, 0.
    for index in order:
        entry = entries[index]
        if used+entry["cost"] > budget: continue
        selected[entry["quantum"]].extend(range(entry["start"], entry["stop"]))
        used += entry["cost"]; captured += float(entry["weight"])
    labels, left_blocks, remap = [], {}, {}
    for quantum, group in sorted(groups.items()):
        new = group["new_left"][:, selected[quantum]]
        old_left = group["old_left"]
        # Reorthogonalize only newly added directions; old support is copied.
        for _ in range(2): new -= old_left @ (old_left.T @ new)
        if new.shape[1]:
            u, _, vh = torch.linalg.svd(new, full_matrices=False)
            new = u @ vh
        values = torch.cat((old_left, new), 1)
        if values.shape[1] == 0: continue
        if float((values.T @ values-torch.eye(values.shape[1], device=values.device)).norm()) > 1e-8:
            raise RuntimeError("Added physical directions are not orthogonal to the retained state")
        for index, channel in enumerate(group["old_channels"]): remap[(*quantum, channel)] = index
        labels.extend((*quantum, channel, magnetic) for channel in range(values.shape[1])
                      for magnetic in range(-quantum[1], quantum[1]+1, 2))
        left_blocks[quantum] = values
    if len(labels) != len(old)+used: raise RuntimeError("Sector enrichment lost existing magnetic states")
    new_basis = LocalSpinBasis(state.labels[center], labels, target_parameters.device)
    values = target_parameters.new_zeros(new_basis.dimension, 1)
    for index, (ln, ls, lc, physical, rn, rs, rc) in enumerate(new_basis.edges):
        values[index, 0] = left_blocks[rn, rs][groups[rn, rs]["rows"][ln, ls, lc, physical], rc]
    lookup = {label: index for index, label in enumerate(labels)}
    source = torch.tensor([lookup[number, spin, remap[number, spin, channel], magnetic]
                           for number, spin, channel, magnetic in old], device=values.device)
    cores = list(state.cores)
    cores[center] = new_basis.core(values)
    cores[center+1] = cores[center+1].new_zeros(len(labels), *cores[center+1].shape[1:])
    cores[center+1][source] = state.cores[center+1]
    bonds = list(state.labels); bonds[center+1] = labels
    result = SpinChain(cores, bonds, state.electrons, state.twice_spin)
    norm_old, norm_new = purified_norm(state.cores), purified_norm(result.cores)
    state_error = float((norm_old+norm_new-2*overlap(state, result)).abs()/norm_old.clamp_min(1e-30))
    if state_error > 1e-10: raise RuntimeError("Sector enrichment changed the retained physical state")
    total = float(torch.stack([entry["weight"] for entry in entries]).sum()) if entries else 0.
    return result, {"old_bond_dimension": len(old), "new_bond_dimension": len(labels),
                    "added_magnetic_states": used, "maximum_bond_dimension": maximum_bond_dimension,
                    "old_left_canonical_error": canonical_error, "relative_squared_retained_state_error": state_error,
                    "captured_new_direction_squared_norm": captured, "available_new_direction_squared_norm": total,
                    "new_direction_fraction": captured/total if total else 0.,
                    "added_multiplets": [{"number": quantum[0], "twice_spin": quantum[1], "channels": len(indices)}
                                         for quantum, indices in sorted(selected.items()) if indices]}


@torch.no_grad()
def prefix_number_probability(state, cut, maximum_number):
    """Physical probability of a prefix charge, contracting both sides."""
    left = state.cores[0].new_ones(1, 1)
    for core in state.cores[:cut]: left = torch.einsum("ab,adkr,bdks->rs", left, core.conj(), core)
    right = state.cores[0].new_ones(1, 1)
    for core in reversed(state.cores[cut:]): right = torch.einsum("rs,adkr,bdks->ab", right, core.conj(), core)
    mask = torch.tensor([label[0] <= maximum_number for label in state.labels[cut]], device=left.device)
    return float((left*right)[mask].sum()/purified_norm(state.cores))
