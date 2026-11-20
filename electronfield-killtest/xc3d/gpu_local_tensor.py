"""Literal local operators in an exact-number/total-spin tensor section.

Local positive ensembles are physical global density operators. The overlap
metric is resolved by a direct physical embedding SVD, not an inverse of a
squared ill-conditioned Gram matrix. No occupation vector is constructed.
This is numerical tensor algebra supporting our density-functional model.
"""
import torch
from .gpu_spin_chain import transitions
from .gpu_fermion_mpo import _transfer


class LocalSpinBasis:
    def __init__(self, left, right, device):
        self.left, self.right, self.device = left, right, device
        lookup = {label: index for index, label in enumerate(right)}
        left_sectors = sorted(set(label[:3] for label in left))
        right_channels = {}
        for number, spin, channel, _ in right:
            right_channels.setdefault((number, spin), set()).add(channel)
        edges, index, parameter, numerators, denominators, signs = [], [], [], [], [], []
        for number, spin, channel in left_sectors:
            for physical, new_number, new_spin in transitions(number, spin):
                for target_channel in sorted(right_channels.get((new_number, new_spin), [])):
                    edge = len(edges)
                    edges.append((number, spin, channel, physical, new_number, new_spin, target_channel))
                    for li, (ln, ls, lc, magnetic) in enumerate(left):
                        if (ln, ls, lc) != (number, spin, channel): continue
                        if physical in (0, 2):
                            target = (new_number, new_spin, target_channel, magnetic)
                            if target not in lookup: continue
                            ri = lookup[target]
                            index.append((li*4+(0 if physical == 0 else 3))*len(right)+ri)
                            parameter.append(edge); numerators.append(1); denominators.append(1); signs.append(1)
                        else:
                            for local, local_m in ((1, 1), (2, -1)):
                                target = (new_number, new_spin, target_channel, magnetic+local_m)
                                if target not in lookup: continue
                                if new_spin == spin+1:
                                    numerator = spin+magnetic+2 if local_m == 1 else spin-magnetic+2
                                    sign = 1
                                else:
                                    numerator = spin-magnetic if local_m == 1 else spin+magnetic
                                    sign = -1 if local_m == 1 else 1
                                if not numerator: continue
                                ri = lookup[target]
                                index.append((li*4+local)*len(right)+ri); parameter.append(edge)
                                numerators.append(numerator); denominators.append(2*(spin+1)); signs.append(sign)
        self.edges = edges
        self.index = torch.tensor(index, dtype=torch.long, device=device)
        self.parameter = torch.tensor(parameter, dtype=torch.long, device=device)
        self.coefficient = (torch.tensor(numerators, device=device)/torch.tensor(denominators, device=device)).sqrt()*torch.tensor(signs, device=device)
        self.column_norm_squared = self.coefficient.new_zeros(len(edges)).index_add(0, self.parameter, self.coefficient.square())
        if not edges or bool((self.column_norm_squared <= 0).any()):
            raise RuntimeError("The local physical tensor section is empty or contains a null intertwiner")

    @property
    def dimension(self): return len(self.edges)

    def vectors(self, parameters):
        if parameters.shape[0] != self.dimension: raise ValueError("Wrong local reduced-coordinate count")
        flat = parameters.new_zeros(len(self.left)*4*len(self.right), parameters.shape[1])
        flat.index_add_(0, self.index, self.coefficient[:, None]*parameters[self.parameter])
        return flat.reshape(len(self.left), 4, len(self.right), parameters.shape[1])

    def adjoint(self, vectors):
        flat = vectors.flatten(0, 2)
        return flat.new_zeros(self.dimension, flat.shape[1]).index_add(0, self.parameter, self.coefficient[:, None]*flat[self.index])

    def coordinates(self, core):
        vectors = core.permute(0, 1, 3, 2)
        values = self.adjoint(vectors)/self.column_norm_squared[:, None]
        relative = (self.vectors(values)-vectors).norm()/vectors.norm().clamp_min(1e-30)
        if float(relative) > 1e-9: raise RuntimeError(f"Source tensor is outside its declared number/spin section: {float(relative)}")
        return values

    def core(self, parameters):
        return self.vectors(parameters).permute(0, 1, 3, 2)


def norm_environments(cores, center):
    left = cores[0].new_ones(1, 1)
    for core in cores[:center]:
        left = torch.einsum("ab,adkr,bdks->rs", left, core.conj(), core)
    right = cores[0].new_ones(1, 1)
    for core in reversed(cores[center+1:]):
        right = torch.einsum("rs,adkr,bdks->ab", right, core.conj(), core)
    return (left+left.T)/2, (right+right.T)/2


@torch.no_grad()
def physical_boundary_factors(cores, center, null_tolerance=1e-14):
    """Direct physical factors; never square weak state singular values."""
    left = cores[0].new_ones(1, 1)
    for core in cores[:center]:
        matrix = torch.einsum("xl,ldar->xdar", left, core).flatten(0, 2)
        _, singular, vh = torch.linalg.svd(matrix, full_matrices=False)
        selected = singular > singular.max()*null_tolerance
        left = singular[selected, None]*vh[selected]
    right = cores[0].new_ones(1, 1)
    for core in reversed(cores[center+1:]):
        matrix = torch.einsum("ldar,rx->ldax", core, right).flatten(1)
        u, singular, _ = torch.linalg.svd(matrix, full_matrices=False)
        selected = singular > singular.max()*null_tolerance
        right = u[:, selected]*singular[selected]
    return left, right


def left_step(mpo, site, environment, core):
    values, rows = [], []
    for operator, input_, output, matrix in mpo.blocks[site]:
        selected = environment.index_select(0, input_)
        merged = torch.sparse.mm(matrix, selected.flatten(1)).reshape(len(output), core.shape[0], core.shape[0])
        parts = [_transfer(merged[start:start+mpo.row_chunk], core, mpo.operators[operator])
                 for start in range(0, len(output), mpo.row_chunk)]
        values.append(torch.cat(parts)); rows.append(output)
    fresh = core.new_zeros(mpo.bond_dimensions[site+1], core.shape[-1], core.shape[-1])
    return fresh.index_add(0, torch.cat(rows), torch.cat(values)) if values else fresh


def right_step(mpo, site, environment, core):
    values, rows = [], []
    for operator, input_, output, matrix in mpo.blocks[site]:
        selected = environment.index_select(0, output)
        merged = torch.sparse.mm(matrix.T, selected.flatten(1)).reshape(len(input_), core.shape[-1], core.shape[-1])
        parts = []
        for start in range(0, len(input_), mpo.row_chunk):
            first = torch.einsum("xrs,mcas->xmcar", merged[start:start+mpo.row_chunk], core)
            first = torch.einsum("dc,xmcar->xmdar", mpo.operators[operator], first)
            parts.append(torch.einsum("ldar,xmdar->xlm", core.conj(), first))
        values.append(torch.cat(parts)); rows.append(input_)
    fresh = core.new_zeros(mpo.bond_dimensions[site], core.shape[0], core.shape[0])
    return fresh.index_add(0, torch.cat(rows), torch.cat(values)) if values else fresh


@torch.no_grad()
def operator_environments(mpo, cores, center, maximum_environment_bytes=1_500_000_000):
    left = cores[0].new_ones(1, 1, 1)
    for site in range(center):
        required = mpo.bond_dimensions[site+1]*cores[site].shape[-1]**2*8
        if required > maximum_environment_bytes: raise RuntimeError("Left operator environment exceeds its declared cap")
        left = left_step(mpo, site, left, cores[site])
    right = cores[0].new_ones(1, 1, 1)
    for site in range(len(cores)-1, center, -1):
        required = mpo.bond_dimensions[site]*cores[site].shape[0]**2*8
        if required > maximum_environment_bytes: raise RuntimeError("Right operator environment exceeds its declared cap")
        right = right_step(mpo, site, right, cores[site])
    return left, right


def local_operator_image(mpo, site, left, right, vectors, channel_chunk=8):
    result = torch.zeros_like(vectors)
    for operator, input_, output, matrix in mpo.blocks[site]:
        merged = torch.sparse.mm(matrix, left[input_].flatten(1)).reshape(len(output), vectors.shape[0], vectors.shape[0])
        selected = right[output]
        for start in range(0, len(output), channel_chunk):
            first = torch.einsum("wab,bcsk->wacsk", merged[start:start+channel_chunk], vectors)
            first = torch.einsum("dc,wacsk->wadsk", mpo.operators[operator], first)
            result += torch.einsum("wts,wadsk->adtk", selected[start:start+channel_chunk], first)
    return result


@torch.no_grad()
def pullback_operator(mpo, state, center, basis, transform, batch=16, maximum_environment_bytes=1_500_000_000):
    left, right = operator_environments(mpo, state.cores, center, maximum_environment_bytes)
    result = transform.new_zeros(transform.shape[1], transform.shape[1])
    for start in range(0, transform.shape[1], batch):
        vectors = basis.vectors(transform[:, start:start+batch])
        image = local_operator_image(mpo, center, left, right, vectors)
        result[:, start:start+batch] = transform.T @ basis.adjoint(image)
    error = float((result-result.T).norm()/result.norm().clamp_min(1e-30))
    if error > 1e-8: raise RuntimeError(f"Literal local operator lost Hermiticity: {error}")
    return (result+result.T)/2


@torch.no_grad()
def physical_local_space(state, center, maximum_embedding_bytes=1_500_000_000, relative_null_tolerance=1e-12):
    core = state.cores[center]
    basis = LocalSpinBasis(state.labels[center], state.labels[center+1], core.device)
    left, right = physical_boundary_factors(state.cores, center)
    requested = left.shape[0]*4*right.shape[1]*basis.dimension*8
    if requested > maximum_embedding_bytes: raise RuntimeError(f"Local physical embedding exceeds its declared cap: {requested} bytes")
    embedding = core.new_zeros(left.shape[0]*4*right.shape[1], basis.dimension)
    for start in range(0, basis.dimension, 16):
        parameters = core.new_zeros(basis.dimension, min(16, basis.dimension-start))
        parameters[torch.arange(start, start+parameters.shape[1], device=core.device), torch.arange(parameters.shape[1], device=core.device)] = 1.
        vectors = basis.vectors(parameters)
        physical = torch.einsum("al,ldrk,rb->adbk", left, vectors, right)
        embedding[:, start:start+parameters.shape[1]] = physical.flatten(0, 2)
    _, singular, vh = torch.linalg.svd(embedding, full_matrices=False)
    retained = singular > singular.max()*relative_null_tolerance
    transform = vh[retained].T/singular[retained]
    inverse = singular[retained, None]*vh[retained]
    old = basis.coordinates(core)
    coefficients = inverse @ old
    retention_error = float((embedding @ (transform @ coefficients-old)).norm()/(embedding @ old).norm().clamp_min(1e-30))
    metric_error = float(((embedding @ transform).T @ (embedding @ transform)-torch.eye(transform.shape[1], device=core.device)).norm())
    norm_left, norm_right = norm_environments(state.cores, center)
    literal_metric = transform.new_zeros(transform.shape[1], transform.shape[1])
    for start in range(0, transform.shape[1], 16):
        vectors = basis.vectors(transform[:, start:start+16])
        image = torch.einsum("ab,bcsk,ts->actk", norm_left, vectors, norm_right)
        literal_metric[:, start:start+16] = transform.T @ basis.adjoint(image)
    literal_metric_error = float((literal_metric-torch.eye(len(literal_metric), device=core.device)).norm())
    if max(retention_error, metric_error, literal_metric_error) > 1e-8:
        raise RuntimeError(f"Local physical space lost input/metric consistency: {retention_error}, {metric_error}, {literal_metric_error}")
    ensemble = coefficients @ coefficients.T
    ensemble /= ensemble.trace()
    return {"basis": basis, "transform": transform, "inverse": inverse, "initial_ensemble": ensemble,
            "input_retention_error": retention_error, "metric_orthogonality_error": metric_error,
            "literal_metric_orthogonality_error": literal_metric_error,
            "reduced_coordinate_count": basis.dimension, "physical_dimension": transform.shape[1],
            "embedding_bytes": requested}
