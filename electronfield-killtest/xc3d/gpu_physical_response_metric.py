"""Physical-norm Hamiltonian response to a product determinant.

This replaces the arbitrary MPO-channel norm used in one-pass zip-up by a
right canonical physical frame. Sparse CUDA Gram contraction also measures
the full image norm without any global occupation vector. Tensor SVD and
Schmidt metrics are established mathematics, not the proposed novelty.

This implementation deliberately requires a product source. Extending its
exact Gram to an arbitrary correlated source has a different memory cost;
do not infer a certified larger-system ground-state residual from this test.
"""
import torch
from .gpu_spin_chain import quantum_bonds
from .gpu_spin_response import SpinChain, _isometry, _project_endpoint
from .gpu_purified_spin import purified_norm


@torch.no_grad()
def product_image_cores(mpo, source):
    if any(tuple(core.shape) != (1, 4, 1, 1) for core in source.cores):
        raise ValueError("The exact sparse physical metric currently requires a product source")
    cores = []
    for site, core in enumerate(source.cores):
        entries = [[] for _ in range(4)]
        for operator, input_, output, matrix in mpo.blocks[site]:
            row, column = matrix.indices()
            location = torch.stack((output[row], input_[column]))
            local = mpo.operators[operator] @ core[0, :, 0, 0]
            for physical in range(4):
                if bool(local[physical] != 0):
                    entries[physical].append((location, matrix.values()*local[physical]))
        local_cores = []
        for parts in entries:
            if parts:
                index = torch.cat([value[0] for value in parts], dim=1)
                values = torch.cat([value[1] for value in parts])
            else:
                index = torch.empty(2, 0, dtype=torch.long, device=core.device)
                values = core.new_zeros(0)
            local_cores.append(torch.sparse_coo_tensor(index, values,
                                                       (mpo.bond_dimensions[site+1], mpo.bond_dimensions[site]),
                                                       device=core.device).coalesce())
        cores.append(local_cores)
    return cores


@torch.no_grad()
def exact_product_image_norm(sparse_cores, maximum_gram_bytes, callback=None):
    gram = sparse_cores[0][0].values().new_ones(1, 1)
    prefix_norms = [gram.new_ones(())]
    for site, core in enumerate(sparse_cores):
        dimension = core[0].shape[0]
        requested = dimension*dimension*gram.element_size()
        if requested > maximum_gram_bytes:
            raise RuntimeError(f"Exact physical metric Gram exceeds its declared cap: {requested} bytes")
        fresh = gram.new_zeros(dimension, dimension)
        for matrix in core:
            if not matrix._nnz(): continue
            temporary = torch.sparse.mm(matrix, gram)
            fresh += torch.sparse.mm(matrix, temporary.T)
        gram = (fresh+fresh.T)/2
        prefix_norms.append(gram.trace().clamp_min(0).sqrt())
        if callback: callback({"phase": "exact_physical_metric", "site": site,
                               "gram_bytes": requested, "prefix_frobenius_norm": float(prefix_norms[-1])})
    if gram.shape != (1, 1) or float(gram[0, 0]) <= 0:
        raise RuntimeError("Hamiltonian image has zero/nonpositive squared norm")
    return gram[0, 0], prefix_norms


@torch.no_grad()
def right_physical_frame(sparse_cores, prefix_norms, null_tolerance, maximum_frame_bytes, callback=None):
    factor = sparse_cores[-1][0].values().new_ones(1, 1)
    result = [None]*len(sparse_cores)
    bound = factor.new_zeros(())
    history = []
    for site in range(len(sparse_cores)-1, -1, -1):
        core = sparse_cores[site]
        requested = core[0].shape[1]*4*factor.shape[1]*factor.element_size()
        if requested > maximum_frame_bytes:
            raise RuntimeError(f"Physical right-frame intermediate exceeds cap: {requested} bytes")
        part = torch.stack([torch.sparse.mm(matrix.T, factor) for matrix in core], dim=1)
        matrix = part.flatten(1)
        if site == 0:
            if matrix.shape[0] != 1: raise RuntimeError("Physical image has an open left boundary")
            result[site] = matrix.reshape(1, 4, 1, factor.shape[1])
            break
        u, singular, vh = torch.linalg.svd(matrix, full_matrices=False)
        retained = singular > singular.max()*null_tolerance
        if not bool(retained.any()): raise RuntimeError("Right physical frame is empty")
        discarded = singular[~retained].norm()
        bound += prefix_norms[site]*discarded
        factor = u[:, retained]*singular[retained]
        result[site] = vh[retained].reshape(int(retained.sum()), 4, 1, part.shape[-1])
        row = {"phase": "right_physical_frame", "site": site, "retained_right_dimension": int(retained.sum()),
               "discarded_local_norm": float(discarded), "prefix_weighted_error_bound": float(prefix_norms[site]*discarded)}
        history.append(row)
        if callback: callback(row)
    return result, bound, history


@torch.no_grad()
def physical_product_response(mpo, source, maximum_bond=64, singular_tolerance=1e-11,
                              frame_null_tolerance=1e-13, maximum_gram_bytes=250_000_000,
                              maximum_frame_bytes=1_500_000_000, callback=None):
    sparse = product_image_cores(mpo, source)
    exact_squared_norm, prefix_norms = exact_product_image_norm(sparse, maximum_gram_bytes, callback)
    frame, right_bound, right_history = right_physical_frame(sparse, prefix_norms, frame_null_tolerance,
                                                             maximum_frame_bytes, callback)
    allowed = quantum_bonds(mpo.sites, source.electrons, source.twice_spin, 1)
    labels, cores, history = [[(0, 0, 0, 0)]], [], []
    carry = frame[0].new_ones(1, 1)
    discarded = carry.new_zeros(())
    endpoint_error = 0.
    for site, original in enumerate(frame):
        raw = torch.einsum("xl,ldar->xdar", carry, original)
        if site+1 == len(frame):
            projected, endpoint_error = _project_endpoint(raw, labels[-1], source.electrons, source.twice_spin)
            cores.append(projected)
            labels.append([(source.electrons, source.twice_spin, 0, source.twice_spin)])
            break
        core, carry, new_labels, row = _isometry(raw, labels[-1], allowed[site+1], maximum_bond, singular_tolerance)
        cores.append(core); labels.append(new_labels)
        discarded += carry.new_tensor(row["local_full_discarded_squared_norm"])
        row.update({"phase": "physical_schmidt_truncation", "site": site})
        history.append(row)
        if callback: callback(row)
    result = SpinChain(cores, labels, source.electrons, source.twice_spin)
    approximate_squared_norm = purified_norm(result.cores)
    cross = mpo.contract(source.cores, bra=result.cores)
    difference = exact_squared_norm+approximate_squared_norm-2*cross
    bound = right_bound+(discarded+endpoint_error).sqrt()
    floating_allowance = 128*torch.finfo(torch.float64).eps*(exact_squared_norm+approximate_squared_norm+2*cross.abs())
    relative_error = ((difference.clamp_min(0)+floating_allowance)/exact_squared_norm).sqrt()
    diagnostics = {"exact_full_image_squared_norm": float(exact_squared_norm),
                   "approximate_image_squared_norm": float(approximate_squared_norm), "exact_approximate_image_overlap": float(cross),
                   "full_image_difference_squared_norm": float(difference),
                   "full_image_relative_error_upper_estimate": float(relative_error),
                   "right_frame_triangle_error_bound": float(right_bound),
                   "sequential_projection_error_bound": float(bound), "floating_squared_norm_allowance": float(floating_allowance),
                   "rigorous_interval_arithmetic_bound": False,
                   "physical_compression_norm_measured_without_global_vector": True,
                   "maximum_bond": max(value.shape[-1] for value in result.cores),
                   "capacity_limited": any(row["capacity_limited"] for row in history),
                   "right_frame_history": right_history, "truncation_history": history,
                   "full_stationarity_of_a_correlated_candidate_certified": False}
    return result, diagnostics
