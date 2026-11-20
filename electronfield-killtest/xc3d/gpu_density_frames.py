"""Scale-covariant density-region operators for our constrained state model.

Geometry enters only through distances/ratios and the quadrature basis. The
radial length is measured from the supplied electronic density, not a fixed
Bohr radius or an element embedding. All numerical operations run on CUDA.
"""
import torch


def becke_partition(coords, landmarks):
    radius = torch.cdist(coords, landmarks)
    if len(landmarks) == 1:
        return torch.ones_like(radius), radius
    separation = torch.cdist(landmarks, landmarks)
    eye = torch.eye(len(landmarks), device=coords.device, dtype=torch.bool)
    mu = (radius[:, :, None]-radius[:, None, :])/separation.masked_fill(eye, 1.)
    mu = mu.clamp(-1, 1)
    for _ in range(3):
        mu = 1.5*mu-.5*mu.pow(3)
    factors = ((1-mu)/2).masked_fill(eye[None], 1.)
    raw = factors.prod(-1)
    return raw/raw.sum(-1, keepdim=True).clamp_min(1e-300), radius


def moment_operators(coords, weights, landmarks, ao, orbital_map, input_density):
    partition, radius = becke_partition(coords, landmarks)
    charge = (weights[:, None]*input_density[:, None]*partition).sum(0)
    length = (weights[:, None]*input_density[:, None]*partition*radius).sum(0)/charge.clamp_min(1e-14)
    radial = radius/(radius+length[None].clamp_min(1e-12))
    fields = torch.stack((partition*(1-radial), partition*radial), -1).flatten(1)
    local = ao.T[None] @ ((weights[:, None]*fields).T[:, :, None]*ao[None])
    matrices = orbital_map.T @ local @ orbital_map
    matrices = (matrices+matrices.transpose(-1, -2))/2
    # Remove the fixed-N gauge and linear dependence, retaining the same span.
    identity = torch.eye(matrices.shape[-1], device=matrices.device, dtype=matrices.dtype)
    traceless = matrices-matrices.diagonal(dim1=-2, dim2=-1).sum(-1)[:, None, None]*identity/len(identity)
    vectors, singular, _ = torch.linalg.svd(traceless.flatten(1), full_matrices=False)
    selected = singular > singular.max()*1e-8
    transform = vectors[:, selected].T/singular[selected, None]
    independent = torch.einsum("ab,bij->aij", transform, traceless)
    return independent, {"local_electronic_lengths": length, "local_input_charges": charge,
                         "singular_values": singular, "moment_transform": transform,
                         "raw_moment_count": len(matrices), "independent_moment_count": len(independent)}
