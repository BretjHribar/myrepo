"""Coulomb-density coordinates without external-potential probe locations.

The ERI is the Coulomb Gram operator of the represented orbital products.
Its resolved range therefore describes the represented spatial density, not
an atom-labelled feature list. All non-null modes at the declared numerical
threshold are kept. This is finite-basis constrained-search bookkeeping,
not a claim of new Coulomb/SVD mathematics or continuum completeness.
"""
import torch
from .gpu_fock_blocks import word_connections


@torch.no_grad()
def density_frame(eri, relative_null_threshold=1e-13):
    if eri.device.type != "cuda":
        raise RuntimeError("Density-frame numerical work requires CUDA")
    sites = len(eri)
    pairs = torch.triu_indices(sites, sites, device=eri.device)
    symmetric = eri.new_zeros(pairs.shape[1], sites, sites)
    order = torch.arange(len(symmetric), device=eri.device)
    symmetric[order, pairs[0], pairs[1]] = 1
    symmetric[order, pairs[1], pairs[0]] = 1
    symmetric /= symmetric.square().sum((-1, -2)).sqrt()[:, None, None]
    gram = torch.einsum("aij,ijkl,bkl->ab", symmetric, eri, symmetric)
    eigenvalues, directions = torch.linalg.eigh((gram+gram.T)/2)
    threshold = eigenvalues.abs().max()*relative_null_threshold
    if float(eigenvalues.min()) < -float(eigenvalues.abs().max())*1e-10:
        raise RuntimeError("Orbital-product Coulomb Gram matrix is not positive semidefinite")
    selected = eigenvalues > threshold
    modes = torch.einsum("ab,aij->bij", directions[:, selected], symmetric)
    identity = torch.eye(sites, device=eri.device, dtype=eri.dtype)
    modes -= modes.diagonal(dim1=-2, dim2=-1).sum(-1)[:, None, None]*identity/sites
    _, singular, vh = torch.linalg.svd(modes.flatten(1), full_matrices=False)
    traceless = vh[singular > 1e-10].reshape(-1, sites, sites)
    operators = torch.cat((identity[None]/identity.new_tensor(sites).sqrt(), traceless))
    return {"operators": operators, "coulomb_eigenvalues": eigenvalues,
            "retained_coulomb_modes": int(selected.sum()), "null_threshold": float(threshold),
            "symmetric_matrix_dimension": len(symmetric)}


@torch.no_grad()
def project_operator(matrix, frame):
    coefficients = torch.einsum("aij,ij->a", frame["operators"], matrix)
    reconstructed = torch.einsum("a,aij->ij", coefficients, frame["operators"])
    return coefficients, reconstructed, (matrix-reconstructed).norm()


@torch.no_grad()
def one_body_density(vector, bits, sites):
    density = vector.new_zeros(sites, sites)
    for first in range(sites):
        for second in range(first, sites):
            words = torch.tensor([((first, spin, 1), (second, spin, 0)) for spin in range(2)], device=bits.device)
            row, column, factor = word_connections(words, vector.new_ones(2), bits, bits)
            value = (vector[row]*factor*vector[column]).sum()
            density[first, second] = density[second, first] = value
    return density


@torch.no_grad()
def product_value_frame(values, relative_null_threshold=1e-11):
    """Resolve the product span directly, without squaring its condition number.

    Row normalization preserves the span and uses every resolved spatial
    direction, including weak overlap regions. No potential or energy enters.
    """
    norms = values.norm(dim=1)
    selected = norms > 1e-150
    normalized = values[selected]/norms[selected, None]
    products = (normalized[:, :, None]*normalized[:, None, :]).flatten(1)
    _, singular, vh = torch.linalg.svd(products, full_matrices=False)
    modes = vh[singular > singular[0]*relative_null_threshold].reshape(-1, values.shape[1], values.shape[1])
    identity = torch.eye(values.shape[1], device=values.device, dtype=values.dtype)
    modes -= modes.diagonal(dim1=-2, dim2=-1).sum(-1)[:, None, None]*identity/len(identity)
    _, gauge_singular, gauge_vh = torch.linalg.svd(modes.flatten(1), full_matrices=False)
    traceless = gauge_vh[gauge_singular > 1e-10].reshape(-1, len(identity), len(identity))
    operators = torch.cat((identity[None]/identity.new_tensor(len(identity)).sqrt(), traceless))
    return {"operators": operators, "product_singular_values": singular,
            "retained_product_modes": len(modes), "null_threshold": float(singular[0]*relative_null_threshold),
            "symmetric_matrix_dimension": len(identity)*(len(identity)+1)//2}


@torch.no_grad()
def spatial_product_values(case, basis, device, seed, radial_count=24, directions_count=32):
    """CUDA AO values at basis-scaled points; metadata uses the host only.

    Centers/exponents describe the supplied orbital basis. Nuclear charges,
    h, T and reference energies do not choose the sample locations or span.
    """
    import cupy as cp
    from pyscf import gto
    from gpu4pyscf.dft.numint import eval_ao
    from gpu4pyscf.pbc.gto.int1e import int1e_ovlp
    mol = gto.M(atom=case["atom"], basis=basis, unit="Angstrom",
                charge=case["charge"], spin=case["spin"], verbose=0)
    centers = torch.tensor(mol.atom_coords().tolist(), device=device)
    exponents = torch.tensor([float(x) for shell in range(mol.nbas) for x in mol.bas_exp(shell)], device=device)
    smallest = exponents.max().rsqrt()*.1
    largest = exponents.min().rsqrt()*6
    radii = torch.exp(torch.linspace(smallest.log(), largest.log(), radial_count, device=device))
    generator = torch.Generator(device=device).manual_seed(seed)
    directions = torch.randn(directions_count, 3, device=device, generator=generator)
    directions /= directions.norm(dim=1, keepdim=True)
    points = (centers[:, None, None, :]+radii[None, :, None, None]*directions[None, None]).reshape(-1, 3)
    def forbidden(*args, **kwargs):
        raise RuntimeError("CPU AO/integral evaluation is forbidden")
    previous = gto.Mole.intor, gto.Mole.intor_symmetric, gto.Mole.intor_asymmetric, gto.Mole.eval_gto
    gto.Mole.intor = gto.Mole.intor_symmetric = gto.Mole.intor_asymmetric = gto.Mole.eval_gto = forbidden
    try:
        overlap = int1e_ovlp(mol)
        values = eval_ao(mol, cp.from_dlpack(points), deriv=0)
        if not isinstance(overlap, cp.ndarray) or not isinstance(values, cp.ndarray):
            raise RuntimeError("A spatial-density engine returned non-CUDA arrays")
        overlap, values = torch.from_dlpack(overlap), torch.from_dlpack(values)
    finally:
        gto.Mole.intor, gto.Mole.intor_symmetric, gto.Mole.intor_asymmetric, gto.Mole.eval_gto = previous
    eigenvalues, eigenvectors = torch.linalg.eigh(overlap)
    orthogonalizer = (eigenvectors*eigenvalues.rsqrt()) @ eigenvectors.T
    return values @ orthogonalizer
