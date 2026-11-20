"""Compositional spin-singlet coherent blocks and their physical projection.

Each block uses its own orthonormal spatial subspace and fixed even N,S=0.
Tensor products preserve independent-fragment rank without a fixed global
determinant budget. Density couplings are the exact projection of the physical
Hamiltonian into these neutral singlet sectors. Charge/spin-transfer sectors
are deliberately absent and must be assessed, not silently approximated away.

Accuracy status: neutral-singlet-only blocks fail the close H4 case by about
210 kcal/mol. The projector agrees with an independent physical action. This
restricted hypothesis is closed; adding optimizer effort is not a remedy.
"""
import torch
from .gpu_coherent_orbitals import CofactorKernel, determinant


@torch.no_grad()
def complete_pair_orbitals(sites, device):
    identity = torch.eye(sites, device=device)
    vectors = [identity[:, index] for index in range(sites)]
    vectors += [(identity[:, first]+identity[:, second])/(2**.5)
                for first in range(sites) for second in range(first+1, sites)]
    return torch.stack(vectors)[:, :, None]


@torch.no_grad()
def orthonormal_block(h, eri, orbitals):
    kernel = CofactorKernel(h, eri, orbitals.shape[-1])
    matrix, metric = kernel.matrices(orbitals)
    lower = torch.linalg.cholesky(metric)
    transform = torch.linalg.solve_triangular(lower.T, torch.eye(len(metric), device=h.device), upper=True)
    effective = transform.T @ matrix @ transform
    overlap = torch.einsum("ria,sib->rsab", orbitals, orbitals)
    first = kernel.cofactor(overlap, 1)
    raw_density = torch.einsum("rs,rsab,rpa,sqb->pqrs", 2*determinant(overlap), first, orbitals, orbitals)
    density = torch.einsum("ra,pqrs,sb->pqab", transform, raw_density, transform)
    number = torch.einsum("ppab->ab", density)
    error = float((number-2*orbitals.shape[-1]*torch.eye(len(metric), device=h.device)).abs().max())
    if error > 1e-9:
        raise RuntimeError(f"Coherent block violates its number identity: {error}")
    return {"hamiltonian": (effective+effective.T)/2, "density": density,
            "transform": transform, "orbitals": orbitals,
            "metric_error": float((transform.T @ metric @ transform-torch.eye(len(metric), device=h.device)).abs().max()),
            "number_error": error, "dimension": len(metric), "electron_count": 2*orbitals.shape[-1]}


@torch.no_grad()
def project_two_blocks(h, eri, split, nuclear):
    first_h, second_h = h[:split, :split], h[split:, split:]
    first_g, second_g = eri[:split, :split, :split, :split], eri[split:, split:, split:, split:]
    first = orthonormal_block(first_h, first_g, complete_pair_orbitals(split, h.device))
    second = orthonormal_block(second_h, second_g, complete_pair_orbitals(len(h)-split, h.device))
    a, b = first["dimension"], second["dimension"]
    # Exact spin-singlet neutral-sector direct-minus-exchange projection.
    interaction = eri[:split, :split, split:, split:]
    exchange = eri[:split, split:, split:, :split].permute(0, 3, 2, 1)
    coupling = (interaction-.5*exchange).reshape(split*split, (len(h)-split)**2)
    left = first["density"].reshape(split*split, a*a)
    right = second["density"].reshape((len(h)-split)**2, b*b)
    linked = (left.T @ coupling @ right).reshape(a, a, b, b).permute(0, 2, 1, 3).reshape(a*b, a*b)
    matrix = torch.kron(first["hamiltonian"].contiguous(), torch.eye(b, device=h.device))
    matrix += torch.kron(torch.eye(a, device=h.device), second["hamiltonian"].contiguous())
    matrix += linked+nuclear*torch.eye(a*b, device=h.device)
    return {"matrix": (matrix+matrix.T)/2, "first": first, "second": second}


@torch.no_grad()
def neutral_two_block(h, eri, split, nuclear):
    projection = project_two_blocks(h, eri, split, nuclear)
    matrix, first, second = projection["matrix"], projection["first"], projection["second"]
    a, b = first["dimension"], second["dimension"]
    values, vectors = torch.linalg.eigh((matrix+matrix.T)/2)
    local_a, states_a = torch.linalg.eigh(first["hamiltonian"])
    local_b, states_b = torch.linalg.eigh(second["hamiltonian"])
    product = torch.kron(states_a[:, 0].contiguous(), states_b[:, 0].contiguous())
    product_energy = product @ matrix @ product
    coefficient = vectors[:, 0].reshape(a, b)
    schmidt = torch.linalg.svdvals(coefficient)
    return {"energy": values[0], "product_energy": product_energy,
            "dimension": a*b, "block_dimensions": [a, b],
            "block_number_errors": [first["number_error"], second["number_error"]],
            "block_metric_errors": [first["metric_error"], second["metric_error"]],
            "block_entanglement_singular_values": schmidt,
            "local_eigenvalue_sum": local_a[0]+local_b[0]+nuclear}
