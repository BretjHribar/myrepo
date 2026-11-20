"""Lift a fractional spin-summed 1RDM to fixed-N singlet determinants.

Systematic sampling of the hypersimplex gives at most m weighted paired
determinants with the supplied natural occupations. Every component has the
same exact electron number and total spin; this is not a fluctuating-N gas.
"""
import torch


@torch.no_grad()
def slater_vector(orbitals, occupied, bits):
    sites, pairs = len(orbitals), len(occupied)
    spin_orbitals = orbitals.new_zeros(2*sites, 2*pairs)
    spin_orbitals[::2, :pairs] = orbitals[:, occupied]
    spin_orbitals[1::2, pairs:] = orbitals[:, occupied]
    modes = torch.arange(2*sites, device=bits.device)
    present = ((bits[:, None] >> modes[None]) & 1).bool()
    indices = modes[None].expand(len(bits), -1)[present].reshape(len(bits), 2*pairs)
    vector = torch.linalg.det(spin_orbitals[indices])
    return vector/vector.norm()


@torch.no_grad()
def canonical_lift(orbitals, occupations, electrons, bits):
    if electrons % 2:
        raise ValueError("This lift currently handles spin-unpolarized even-N ensembles")
    fraction = (occupations/2).clamp(0, 1)
    original = fraction.clone()
    pairs = electrons//2
    for _ in range(3):
        difference = pairs-fraction.sum()
        capacity = 1-fraction if float(difference) > 0 else fraction
        fraction = (fraction+difference*capacity/capacity.sum().clamp_min(1e-30)).clamp(0, 1)
    cumulative = fraction.cumsum(0)
    # floor(c+u) changes at u=(-c) mod 1, not at c mod 1.
    breaks = torch.cat((fraction.new_tensor([0., 1.]), (-cumulative[:-1]).remainder(1))).unique(sorted=True)
    weights = breaks[1:]-breaks[:-1]
    middle = (breaks[1:]+breaks[:-1])/2
    selected = weights > 1e-14
    weights, middle = weights[selected], middle[selected]
    weights = weights/weights.sum()
    before = torch.cat((fraction.new_zeros(1), cumulative[:-1]))
    masks = ((cumulative[None]+middle[:, None]).floor()-(before[None]+middle[:, None]).floor()).bool()
    if not bool((masks.sum(-1) == pairs).all()):
        raise RuntimeError("Canonical lift failed exact particle-number allocation")
    vectors = torch.stack([slater_vector(orbitals, torch.nonzero(mask).flatten(), bits) for mask in masks])
    reproduced = (weights[:, None]*masks).sum(0)
    return {"vectors": vectors, "weights": weights,
            "occupation_residual": float((reproduced-original).norm()),
            "number_rounding_allowance": float((fraction-original).norm())}
