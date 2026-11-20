"""Fixed generic physical Krylov start; no seed search or reference selection.

A retained eigenstate can have zero residual in an excited invariant
sector. One prespecified generic CUDA component exposes other sectors.
It is a numerical safeguard, not a proof of the global minimum or novelty.
"""
import torch


@torch.no_grad()
def mixed_physical_start(source, random_seed):
    if source.device.type != "cuda" or source.ndim != 1:
        raise RuntimeError("The physical Krylov start requires one CUDA coefficient vector")
    normalized = source/source.norm()
    if len(source) == 1:
        return normalized, {"policy": "unique_physical_direction", "generic_component_used": False}
    generator = torch.Generator(device=source.device).manual_seed(random_seed)
    generic = torch.randn(source.shape, device=source.device, dtype=source.dtype, generator=generator)
    generic /= generic.norm()
    # Strictly less than unit weight prevents a zero vector even when the
    # probe happens to oppose the retained state. Do not orthogonalize it:
    # in two dimensions that would leave only two discrete probe choices.
    mixed = normalized+.5*generic
    mixed /= mixed.norm()
    return mixed, {"policy": "one_fixed_generic_physical_component", "generic_component_used": True,
                   "fixed_seed": random_seed, "generic_relative_amplitude": .5,
                   "normalized_start_error": float((mixed.norm()-1.).abs()),
                   "retained_start_overlap": float(normalized @ mixed),
                   "seed_sweep_performed": False, "reference_used": False,
                   "global_lowest_eigenvalue_certified": False}
