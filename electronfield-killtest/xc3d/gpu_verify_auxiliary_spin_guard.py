"""Inject an independent triplet to test the state-domain and energy guards."""
from .runtime import configure, torch_setup
ROOT = configure(gpu_only_compatible=True)

from datetime import datetime, timezone
import json
from pathlib import Path
import torch
from .auxiliary_density_model import AuxiliaryDensityModel
from .gpu_spin_chain import SpinNumberState, amplitudes
from .gpu_manybody_ops import fixed_sector, molecular_words, dense_operator
from .gpu_fock_blocks import word_connections


@torch.no_grad()
def main():
    device = torch_setup()
    protocol = json.loads(Path("research/auxiliary_forward_protocol_20261117.json").read_text())
    sites, number = 4, 4
    bits, _ = fixed_sector(sites, 2, 2, device)
    identity = torch.eye(sites, device=device)
    symmetric = []
    for first in range(sites):
        for second in range(first, sites):
            matrix = identity.new_zeros(sites, sites); matrix[first, second] = matrix[second, first] = 1
            matrix -= matrix.trace()*identity/sites
            symmetric.append(matrix)
    _, singular, vh = torch.linalg.svd(torch.stack(symmetric).flatten(1), full_matrices=False)
    operators = torch.cat((identity[None]/2, vh[singular > 1e-10].reshape(-1, sites, sites)))
    kinetic = torch.diag(torch.tensor([.5, 1., 1.5, 2.], device=device))
    eri = torch.zeros((sites,)*4, device=device)
    for site in range(sites): eri[site, site, site, site] = .2
    density = torch.diag(torch.tensor([2., 2., 0., 0.], device=device))
    model = AuxiliaryDensityModel({"kinetic": kinetic, "eri": eri, "operators": operators, "bits": bits, "electrons": number}, density, protocol)
    high_bits, configurations = fixed_sector(sites, 3, 1, device)
    state = SpinNumberState(sites, number, twice_spin=2, channels=1, device=device)
    highest = amplitudes(state.cores(), configurations); highest /= highest.norm()
    words = torch.tensor([((i, 1, 1), (i, 0, 0)) for i in range(sites)], device=device)
    row, column, factor = word_connections(words, torch.ones(sites, device=device), bits, high_bits)
    triplet = highest.new_zeros(len(bits)); triplet.scatter_add_(0, row, factor*highest[column]); triplet /= triplet.norm()
    triplet_casimir_error = float((model.singlet.spin_squared(triplet)-2*triplet).norm())
    candidate = torch.cat((model.basis, triplet[:, None]), 1)
    purified = model.purify_extension(model.basis, candidate)
    rejected_by_projection = purified.shape[1] == model.basis.shape[1]
    hwords, coefficients = molecular_words(kinetic, eri)
    dense = dense_operator(hwords, coefficients, bits, sites)
    raw_energy = triplet @ model.action(triplet)
    independent_error = float((raw_energy-triplet @ dense @ triplet).abs())
    model.basis, model.projected_dimension = candidate, 0
    rejected_by_guard = False
    try:
        model.refresh()
    except RuntimeError as exc:
        rejected_by_guard = "singlet domain" in str(exc)
    valid = (triplet_casimir_error < 1e-10 and rejected_by_projection and rejected_by_guard
             and independent_error < 1e-10 and float(raw_energy) >= number*float(torch.linalg.eigvalsh(kinetic)[0])-1e-10)
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "device": device, "passed": valid,
              "independent_triplet_casimir_error": triplet_casimir_error, "triplet_extension_removed": rejected_by_projection,
              "invalid_space_rejected_before_energy": rejected_by_guard, "raw_triplet_energy_hartree": float(raw_energy),
              "independent_full_hamiltonian_error_hartree": independent_error, "new_ground_state_or_reference": False}
    Path("research/auxiliary_spin_guard_checks_gpu_20261117.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report), flush=True)
    if not valid: raise RuntimeError("Auxiliary state spin/energy guards failed")


if __name__ == "__main__": main()
