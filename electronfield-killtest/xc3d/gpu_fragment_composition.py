"""New N=8 product-composition requirement, without reference regeneration."""
from .runtime import configure, torch_setup
ROOT = configure(gpu_only_compatible=True)

from datetime import datetime, timezone
import json
from pathlib import Path
import time
import torch
from .gpu_spin_chain import SpinNumberState, compose_singlets, norm, expect_product
from .gpu_manybody_ops import molecular_words, expectation, spin_words


def overlap(left, right):
    environment = left[0].new_ones(1, 1)
    for a, b in zip(left, right):
        environment = torch.einsum("ij,idk,jdl->kl", environment, a, b)
    return environment.squeeze()


@torch.no_grad()
def main():
    device = torch_setup()
    started = time.perf_counter()
    name = "LiH_neutral_1.595"
    checkpoint = torch.load(f"research/{name}.pt", map_location=device, weights_only=True)
    data = torch.load(ROOT / "data/general_chemistry_transfer" / f"{name}.pt", map_location=device, weights_only=True)
    fragment = SpinNumberState(checkpoint["sites"], checkpoint["electrons"], channels=2, device=device)
    fragment.load_state_dict(checkpoint["model"])
    combined = compose_singlets([fragment, fragment])
    a, b = fragment.cores(), combined.cores()
    direct_product = a+a
    normalized_overlap = overlap(b, direct_product).square()/(norm(b)*norm(direct_product))
    words, coefficient = molecular_words(data["h"], data["eri"])
    shifted = words.clone()
    shifted[:, :, 0] = torch.where(shifted[:, :, 0] >= 0, shifted[:, :, 0]+fragment.sites, shifted[:, :, 0])
    fragment_energy = expectation(a, words, coefficient)
    combined_energy = expectation(b, torch.cat((words, shifted)), torch.cat((coefficient, coefficient)), batch=16)
    number_operators = torch.eye(4, device=device).repeat(combined.sites, combined.sites, 1, 1)
    for site in range(combined.sites):
        number_operators[site, site] = torch.diag(torch.tensor([0., 1., 1., 2.], device=device))
    number = expect_product(b, number_operators).sum()/norm(b)
    sw, sc = spin_words(combined.sites, device)
    spin = expectation(b, sw, sc, batch=16)
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "device": device,
              "scope": "Two independent ordered singlet fragments, ideal disconnected Hamiltonian; N=8, 12 spatial orbitals",
              "normalized_product_overlap_squared": float(normalized_overlap),
              "energy_additivity_error_hartree": float((combined_energy-2*fragment_energy).abs()),
              "particle_number": float(number), "spin_squared": float(spin),
              "bond_dimensions": combined.bond_dimensions, "reference_regenerated": False,
              "seconds": time.perf_counter()-started, "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
              "limitations": ["Known component partition and orbital order are supplied",
                              "Singlet composition only", "Not an automatic fragmentation algorithm",
                              "Global residual completion still lacks a fragment factorization guarantee"]}
    destination = Path("research/fragment_composition_gpu_20261116.json")
    destination.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
