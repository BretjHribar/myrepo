"""First N=4 experiment for the general number/spin tensor representation.

Reuses a cached H4 nuclear Hamiltonian, changes the filling to four electrons,
and freezes six active spatial orbitals before optimization. Generates the
new active-space reference on CUDA; no old FCI label is rerun. This is a
representation experiment, not a full-basis chemical accuracy claim.
"""
from .runtime import configure, torch_setup
ROOT = configure(gpu_only_compatible=True)

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import numpy as np
import torch
from .gpu_geminal import electronic_hamiltonian
from .gpu_spin_chain import SpinNumberState, norm, amplitudes
from .gpu_manybody_ops import molecular_words, spin_words, expectation, fixed_sector, dense_operator

KCAL = 627.509474


def transform_eri(eri, orbitals):
    value = torch.einsum("ia,ijkl->ajkl", orbitals, eri)
    value = torch.einsum("jb,ajkl->abkl", orbitals, value)
    value = torch.einsum("kc,abkl->abcl", orbitals, value)
    return torch.einsum("ld,abcl->abcd", orbitals, value)


@torch.no_grad()
def bounded_reference(h, eri, pairs=2):
    _, orbitals = torch.linalg.eigh(h)
    density = 2*orbitals[:, :pairs] @ orbitals[:, :pairs].T
    history_f, history_e = [], []
    previous = None
    for cycle in range(80):
        coulomb = torch.einsum("ijkl,kl->ij", eri, density)
        exchange = torch.einsum("ikjl,kl->ij", eri, density)
        fock = h+coulomb-.5*exchange
        residual = fock @ density-density @ fock
        energy = (density*(h+.5*(coulomb-.5*exchange))).sum()
        if previous is not None and float((energy-previous).abs()) < 1e-10 and float(residual.norm()) < 1e-7:
            _, orbitals = torch.linalg.eigh(fock)
            return orbitals, {"converged": True, "cycles": cycle+1, "residual": float(residual.norm())}
        previous = energy
        history_f.append(fock); history_e.append(residual)
        history_f, history_e = history_f[-7:], history_e[-7:]
        if len(history_f) > 2:
            errors = torch.stack(history_e).flatten(1)
            count = len(history_f)
            matrix = h.new_zeros(count+1, count+1)
            matrix[:count, :count] = errors @ errors.T+torch.eye(count, device=h.device)*1e-14
            matrix[count, :count] = matrix[:count, count] = -1.
            rhs = h.new_zeros(count+1); rhs[-1] = -1.
            coefficient = torch.linalg.solve(matrix, rhs)[:count]
            fock = torch.einsum("a,aij->ij", coefficient, torch.stack(history_f))
        _, orbitals = torch.linalg.eigh((fock+fock.T)/2)
        updated = 2*orbitals[:, :pairs] @ orbitals[:, :pairs].T
        density = .7*density+.3*updated if cycle < 5 else updated
    return orbitals, {"converged": False, "cycles": 80, "residual": float(residual.norm()),
                      "note": "Still a valid input determinant; no repeated SCF campaign"}


@torch.no_grad()
def prepare(device, active):
    destination = ROOT / f"data/general_state_h4_4e_active{active}.pt"
    if destination.exists():
        return torch.load(destination, map_location=device, weights_only=True)
    name = "H4_square_2plus_1.50"
    cached = torch.load(ROOT / "data/gpu_pair_state" / f"{name}.pt", map_location=device, weights_only=True)
    with np.load(ROOT / "data/pilot_3d_ccpvdz" / f"{name}.npz") as data:
        dm, ao, orth, fock, weights, landmarks = [torch.from_numpy(data[key]).to(device) for key in
                                                ("dm", "ao", "orth", "fock", "weights", "landmarks")]
    h = electronic_hamiltonian(dm[0], ao, cached["operator_eri"], fock[0], weights)
    full_h = orth.T @ h @ orth
    full_eri = transform_eri(cached["operator_eri"], orth)
    orbitals, reference_status = bounded_reference(full_h, full_eri)
    selected = orbitals[:, :active]
    h, eri = selected.T @ full_h @ selected, transform_eri(full_eri, selected)
    words, coefficient = molecular_words(h, eri)
    sw, sc = spin_words(active, device)
    bits, configurations = fixed_sector(active, 2, 2, device)
    matrix = dense_operator(words, coefficient, bits, active)
    spin_matrix = dense_operator(sw, sc, bits, active)
    spin_values, spin_vectors = torch.linalg.eigh(spin_matrix)
    singlets = spin_vectors[:, spin_values.abs() < 1e-8]
    projected = singlets.T @ matrix @ singlets
    exact = torch.linalg.eigvalsh((projected+projected.T)/2)[0]
    distances = torch.cdist(landmarks, landmarks).clamp_min(1e-12)
    nuclear = torch.triu(1/distances, diagonal=1).sum()
    density = h.new_zeros(active, active); density[:2, :2] = 2*torch.eye(2, device=device)
    coulomb = torch.einsum("ijkl,kl->ij", eri, density)
    exchange = torch.einsum("ikjl,kl->ij", eri, density)
    wick = (density*(h+.5*(coulomb-.5*exchange))).sum()
    payload = {"h": h, "eri": eri, "words": words, "coefficient": coefficient,
               "spin_words": sw, "spin_coefficient": sc, "bits": bits, "configurations": configurations,
               "matrix": matrix, "spin_matrix": spin_matrix, "reference_electronic": exact,
               "nuclear": nuclear, "wick_reference": wick, "input_reference_status": reference_status,
               "active_orbitals": active, "original_orbitals": len(full_h), "electrons": 4,
               "fixed_spin_dimension": len(bits), "singlet_dimension": singlets.shape[1],
               "source": name, "reference_method": "New exact active-space singlet diagonalization on CUDA"}
    torch.save({key: value.cpu() if isinstance(value, torch.Tensor) else value for key, value in payload.items()}, destination)
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--active", type=int, default=6)
    parser.add_argument("--channels", type=int, default=2)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--output", type=Path, default=Path("research/general_spin_state_4e_gpu_20261116.json"))
    args = parser.parse_args()
    device = torch_setup()
    started = time.perf_counter()
    data = prepare(device, args.active)
    state = SpinNumberState(args.active, 4, channels=args.channels, device=device)
    reference = state.cores(reference=True)
    reference_energy = expectation(reference, data["words"], data["coefficient"])
    discrepancy = float((reference_energy-data["wick_reference"]).abs())
    if discrepancy > 1e-9:
        raise RuntimeError(f"New fermionic contraction disagrees with independent Wick reference: {discrepancy}")
    optimizer = torch.optim.Adam(state.parameters(), lr=.02)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.steps, eta_min=.0002)
    best_energy, best_parameters = float(reference_energy), None
    history = []
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        cores = state.cores()
        energy = expectation(cores, data["words"], data["coefficient"])
        energy.backward()
        if not bool(torch.isfinite(energy)) or not all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in state.parameters()):
            raise RuntimeError("Nonfinite general particle/spin-state optimization")
        value = float(energy.detach())
        if value < best_energy:
            best_energy = value
            best_parameters = [p.detach().clone() for p in state.reduced]
        torch.nn.utils.clip_grad_norm_(state.parameters(), 10.)
        optimizer.step(); scheduler.step()
        if step == 0 or (step+1) % 50 == 0 or step+1 == args.steps:
            row = {"step": step+1, "energy_error_kcal": float((energy.detach()-data["reference_electronic"])*KCAL),
                   "norm": float(norm(cores).detach()), "seconds": time.perf_counter()-started}
            history.append(row)
            print(json.dumps(row), flush=True)
    if best_parameters is None:
        cores = reference
    else:
        for parameter, value in zip(state.reduced, best_parameters):
            parameter.data.copy_(value)
        cores = state.cores()
    with torch.no_grad():
        vector = amplitudes(cores, data["configurations"])
        vector = vector/vector.norm()
        independent = vector @ data["matrix"] @ vector
        spin_squared = expectation(cores, data["spin_words"], data["spin_coefficient"])
        number_op = torch.eye(4, device=device).repeat(args.active, 1, 1)
        local_number = torch.diag(torch.tensor([0., 1., 1., 2.], device=device))
        number = independent.new_zeros(())
        for site in range(args.active):
            operators = number_op.clone(); operators[site] = local_number
            from .gpu_spin_chain import expect_product
            number += expect_product(cores, operators[None])[0]/norm(cores)
        final_error = float((independent-data["reference_electronic"])*KCAL)
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "device": device,
              "scope": "First four-electron test of general number/total-spin tensor representation; six-orbital projection, not full-basis/CBS chemistry",
              "research_role": "Representation for our density-conditioned correlation model; not a frontier-method reproduction",
              "config": vars(args) | {"output": str(args.output)},
              "reference_method": data["reference_method"], "input_reference_status": data["input_reference_status"],
              "active_orbitals": args.active, "full_orbitals": data["original_orbitals"],
              "fixed_spin_dimension": data["fixed_spin_dimension"], "singlet_dimension": data["singlet_dimension"],
              "bond_dimensions": state.bond_dimensions, "parameters": sum(p.numel() for p in state.parameters()),
              "reference_wick_discrepancy_hartree": discrepancy, "history": history,
              "final_error_kcal": final_error, "particle_number": float(number), "spin_squared": float(spin_squared),
              "independent_energy_discrepancy_hartree": abs(float(independent)-best_energy),
              "variational_bound_satisfied": final_error >= -1e-3,
              "reference_not_degraded": float(independent) <= float(reference_energy)+1e-8,
              "peak_cuda_bytes": torch.cuda.max_memory_allocated(), "seconds": time.perf_counter()-started}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    torch.save({"architecture": "number_spin_tensor", "model": state.state_dict(), "config": report["config"]}, args.output.with_suffix(".pt"))
    print(json.dumps({key: value for key, value in report.items() if key != "history"}), flush=True)


if __name__ == "__main__":
    main()
