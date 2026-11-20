"""Frozen-protocol transfer of our physical state to neutral LiH and BeH2.

All one- and two-electron integrals, optimization and new reference work use
CUDA. PySCF supplies basis/molecule metadata only; its CPU integral calls are
blocked. Uses the complete stated minimal basis and no reference-energy loss.
"""
from .runtime import configure, torch_setup
ROOT = configure(gpu_only_compatible=True)

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import torch
from .gpu_four_electron import bounded_reference, transform_eri
from .gpu_spin_chain import SpinNumberState, amplitudes
from .gpu_manybody_ops import molecular_words, spin_words, dense_operator, expectation, fixed_sector
from .gpu_density_completion import complete_molecular

KCAL = 627.509474


@torch.no_grad()
def cuda_operators(case, basis):
    import cupy as cp
    from pyscf import gto
    from gpu4pyscf.scf.hf import get_hcore
    from gpu4pyscf.pbc.gto.int1e import int1e_ovlp, int1e_kin
    from gpu4pyscf.scf.int4c2e import get_int4c2e
    mol = gto.M(atom=case["atom"], unit="Angstrom", basis=basis,
                charge=case["charge"], spin=case["spin"], verbose=0)
    def forbidden(*args, **kwargs):
        raise RuntimeError("CPU molecular integrals are forbidden by the GPU-only instruction")
    previous = gto.Mole.intor, gto.Mole.intor_symmetric, gto.Mole.intor_asymmetric
    gto.Mole.intor = gto.Mole.intor_symmetric = gto.Mole.intor_asymmetric = forbidden
    try:
        arrays = [get_hcore(mol), int1e_ovlp(mol), int1e_kin(mol), get_int4c2e(mol)]
        if not all(isinstance(value, cp.ndarray) for value in arrays):
            raise RuntimeError("An integral engine returned a non-CUDA result")
        h, overlap, kinetic, eri = [torch.from_dlpack(value) for value in arrays]
    finally:
        gto.Mole.intor, gto.Mole.intor_symmetric, gto.Mole.intor_asymmetric = previous
    eigenvalues, vectors = torch.linalg.eigh(overlap)
    if float(eigenvalues.min()) < 1e-9:
        raise RuntimeError("The complete stated basis is numerically singular")
    orth = (vectors*eigenvalues.rsqrt()) @ vectors.T
    h_orth, eri_orth = orth.T @ h @ orth, transform_eri(eri, orth)
    if case["spin"]:
        _, orbitals = torch.linalg.eigh(h_orth)
        reference_status = {"converged": None, "source": "Core orbitals, exact high-spin determinant; no open-shell SCF"}
    else:
        orbitals, reference_status = bounded_reference(h_orth, eri_orth, pairs=case["electrons"]//2)
    orbital_map = orth @ orbitals
    h = orbital_map.T @ h @ orbital_map
    kinetic = orbital_map.T @ kinetic @ orbital_map
    eri = transform_eri(eri, orbital_map)
    # Coordinates/charges below are metadata transfers; their arithmetic is CUDA.
    coords = torch.as_tensor(mol.atom_coords().tolist(), device=h.device)
    charges = torch.as_tensor(mol.atom_charges().tolist(), device=h.device, dtype=h.dtype)
    distance = torch.cdist(coords, coords).clamp_min(1e-12)
    nuclear = torch.triu(charges[:, None]*charges[None]/distance, diagonal=1).sum()
    return {"h": h, "kinetic": kinetic, "eri": eri, "nuclear": nuclear,
            "orbital_map": orbital_map, "coords": coords, "charges": charges,
            "local_h": h_orth, "local_eri": eri_orth, "reference_orbitals": orbitals,
            "orthogonalizer": orth, "atom_ao_slices": mol.aoslice_by_atom()[:, 2:4].tolist(),
            "basis": basis, "case": case, "reference_status": reference_status,
            "cpu_integrals_disabled": True}


def molecular_case(case, protocol, device, output):
    started = time.perf_counter()
    destination = ROOT / "data/general_chemistry_transfer" / f"{case['id']}.pt"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        data = torch.load(destination, map_location=device, weights_only=True)
    else:
        print(json.dumps({"id": case["id"], "phase": "new_cuda_integrals"}), flush=True)
        data = cuda_operators(case, protocol["basis"])
        torch.save({k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in data.items()}, destination)
    sites, electrons = len(data["h"]), case["electrons"]
    up, down = (electrons+case["spin"])//2, (electrons-case["spin"])//2
    bits, configurations = fixed_sector(sites, up, down, device)
    words, coefficient = molecular_words(data["h"], data["eri"])
    matrix = dense_operator(words, coefficient, bits, sites)
    state = SpinNumberState(sites, electrons, twice_spin=case["spin"], channels=2, device=device)
    reference_vector = amplitudes(state.cores(reference=True), configurations)
    reference_energy = reference_vector @ matrix @ reference_vector
    # Exact reference labels have not been computed or loaded at this point.
    config = protocol["optimizer"]
    optimizer = torch.optim.Adam(state.parameters(), lr=config["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config["steps"], eta_min=config["cosine_final_lr"])
    best_energy, best = float(reference_energy), None
    history = []
    for step in range(config["steps"]):
        optimizer.zero_grad(set_to_none=True)
        vector = amplitudes(state.cores(), configurations)
        vector = vector/vector.norm()
        energy = vector @ matrix @ vector
        energy.backward()
        if not bool(torch.isfinite(energy)) or not all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in state.parameters()):
            raise RuntimeError("Nonfinite state in frozen transfer experiment")
        if float(energy) < best_energy:
            best_energy = float(energy.detach())
            best = [p.detach().clone() for p in state.reduced]
        torch.nn.utils.clip_grad_norm_(state.parameters(), 10.)
        optimizer.step(); scheduler.step()
        if step == 0 or (step+1) % 100 == 0:
            row = {"id": case["id"], "step": step+1, "lowering_from_input_kcal": float((energy.detach()-reference_energy)*KCAL),
                   "seconds": time.perf_counter()-started}
            history.append(row); print(json.dumps(row), flush=True)
    if best is not None:
        for parameter, value in zip(state.reduced, best):
            parameter.data.copy_(value)
        cores = state.cores()
    else:
        cores = state.cores(reference=True)
    with torch.no_grad():
        vector = amplitudes(cores, configurations)
        vector = vector/vector.norm()
        energy = vector @ matrix @ vector
        seed_energy = energy
        independent_energy = expectation(cores, words, coefficient)
        completion = None
        if "completion" in protocol:
            completion = complete_molecular(matrix, vector, maximum=protocol["completion"]["maximum_dimension"],
                                             tolerance=protocol["completion"]["residual_tolerance"])
            vector, energy = completion["state"], completion["energy"]
        # New reference generation only after the frozen model solve is done.
        eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
        exact = eigenvalues[0]
        sw, sc = spin_words(sites, device)
        spin_matrix = dense_operator(sw, sc, bits, sites)
        exact_spin = eigenvectors[:, 0] @ spin_matrix @ eigenvectors[:, 0]+(case["spin"]/2)**2
        model_spin = vector @ spin_matrix @ vector+(case["spin"]/2)**2
        number = (vector.square()*configurations.ne(0).sum(-1)
                  +vector.square()*configurations.eq(3).sum(-1)).sum()
        error = float((energy-exact)*KCAL)
    row = {"id": case["id"], "electrons": electrons, "spatial_orbitals": sites,
           "complete_stated_basis": True, "basis": protocol["basis"], "sector_dimension": len(bits),
           "parameters": sum(p.numel() for p in state.parameters()), "bond_dimensions": state.bond_dimensions,
           "input_reference_status": data["reference_status"], "energy_error_kcal": error,
           "model_energy_hartree": float(energy+data["nuclear"]), "exact_energy_hartree": float(exact+data["nuclear"]),
           "particle_number": float(number), "model_spin_squared": float(model_spin), "reference_spin_squared": float(exact_spin),
           "reference_not_degraded": float(energy) <= float(reference_energy)+1e-8,
           "independent_seed_contraction_error_hartree": float((independent_energy-seed_energy).abs()),
           "variational_sign_valid": error >= -1e-6, "history": history,
           "seconds": time.perf_counter()-started}
    if completion is not None:
        row["completion"] = {key: value for key, value in completion.items() if key not in ("state", "energy")}
        row["seed_energy_error_kcal"] = float((seed_energy-exact)*KCAL)
    torch.save({"model": state.state_dict(), "case": case, "basis": protocol["basis"],
                "sites": sites, "electrons": electrons, "completed_state": vector.cpu(), "report": row}, output.parent / f"{case['id']}.pt")
    print(json.dumps({key: value for key, value in row.items() if key != "history"}), flush=True)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=Path("research/general_state_transfer_protocol_20261116.json"))
    parser.add_argument("--output", type=Path, default=Path("research/general_chemistry_transfer_gpu_20261116.json"))
    args = parser.parse_args()
    device = torch_setup()
    protocol = json.loads(args.protocol.read_text())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "device": device,
              "protocol_sha256": hashlib.sha256(args.protocol.read_bytes()).hexdigest(), "protocol": protocol,
              "results": [], "failures": [], "reference_energy_fitted": False}
    for case in protocol["cases"]:
        try:
            report["results"].append(molecular_case(case, protocol, device, args.output))
        except Exception as exc:
            report["failures"].append({"id": case["id"], "error": f"{type(exc).__name__}: {exc}"})
            print(json.dumps(report["failures"][-1]), flush=True)
        args.output.write_text(json.dumps(report, indent=2)+"\n")
    if report["results"]:
        errors = torch.tensor([row["energy_error_kcal"] for row in report["results"]], device=device)
        report["mae_kcal"] = float(errors.abs().mean())
        report["max_absolute_error_kcal"] = float(errors.abs().max())
        report["all_cases_below_one"] = not report["failures"] and bool((errors.abs() < 1).all())
        report["balanced_stretch_errors_kcal"] = {}
        for prefix in sorted({case["id"].split("_")[0] for case in protocol["cases"]}):
            pair = [row for row in report["results"] if row["id"].startswith(prefix)]
            if len(pair) == 2:
                report["balanced_stretch_errors_kcal"][prefix] = float(pair[1]["energy_error_kcal"]-pair[0]["energy_error_kcal"])
    report["peak_cuda_bytes"] = torch.cuda.max_memory_allocated()
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({key: value for key, value in report.items() if key not in ("protocol", "results")}), flush=True)


if __name__ == "__main__":
    main()
