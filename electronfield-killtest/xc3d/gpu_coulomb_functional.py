"""Coulomb-complete density moments with a kinetic completion.

Our research hypothesis is E_xc[D] = F_M(q[D])-Ts_M(q[D])-E_H[D], where
F_M constrains spatial Coulomb-probe moments, and Ts_M is the matching
noninteracting constrained kinetic energy. The total energy retains Tr(T D).
For a frozen frame, v_xc = (-lambda+u_s).Q-J[D]. This is a small-basis
research implementation; no novelty claim is made for constrained search,
Kohn-Sham duality or subspace linear algebra.
"""
from .runtime import configure, torch_setup
ROOT = configure(gpu_only_compatible=True)

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import torch
from .gpu_chemistry_transfer import cuda_operators
from .gpu_manybody_ops import molecular_words, dense_operator, fixed_sector
from .gpu_moment_functional import one_body_words
from .gpu_spin_chain import SpinNumberState, amplitudes
from .gpu_density_completion import complete_density

KCAL = 627.509474


@torch.no_grad()
def charge_probes(case, basis, orbital_map):
    import cupy as cp
    from pyscf import gto
    from gpu4pyscf.df.j_engine_3c2e import contract_int3c2e_auxvec
    mol = gto.M(atom=case["atom"], unit="Angstrom", basis=basis,
                charge=case["charge"], spin=case["spin"], verbose=0)
    nucmol = gto.mole.fakemol_for_charges(mol.atom_coords())
    def forbidden(*args, **kwargs):
        raise RuntimeError("CPU molecular integrals are forbidden")
    previous = gto.Mole.intor, gto.Mole.intor_symmetric, gto.Mole.intor_asymmetric
    gto.Mole.intor = gto.Mole.intor_symmetric = gto.Mole.intor_asymmetric = forbidden
    probes = []
    try:
        for atom in range(mol.natm):
            unit_charge = torch.zeros(mol.natm, device=orbital_map.device)
            unit_charge[atom] = 1.
            value = contract_int3c2e_auxvec(mol, nucmol, cp.from_dlpack(unit_charge))
            if not isinstance(value, cp.ndarray):
                raise RuntimeError("Nuclear-probe integral was not evaluated on CUDA")
            operator = torch.from_dlpack(value)
            probes.append(orbital_map.T @ operator @ orbital_map)
    finally:
        gto.Mole.intor, gto.Mole.intor_symmetric, gto.Mole.intor_asymmetric = previous
    probes = torch.stack(probes)
    identity = torch.eye(len(orbital_map), device=orbital_map.device)
    traceless = probes-probes.diagonal(dim1=-2, dim2=-1).sum(-1)[:, None, None]*identity/len(identity)
    vectors, singular, _ = torch.linalg.svd(traceless.flatten(1), full_matrices=False)
    selected = singular > singular.max()*1e-10
    transform = vectors[:, selected].T/singular[selected, None]
    moments = torch.einsum("ab,bij->aij", transform, traceless)
    return moments, probes, transform


@torch.no_grad()
def kinetic_dual(kinetic, moments, target, pairs, initial=None, maximum=80):
    """Noninteracting Legendre lower bound; numerical work stays on CUDA."""
    potential = torch.zeros_like(target) if initial is None else initial.clone()
    for iteration in range(maximum):
        eigenvalues, orbitals = torch.linalg.eigh(kinetic+torch.einsum("a,aij->ij", potential, moments))
        rotated = torch.einsum("ia,mij,jb->mab", orbitals, moments, orbitals)
        density = 2*rotated[:, :pairs, :pairs].diagonal(dim1=-2, dim2=-1).sum(-1)
        residual = density-target
        dual = 2*eigenvalues[:pairs].sum()-potential @ target
        if float(residual.norm()) < 1e-9:
            break
        gap = (eigenvalues[pairs:][None]-eigenvalues[:pairs, None]).clamp_min(1e-9)
        transition = rotated[:, :pairs, pairs:].flatten(1)
        response = -4*(transition/gap.flatten()) @ transition.T
        update = -torch.linalg.solve(response-torch.eye(len(target), device=target.device)*1e-9, residual)
        update *= torch.clamp(.75/update.norm().clamp_min(1e-30), max=1.)
        accepted = False
        for scale in (1., .5, .25, .125, .0625, .03125):
            trial = potential+scale*update
            trial_values = torch.linalg.eigvalsh(kinetic+torch.einsum("a,aij->ij", trial, moments))
            trial_dual = 2*trial_values[:pairs].sum()-trial @ target
            if float(trial_dual) >= float(dual)-1e-12:
                potential = trial
                accepted = True
                break
        if not accepted:
            break
    # Report a potential, lower bound and residual at the same final point.
    eigenvalues, orbitals = torch.linalg.eigh(kinetic+torch.einsum("a,aij->ij", potential, moments))
    determinant = 2*orbitals[:, :pairs] @ orbitals[:, :pairs].T
    density = torch.einsum("ij,aij->a", determinant, moments)
    return {"functional_lower_bound": 2*eigenvalues[:pairs].sum()-potential @ target,
            "potential": potential, "residual": float((density-target).norm()),
            "determinant": determinant, "iterations": iteration+1}


@torch.no_grad()
def prepare(case, basis, device):
    destination = ROOT / "data/coulomb_moment_functional" / f"{case['id']}.pt"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return torch.load(destination, map_location=device, weights_only=True)
    data = cuda_operators(case, basis)
    moments, probes, transform = charge_probes(case, basis, data["orbital_map"])
    external = data["h"]-data["kinetic"]
    identity = torch.eye(len(external), device=device)
    constant = external.trace()/len(external)
    coefficients = torch.einsum("aij,ij->a", moments, external)
    reconstructed = torch.einsum("a,aij->ij", coefficients, moments)+constant*identity
    discrepancy = float((external-reconstructed).abs().max())
    probe_discrepancy = float((external+torch.einsum("a,aij->ij", data["charges"].to(external.dtype), probes)).abs().max())
    if discrepancy > 1e-8 or probe_discrepancy > 1e-8:
        raise RuntimeError("Coulomb moment frame does not span the actual external potential")
    sites, pairs = len(external), case["electrons"]//2
    bits, configurations = fixed_sector(sites, pairs, pairs, device)
    words, coefficient = molecular_words(data["kinetic"], data["eri"])
    universal = dense_operator(words, coefficient, bits, sites)
    qw, qc = one_body_words(moments)
    q_matrices = torch.stack([dense_operator(qw, row, bits, sites) for row in qc])
    initial = SpinNumberState(sites, case["electrons"], channels=2, device=device)
    seed = amplitudes(initial.cores(reference=True), configurations)
    determinant = identity.new_zeros(sites, sites)
    determinant[:pairs, :pairs] = 2*torch.eye(pairs, device=device)
    target = torch.einsum("ij,aij->a", determinant, moments)
    data.update(moments=moments, raw_probes=probes, moment_transform=transform,
                external_coefficients=coefficients, external_constant=constant,
                external_span_error_hartree=discrepancy, charge_probe_error_hartree=probe_discrepancy,
                bits=bits, configurations=configurations, universal=universal,
                q_matrices=q_matrices, input_determinant=determinant, input_target=target, seed=seed)
    torch.save({key: value.cpu() if isinstance(value, torch.Tensor) else value for key, value in data.items()}, destination)
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=Path("research/coulomb_functional_protocol_20261116.json"))
    parser.add_argument("--output", type=Path, default=Path("research/coulomb_functional_water_gpu_20261116.json"))
    args = parser.parse_args()
    device = torch_setup()
    started = time.perf_counter()
    protocol = json.loads(args.protocol.read_text())
    case = protocol["case"]
    data = prepare(case, protocol["basis"], device)
    target = data["input_target"]
    kinetic = kinetic_dual(data["kinetic"], data["moments"], target, case["electrons"]//2)
    interacting = complete_density(data["universal"], data["q_matrices"], target, data["seed"],
                                    maximum=protocol["maximum_subspace_dimension"],
                                    tolerance=protocol["stationarity_tolerance"])
    determinant = data["input_determinant"]
    kinetic_input = (determinant*data["kinetic"]).sum()
    hartree = .5*torch.einsum("ij,ijkl,kl->", determinant, data["eri"], determinant)
    xc = interacting["energy"]-kinetic["functional_lower_bound"]-hartree
    total = kinetic_input+hartree+xc+data["external_coefficients"] @ target
    total = total+data["external_constant"]*case["electrons"]+data["nuclear"]
    # New exact physical label is generated after the model has stopped.
    physical = data["universal"]+torch.einsum("a,aij->ij", data["external_coefficients"], data["q_matrices"])
    physical = physical+torch.eye(len(physical), device=device)*(data["external_constant"]*case["electrons"]+data["nuclear"])
    reference = torch.linalg.eigvalsh(physical)[0]
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "device": device,
              "hypothesis": "Coulomb-complete constrained correlation plus kinetic completion outside the moment frame",
              "protocol": protocol, "scope": "One new closed-shell water geometry, full stated minimal basis; not broad/CBS validation",
              "input_reference_status": data["reference_status"], "moment_count": len(target),
              "external_span_error_hartree": data["external_span_error_hartree"],
              "charge_probe_error_hartree": data["charge_probe_error_hartree"],
              "interacting_status": interacting["status"], "density_moment_residual": interacting["density_residual"],
              "state_stationarity_residual": interacting["residual"], "kinetic_density_residual": kinetic["residual"],
              "kinetic_iterations": kinetic["iterations"], "xc_energy_hartree": float(xc),
              "model_total_hartree": float(total), "reference_total_hartree": float(reference),
              "total_error_kcal": float((total-reference)*KCAL),
              "kinetic_completion_hartree": float(kinetic_input-kinetic["functional_lower_bound"]),
              "moment_xc_potential_coefficients": (-interacting["potential"]+kinetic["potential"]).tolist(),
              "full_xc_potential": "(-lambda+u_s).Q-J[D], frozen-frame derivative; stationarity required",
              "history": interacting["history"], "reference_energy_fitted": False,
              "seconds": time.perf_counter()-started, "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
              "limitations": ["One-shot input density; model-driven self-consistency remains unimplemented",
                              "Potential derivative requires interacting and noninteracting convergence",
                              "Frame/basis derivatives are required for nuclear forces",
                              "Ensemble noninteracting representability can require fractional occupations",
                              "Dense finite-space action; no compression/speed claim",
                              "Global finite subspace does not guarantee fragment separability"]}
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    torch.save({"interacting_state": interacting["state"].cpu(), "interacting_potential": interacting["potential"].cpu(),
                "kinetic_potential": kinetic["potential"].cpu(), "report": report}, args.output.with_suffix(".pt"))
    print(json.dumps({key: value for key, value in report.items() if key not in ("history", "protocol")}), flush=True)


if __name__ == "__main__":
    main()
