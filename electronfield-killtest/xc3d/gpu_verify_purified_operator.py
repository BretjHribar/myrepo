"""Independent physical/operator tests for the new compositional scope backend."""
from .runtime import configure, torch_setup
ROOT = configure(gpu_only_compatible=True)

import argparse
from datetime import datetime, timezone
import json
from math import comb
from pathlib import Path
import torch
from .gpu_fermion_mpo import FermionMPO
from .gpu_purified_spin import (PurifiedSpinState, purified_norm, purification_amplitudes,
                                compose_purifications)
from .gpu_spin_chain import SpinNumberState, amplitudes
from .gpu_manybody_ops import molecular_words, dense_operator, spin_words


def random_hamiltonian(sites, device):
    h = torch.randn(sites, sites, device=device); h = (h+h.T)/2
    pair = torch.randn(sites+1, sites, sites, device=device)
    pair = .2*(pair+pair.transpose(-1, -2))/2
    eri = torch.einsum("apq,ars->pqrs", pair, pair)
    return h, eri


def randomize(state):
    with torch.no_grad():
        for parameter in state.parameters(): parameter.normal_()


def diagnostic_matrix(mpo):
    """Tiny diagnostic only; never allowed for the production system size."""
    if mpo.sites > 4: raise ValueError("Dense diagnostic is limited to four spatial sites")
    environment = mpo.operators.new_ones(1, 1, 1)
    for site in range(mpo.sites):
        dimension = environment.shape[-1]
        values, rows = [], []
        for operator, input_, output, matrix in mpo.blocks[site]:
            merged = torch.sparse.mm(matrix, environment[input_].flatten(1)).reshape(len(output), dimension, dimension)
            enlarged = torch.einsum("dc,xab->xdacb", mpo.operators[operator], merged)
            values.append(enlarged.reshape(len(output), dimension*4, dimension*4)); rows.append(output)
        fresh = environment.new_zeros(mpo.bond_dimensions[site+1], dimension*4, dimension*4)
        environment = fresh.index_add(0, torch.cat(rows), torch.cat(values)) if values else fresh
    return environment[0]


def exhaustive_purification(cores):
    sites = len(cores)
    local_ancillas = [core.shape[2] for core in cores]
    physical = torch.arange(4**sites, device=cores[0].device)
    local = (physical[:, None] >> (2*torch.arange(sites, device=cores[0].device))) & 3
    strides, total = [], 1
    for dimension in local_ancillas:
        strides.append(total); total *= dimension
    environment = torch.arange(total, device=cores[0].device)
    labels = (environment[:, None]//torch.tensor(strides, device=physical.device)) % torch.tensor(local_ancillas, device=physical.device)
    configurations = local[:, None].expand(-1, len(environment), -1).reshape(-1, sites)
    ancillas = labels[None].expand(len(physical), -1, -1).reshape(-1, sites)
    wave = purification_amplitudes(cores, configurations, ancillas).reshape(len(physical), len(environment))
    return physical, local, wave


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="research/purified_operator_checks_gpu_20261117.json")
    args = parser.parse_args()
    protocol = json.loads(Path("research/purified_operator_protocol_20261117.json").read_text())
    output = Path(args.output)
    if output.exists(): raise RuntimeError("This protocol has a report; do not repeat its completed checks")
    device = torch_setup(protocol["seed"])
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "device": device, "protocol": protocol,
              "results": [], "failures": [], "status": "running"}
    def record(row):
        report["results"].append(row); print(json.dumps(row), flush=True)
        output.write_text(json.dumps(report, indent=2)+"\n")
    output.write_text(json.dumps(report, indent=2)+"\n")
    try:
        sites = protocol["dense_operator_check_sites"]
        h, eri = random_hamiltonian(sites, device)
        words, coefficients = molecular_words(h, eri, threshold=0.)
        mpo = FermionMPO(words, coefficients, sites)
        bits = torch.arange(4**sites, device=device)
        with torch.no_grad():
            independent = dense_operator(words, coefficients, bits, sites)
            constructed = diagnostic_matrix(mpo)
            difference = float((constructed-independent).abs().max())
            hermiticity = float((constructed-constructed.T).abs().max())
        record({"check": "full_operator", "maximum_matrix_difference_hartree": difference,
                "hermiticity_error_hartree": hermiticity, "passed": max(difference, hermiticity) < protocol["operator_and_energy_tolerance_hartree"],
                **mpo.summary()})
        if not report["results"][-1]["passed"]: raise RuntimeError("Literal fermion operator test failed")
        s_words, s_coefficient = spin_words(sites, device)
        spin_matrix = dense_operator(s_words, s_coefficient, bits, sites)
        for specification in protocol["state_cases"]:
            number, spin = specification["electrons"], specification["twice_spin"]
            state = PurifiedSpinState(sites, number, spin, channels=protocol["channels_per_quantum_sector"],
                                      ancilla=protocol["ancilla_dimension"], device=device)
            randomize(state)
            cores = state.cores()
            physical, local, wave = exhaustive_purification(cores)
            total = wave.square().sum()
            trace_difference = float((total-purified_norm(cores)).abs())
            energy = mpo.expectation(cores)
            direct = (wave*(independent @ wave)).sum()/total
            parameters = list(state.parameters())
            gradient = torch.autograd.grad(energy, parameters, retain_graph=True)
            direct_gradient = torch.autograd.grad(direct, parameters)
            gradient_error = max(float((first-second).abs().max()) for first, second in zip(gradient, direct_gradient))
            with torch.no_grad():
                rho = wave @ wave.T/total
                eigenvalues = torch.linalg.eigvalsh(rho)
                occupations = ((physical[:, None] >> torch.arange(2*sites, device=device)) & 1)
                n_label = occupations.sum(1)
                sz_label = (occupations[:, 0::2].sum(1)-occupations[:, 1::2].sum(1))/2
                spin_image = spin_matrix @ wave+sz_label[:, None].square()*wave
                spin_error = float((spin_image-(spin/2)*(spin/2+1)*wave).norm()/total.sqrt())
                number_error = float(((n_label-number)[:, None]*wave).norm()/total.sqrt())
                magnetic_error = float(((sz_label-spin/2)[:, None]*wave).norm()/total.sqrt())
                purity = float(rho.square().sum())
                energy_error = float((energy-direct).abs())
                minimum = float(eigenvalues.min())
                valid = (max(trace_difference, spin_error, number_error, magnetic_error, max(0., -minimum)) < protocol["physical_constraint_tolerance"]
                         and energy_error < protocol["operator_and_energy_tolerance_hartree"] and gradient_error < protocol["gradient_tolerance"] and purity < .999)
            record({"check": "mixed_state_physics_and_gradient", "electrons": number, "twice_spin": spin, "passed": valid,
                    "norm_difference": trace_difference, "energy_difference_hartree": energy_error,
                    "maximum_parameter_gradient_difference": gradient_error, "minimum_density_eigenvalue": minimum,
                    "purity": purity, "number_sector_error": number_error, "magnetic_sector_error": magnetic_error,
                    "total_spin_sector_error": spin_error})
            if not valid: raise RuntimeError("Purified state physical/gradient test failed")
            del state, cores, wave, rho, gradient, direct_gradient

        with torch.no_grad():
            pure = SpinNumberState(4, 4, channels=2, device=device)
            lifted = PurifiedSpinState(4, 4, channels=2, ancilla=1, device=device)
            for source, target in zip(pure.reduced, lifted.reduced): target[:, 0] = source
            _, local, wave = exhaustive_purification(lifted.cores())
            original = amplitudes(pure.cores(), local)
            special_case_error = float((wave[:, 0]-original).norm())
            record({"check": "pure_state_special_case", "state_difference": special_case_error,
                    "passed": special_case_error < protocol["physical_constraint_tolerance"]})
            fragments, fragment_words, fragment_coefficients, fragment_energies = [], [], [], []
            for index in range(8):
                state = PurifiedSpinState(3, 4, channels=2, ancilla=2, device=device); randomize(state)
                h, eri = random_hamiltonian(3, device)
                words, coefficients = molecular_words(h, eri, threshold=0.)
                local_mpo = FermionMPO(words, coefficients, 3)
                fragment_energies.append(local_mpo.expectation(state.cores()))
                shifted = words.clone(); shifted[:, :, 0] = torch.where(shifted[:, :, 0] >= 0, shifted[:, :, 0]+3*index, -1)
                fragments.append(state); fragment_words.append(shifted); fragment_coefficients.append(coefficients)
            embedded = compose_purifications(fragments[:3])
            direct_cores = [core for state in fragments[:3] for core in state.cores()]
            product_mpo = FermionMPO(torch.cat(fragment_words[:3]), torch.cat(fragment_coefficients[:3]), 9)
            embedded_energy_error = float((product_mpo.expectation(embedded.cores())-product_mpo.expectation(direct_cores)).abs())
            embedded_norm_error = float((purified_norm(embedded.cores())-purified_norm(direct_cores)).abs())
            record({"check": "same_family_fragment_embedding", "fragments": 3, "electrons": 12,
                    "energy_difference_hartree": embedded_energy_error, "norm_difference": embedded_norm_error,
                    "passed": max(embedded_energy_error, embedded_norm_error) < protocol["operator_and_energy_tolerance_hartree"]})
            del embedded, direct_cores, product_mpo
            cores = [core for state in fragments for core in state.cores()]
            product_mpo = FermionMPO(torch.cat(fragment_words), torch.cat(fragment_coefficients), 24)
            energy = product_mpo.expectation(cores)
            error = float((energy-torch.stack(fragment_energies).sum()).abs())
            dimensions = comb(24, 16)**2
            record({"check": "independent_fragment_product_without_global_vector", "fragments": 8, "sites": 24,
                    "electrons": 32, "energy_additivity_error_hartree": error,
                    "hypothetical_fixed_sz_dimension": dimensions, "hypothetical_global_vector_bytes": dimensions*8,
                    "physical_tensor_parameter_count": sum(value.numel() for state in fragments for value in state.parameters()),
                    "maximum_state_bond_dimension": max(core.shape[-1] for core in cores),
                    "passed": error < protocol["operator_and_energy_tolerance_hartree"], **product_mpo.summary()})
    except Exception as exc:
        report["failures"].append({"error": f"{type(exc).__name__}: {exc}"})
        print(json.dumps(report["failures"][-1]), flush=True)
    report["passed"] = bool(report["results"]) and not report["failures"] and all(row["passed"] for row in report["results"])
    report["status"] = "completed"
    report["peak_cuda_bytes"] = torch.cuda.max_memory_allocated()
    report["scope"] = "Representation/operator/gradient tests only; synthetic Hamiltonians and supplied independent fragments, no chemical-accuracy or automatic size-consistency claim"
    output.write_text(json.dumps(report, indent=2)+"\n")
    if not report["passed"]: raise RuntimeError("New scope backend did not pass its physical/operator gate")


if __name__ == "__main__": main()
