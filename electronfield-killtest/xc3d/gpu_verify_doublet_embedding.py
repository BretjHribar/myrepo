"""Independent physical partial-trace and energy checks for doublet support."""
from .runtime import configure, torch_setup
ROOT = configure(gpu_only_compatible=True)

from datetime import datetime, timezone
import json
from pathlib import Path
import torch
from .gpu_doublet_embedding import DoubletEmbedding
from .gpu_fermion_mpo import FermionMPO
from .gpu_verify_purified_operator import random_hamiltonian, exhaustive_purification
from .physical_two_site_forward import solve_two_site
from .gpu_manybody_ops import molecular_words, spin_words, dense_operator, fixed_sector
from .gpu_three_particle import ThreeParticleOperator, ThreeParticleFockAction, ThreeElectronDoubletProjector


@torch.no_grad()
def main():
    output = Path("research/doublet_embedding_checks_gpu_20261117.json")
    if output.exists(): raise RuntimeError("Do not repeat the new doublet embedding check")
    protocol = json.loads(Path("research/doublet_embedding_checks_protocol_20261117.json").read_text())
    device = torch_setup(protocol["seed"])
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "device": device, "protocol": protocol,
              "status": "running", "history": [], "failures": []}
    def log(row):
        report["history"].append(row); print(json.dumps(row), flush=True)
        output.write_text(json.dumps(report, indent=2)+"\n")
    try:
        if not json.loads(Path("research/two_site_forward_checks_gpu_20261117.json").read_text())["passed"]:
            raise RuntimeError("The original changing-sector forward checks have not passed")
        if not json.loads(Path("research/ritz_start_guard_checks_gpu_20261117.json").read_text())["passed"]:
            raise RuntimeError("The new fixed-start spectral safeguard must pass first")
        domain = DoubletEmbedding(protocol["physical_sites"], protocol["physical_electrons"])
        h, eri = random_hamiltonian(domain.physical_sites, device)
        padded_h, padded_eri = domain.pad_integrals(h, eri)
        operator = FermionMPO.molecular(padded_h, padded_eri)
        result = solve_two_site(operator, domain.seed(device), protocol, callback=log, problem_builder=domain.problem)
        state = result["state"]
        density, checks = domain.physical_checks(state)
        # Independent physical wavefunction and physical-sector reference
        # work occurs only after the algorithm's frozen search and audit.
        _, _, full_wave = exhaustive_purification(state.cores)
        physical_wave = full_wave[:, 0].reshape(4, 4**domain.physical_sites).T
        physical_ensemble = physical_wave @ physical_wave.T
        auxiliary_ensemble = physical_wave.T @ physical_wave
        expected_auxiliary = torch.diag(h.new_tensor([0., .5, .5, 0.]))
        auxiliary_error = float((auxiliary_ensemble-expected_auxiliary).norm())
        population = torch.linalg.eigvalsh(physical_ensemble)
        positivity_error = float((-population.min()).clamp_min(0.))
        trace_error = float((physical_ensemble.trace()-1.).abs())
        words, coefficient = molecular_words(h, eri, threshold=0.)
        physical_bits = torch.arange(4**domain.physical_sites, device=device)
        full_h = dense_operator(words, coefficient, physical_bits, domain.physical_sites)
        energy_error = float(((physical_ensemble*full_h).sum()-result["electronic_energy_hartree"]).abs())
        sites = torch.arange(domain.physical_sites, device=device)
        up = ((physical_bits[:, None] >> (2*sites[None])) & 1).sum(1)
        down = ((physical_bits[:, None] >> (2*sites[None]+1)) & 1).sum(1)
        invalid_number_weight = float(physical_ensemble.diag()[(up+down) != domain.physical_electrons].sum().abs())
        sw, sc = spin_words(domain.physical_sites, device)
        full_spin = dense_operator(sw, sc, physical_bits, domain.physical_sites)+torch.diag(((up-down)/2).square())
        spin_difference = full_spin-.75*torch.eye(len(physical_bits), device=device)
        spin_variance = float((physical_ensemble*(spin_difference @ spin_difference)).sum().abs())
        bits, _ = fixed_sector(domain.physical_sites, 2, 1, device)
        sector_h = dense_operator(words, coefficient, bits, domain.physical_sites)
        sector_spin = dense_operator(sw, sc, bits, domain.physical_sites)+.25*torch.eye(len(bits), device=device)
        independent_action = ThreeParticleFockAction(ThreeParticleOperator(h, eri), bits)
        projector = ThreeElectronDoubletProjector(bits, domain.physical_sites)
        identity = torch.eye(len(bits), device=device)
        action_matrix = torch.stack([independent_action(column) for column in identity.T], 1)
        projector_matrix = torch.stack([projector(column) for column in identity.T], 1)
        reference_action_error = float((action_matrix-sector_h).abs().max())
        spin_action_error = float((torch.stack([projector.spin_squared(column) for column in identity.T], 1)-sector_spin).abs().max())
        projector_error = float((projector_matrix @ projector_matrix-projector_matrix).norm())
        expected_projector = (3.75*identity-sector_spin)/3
        projector_error = max(projector_error, float((projector_matrix-expected_projector).norm()))
        spins, rotation = torch.linalg.eigh(sector_spin)
        doublets = rotation[:, (spins-.75).abs() < 1e-10]
        exact = torch.linalg.eigvalsh(doublets.T @ sector_h @ doublets)[0]
        reference_error = abs(float(exact)-result["electronic_energy_hartree"])
        identity_maximum = max(auxiliary_error, positivity_error, trace_error, invalid_number_weight, spin_variance,
                               reference_action_error, spin_action_error, projector_error)
        valid = (result["coordinate_stationarity_passed"] and checks["passed"]
                 and max(energy_error, reference_error) < protocol["independent_energy_tolerance_hartree"]
                 and identity_maximum < protocol["independent_state_tolerance"])
        report["result"] = {"passed": valid, "physical_checks": checks,
            "independent_physical_reference_energy_error_hartree": reference_error,
            "independent_partial_trace_energy_error_hartree": energy_error,
            "auxiliary_maximally_mixed_spin_error": auxiliary_error,
            "physical_ensemble_positivity_error": positivity_error, "physical_ensemble_trace_error": trace_error,
            "physical_wrong_number_probability": invalid_number_weight, "physical_spin_casimir_variance": spin_variance,
            "three_particle_reference_operator_error_hartree": reference_action_error,
            "three_particle_spin_action_error": spin_action_error, "doublet_projector_error": projector_error,
            "maximum_common_context_gap_hartree": result["maximum_common_context_gap_hartree"],
            "reference_computed_after_prediction": True, "chemical_accuracy_claimed": False,
            "global_xc_functional_validated": False, "wtmad2_measured": False}
        log({"phase": "doublet_embedding_check_completed", **report["result"]})
    except Exception as exc:
        report["failures"].append({"error": f"{type(exc).__name__}: {exc}"}); log({"phase": "failure", **report["failures"][-1]})
    report["status"] = "completed"; report["passed"] = not report["failures"] and report.get("result", {}).get("passed", False)
    report["peak_cuda_bytes"] = torch.cuda.max_memory_allocated()
    output.write_text(json.dumps(report, indent=2)+"\n")
    if not report["passed"]: raise RuntimeError("The new doublet domain failed its independent physical/reference gate")


if __name__ == "__main__": main()
