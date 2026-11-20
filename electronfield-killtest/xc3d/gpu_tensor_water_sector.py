"""Reference-free test of higher-excitation sectors in the larger basis."""
from .runtime import configure, torch_setup
ROOT = configure(gpu_only_compatible=True)

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import torch
from .gpu_fermion_mpo import FermionMPO
from .gpu_tensor_gauge import rebase_bond
from .gpu_two_site_tensor import pair_problem, TwoSiteSpinBasis, enrich_left_support, prefix_number_probability
from .physical_tensor_forward import restore, payload, update_site, physical_checks
from .gpu_tensor_density import density_matrix
from .gpu_tensor_water_conditional import atomic_save


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    protocol = json.loads(Path("research/tensor_water_sector_protocol_20261117.json").read_text())
    output = Path("research/tensor_water_sector_gpu_20261117.json")
    if output.exists():
        report = json.loads(output.read_text())
        if not args.resume or report["status"] not in ("running", "interrupted") or report["protocol"] != protocol:
            raise RuntimeError("Do not repeat a terminal sector experiment or change an interrupted protocol")
    else:
        if args.resume: raise RuntimeError("There is no sector experiment to resume")
        report = {"created_utc": datetime.now(timezone.utc).isoformat(), "device": "cuda", "protocol": protocol,
                  "status": "running", "history": [], "failures": []}
    device = torch_setup(protocol["seed"])
    folder = ROOT / protocol["checkpoint_folder"]; folder.mkdir(parents=True, exist_ok=True)
    checkpoint_path = folder / "checkpoint.pt"
    def log(row):
        report["history"].append(row); print(json.dumps(row), flush=True)
        output.write_text(json.dumps(report, indent=2)+"\n")
    def save(value):
        value["protocol"] = protocol
        atomic_save(value, checkpoint_path)
        return value
    output.write_text(json.dumps(report, indent=2)+"\n")
    try:
        for name in ("two_site_tensor_checks_gpu_20261117.json", "sector_enrichment_checks_gpu_20261117.json",
                     "tensor_gauge_direct_recovery_gpu_20261117.json"):
            if not json.loads(Path("research", name).read_text())["passed"]: raise RuntimeError(f"Missing gate: {name}")
        data = torch.load(ROOT / protocol["source_input"], map_location=device, weights_only=True)
        center = protocol["center"]
        log({"phase": "building_literal_operator", "orbitals": protocol["orbitals"], "electrons": protocol["electrons"]})
        mpo = FermionMPO.molecular(data["h"], data["eri"])
        if checkpoint_path.exists():
            if not args.resume: raise RuntimeError("Resume the existing sector checkpoint explicitly")
            run = torch.load(checkpoint_path, map_location=device, weights_only=True)
            if run["protocol"] != protocol: raise RuntimeError("Checkpoint protocol changed")
            log({"phase": "resuming", "stage": run["stage"]})
        else:
            state = restore(torch.load(ROOT / protocol["source_state"], map_location=device, weights_only=True))
            if len(state.cores) != protocol["orbitals"] or state.electrons != protocol["electrons"]:
                raise RuntimeError("The new sector protocol does not match its physical source")
            state, gauge = rebase_bond(state, center+1, "left")
            run = save({"stage": "response", "state": payload(state), "gauge": gauge})
        if run["stage"] == "response":
            state = restore(run["state"])
            log({"phase": "building_complete_two_site_section", "old_bond_dimension": len(state.labels[center+1])})
            problem = pair_problem(mpo, state, center, protocol["maximum_environment_bytes"])
            log({"phase": "evaluating_physical_residual", "physical_dimension": problem["basis"].dimension,
                 "merged_operator_blocks": problem["merged_operator_blocks"], "canonical_error": problem["canonical_error"]})
            seed = problem["seed"]; image = problem["action"](seed)
            energy = seed @ image; residual = image-energy*seed
            literal_energy = mpo.expectation(state.cores)
            if float((literal_energy-energy).abs()) > protocol["physical_identity_tolerance_hartree"]:
                raise RuntimeError("Two-site Hamiltonian residual differs from the literal physical energy")
            run.update({"stage": "enrich", "target_parameters": residual*problem["basis"].inverse_norm,
                        "source_energy_hartree": float(literal_energy), "source_density": density_matrix(state),
                        "two_site_residual_norm_hartree": float(residual.norm()), "pair_dimension": problem["basis"].dimension})
            save(run); del problem, image, residual
        if run["stage"] == "enrich":
            state = restore(run["state"])
            basis = TwoSiteSpinBasis(state.labels[center], state.labels[center+2], device)
            enriched, support = enrich_left_support(state, center, basis, run["target_parameters"],
                maximum_bond_dimension=protocol["maximum_bond_dimension"],
                maximum_added_magnetic_states=protocol["maximum_added_magnetic_states"])
            energy_change = float((mpo.expectation(enriched.cores)-run["source_energy_hartree"]).abs())
            density_change = float((density_matrix(enriched)-run["source_density"]).norm())
            if max(energy_change, density_change) > protocol["physical_identity_tolerance_hartree"]:
                raise RuntimeError("Sector embedding failed literal energy/density retention")
            support.update({"retained_energy_change_hartree": energy_change, "retained_density_change": density_change})
            log({"phase": "new_sectors_installed_without_changing_state", **support})
            run.update({"stage": "update", "enriched": payload(enriched), "support": support}); save(run)
        if run["stage"] == "update":
            log({"phase": "physical_variational_update", "center": center+1})
            updated, update = update_site(mpo, restore(run["enriched"]), center+1, protocol)
            run.update({"stage": "verify", "updated": payload(updated), "update": update}); save(run)
        updated = restore(run["updated"])
        density, checks = physical_checks(updated, protocol["physical_tolerance"])
        probability = prefix_number_probability(updated, center+1, protocol["electrons"]-3)
        lowering = run["source_energy_hartree"]-run["update"]["electronic_energy_hartree"]
        old_probability = prefix_number_probability(restore(run["state"]), center+1, protocol["electrons"]-3)
        valid = checks["passed"] and lowering > 0. and old_probability == 0. and probability > protocol["minimum_new_sector_probability"]
        report["result"] = {"physical_sector_test_passed": valid, "support": run["support"], "local_update": run["update"],
                            "physical_checks": checks, "two_site_physical_dimension": run["pair_dimension"],
                            "two_site_residual_norm_hartree": run["two_site_residual_norm_hartree"],
                            "source_total_energy_hartree": run["source_energy_hartree"]+float(data["nuclear"]),
                            "physical_total_energy_hartree": run["update"]["electronic_energy_hartree"]+float(data["nuclear"]),
                            "energy_lowering_hartree": lowering, "old_three_or_more_virtual_electron_probability": old_probability,
                            "new_three_or_more_virtual_electron_probability": probability,
                            "chemical_accuracy_measured": False, "new_reference_generated_or_loaded": False,
                            "global_tensor_stationarity_certified": False, "full_hilbert_residual_certified": False,
                            "goal_accuracy_achieved": False, "wtmad2_measured": False}
        atomic_save({**payload(updated), "density": density, "result": report["result"]}, folder / "sector_state.pt")
        log({"phase": "sector_test_completed", **report["result"]})
        report["passed"] = valid
        run["stage"] = "completed"; save(run)
    except Exception as exc:
        report["failures"].append({"error": f"{type(exc).__name__}: {exc}"}); log({"phase": "failure", **report["failures"][-1]})
        report["passed"] = False
    report["status"] = "completed"; report["peak_cuda_bytes"] = torch.cuda.max_memory_allocated()
    output.write_text(json.dumps(report, indent=2)+"\n")
    if not report["passed"]: raise RuntimeError("The new physical sector experiment failed its prespecified gate")


if __name__ == "__main__": main()
