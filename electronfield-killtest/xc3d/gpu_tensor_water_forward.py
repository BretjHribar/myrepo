"""All-site physical/auxiliary tensor forward calculation in the larger basis."""
from .runtime import configure, torch_setup
ROOT = configure(gpu_only_compatible=True)

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import torch
from .gpu_fermion_mpo import FermionMPO
from .physical_tensor_forward import solve_forward, payload, restore
from .gpu_tensor_water_conditional import atomic_save
from .gpu_density_resolvent import proximal_state


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default="research/tensor_water_forward_protocol_20261117.json")
    parser.add_argument("--output", default="research/tensor_water_forward_gpu_20261117.json")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    protocol = json.loads(Path(args.protocol).read_text())
    output = Path(args.output)
    if protocol.get("status", "").startswith("closed"): raise RuntimeError("This physical forward protocol is closed")
    if output.exists():
        report = json.loads(output.read_text())
        if not args.resume or report["status"] not in ("running", "interrupted") or report["protocol"] != protocol:
            raise RuntimeError("Do not restart a terminal experiment or change the original interrupted protocol")
    else:
        if args.resume: raise RuntimeError("No interrupted report exists")
        report = {"created_utc": datetime.now(timezone.utc).isoformat(), "device": "cuda", "protocol": protocol,
                  "status": "running", "history": [], "failures": []}
    device = torch_setup(protocol["seed"])
    folder = ROOT / protocol["checkpoint_folder"]; folder.mkdir(parents=True, exist_ok=True)
    def log(value):
        report["history"].append(value); print(json.dumps(value), flush=True)
        output.write_text(json.dumps(report, indent=2)+"\n")
    output.write_text(json.dumps(report, indent=2)+"\n")
    try:
        gate = json.loads(Path("research/physical_tensor_forward_checks_gpu_20261117.json").read_text())
        if not gate["passed"]: raise RuntimeError("All-site physical forward checks have not passed")
        data = torch.load(ROOT / protocol["source_input"], map_location=device, weights_only=True)
        saved = torch.load(ROOT / protocol["source_state"], map_location=device, weights_only=True)
        frame = torch.load(ROOT / protocol["source_frame"], map_location=device, weights_only=True)
        initial = restore(saved)
        log({"phase": "building_literal_physical_operator", "sites": len(initial.cores), "electrons": initial.electrons})
        operator = FermionMPO.molecular(data["h"], data["eri"])
        checkpoint_path = folder / "forward_checkpoint.pt"
        checkpoint = None
        if checkpoint_path.exists():
            if not args.resume: raise RuntimeError("A physical forward checkpoint exists; resume explicitly")
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
            if checkpoint["protocol"] != protocol: raise RuntimeError("The saved forward protocol differs")
            checkpoint = checkpoint["run"]
            log({"phase": "resuming_saved_forward", "sweep": checkpoint["sweep"], "position": checkpoint["position"], "stage": checkpoint["stage"]})
        result = solve_forward(operator, initial, protocol, callback=log, checkpoint=checkpoint,
            checkpoint_callback=lambda run: atomic_save({"protocol": protocol, "run": run}, checkpoint_path))
        state, density = result["state"], result["density"]
        atomic_save({**payload(state), "density": density,
                     "report": {key: value for key, value in result.items() if key not in ("state", "density")}}, folder / "forward_state.pt")
        coordinate_valid = result["coordinate_stationarity_passed"] and result["physical_checks"]["passed"]
        on_shell = None
        if coordinate_valid:
            moments = frame["operators"][1:]
            external = data["h"]-data["kinetic"]
            constant = external.trace()/len(external)
            potential = torch.einsum("aij,ij->a", moments, external)
            q = torch.einsum("aij,ij->a", moments, density)
            auxiliary = q-protocol["epsilon_inverse_hartree"]*potential
            log({"phase": "stationary_auxiliary_completion"})
            universal = FermionMPO.molecular(data["kinetic"], data["eri"])
            energy0 = universal.expectation(state.cores)
            statewise = energy0+(q-auxiliary).square().sum()/(2*protocol["epsilon_inverse_hartree"])
            recovered = statewise+potential @ auxiliary+protocol["epsilon_inverse_hartree"]*potential.square().sum()/2+constant*state.electrons+data["nuclear"]
            total_energy = result["electronic_energy_hartree"]+data["nuclear"]
            identity_error = float((recovered-total_energy).abs())
            kinetic = proximal_state(data["kinetic"], moments, auxiliary, protocol["epsilon_inverse_hartree"], density,
                total=state.electrons, cap=2., maximum=protocol["kinetic_maximum_iterations"],
                gap_tolerance=protocol["kinetic_gap_tolerance_hartree"])
            auxiliary_density = torch.eye(len(density), device=device)*(state.electrons/len(density))+torch.einsum("a,aij->ij", auxiliary, moments)
            hartree = .5*torch.einsum("ij,ijkl,kl->", auxiliary_density, data["eri"], auxiliary_density)
            ks_matrix = data["kinetic"]-torch.einsum("a,aij->ij", kinetic["gradient"], moments)
            commutator = float((ks_matrix @ kinetic["state"]-kinetic["state"] @ ks_matrix).norm())
            on_shell = {"statewise_auxiliary_value_hartree": float(statewise), "stationary_branch_xc_scalar_hartree": float(statewise-kinetic["functional"]-hartree),
                        "physical_energy_recovery_error_hartree": identity_error, "kinetic_gap_hartree": kinetic["gap"],
                        "ks_commutator_norm": commutator, "kinetic_converged": kinetic["converged"],
                        "passed": kinetic["converged"] and identity_error < protocol["physical_identity_tolerance_hartree"] and commutator < protocol["ks_commutator_tolerance"],
                        "arbitrary_density_universal_minimum_certified": False, "global_xc_directional_derivative_validated": False}
            atomic_save({**payload(state), "density": density, "auxiliary": auxiliary,
                         "kinetic": kinetic, "on_shell": on_shell}, folder / "stationary_auxiliary_state.pt")
        report["result"] = {key: value for key, value in result.items() if key not in ("state", "density", "history")}
        report["result"].update({"physical_total_energy_hartree": result["electronic_energy_hartree"]+float(data["nuclear"]),
                                  "on_shell": on_shell, "chemical_accuracy_measured": False, "new_reference_generated_or_loaded": False,
                                  "wtmad2_measured": False, "goal_accuracy_achieved": False})
        report["passed"] = coordinate_valid and on_shell is not None and on_shell["passed"]
        log({"phase": "forward_completed", "passed": report["passed"], "maximum_common_context_gap_hartree": result["maximum_common_context_gap_hartree"],
             "physical_total_energy_hartree": report["result"]["physical_total_energy_hartree"]})
    except Exception as exc:
        report["failures"].append({"error": f"{type(exc).__name__}: {exc}"}); log({"phase": "failure", **report["failures"][-1]})
        report["passed"] = False
    report["status"] = "completed"; report["peak_cuda_bytes"] = torch.cuda.max_memory_allocated()
    output.write_text(json.dumps(report, indent=2)+"\n")
    if not report["passed"]: raise RuntimeError("All-site tensor forward did not pass its physical/coordinate gates")


if __name__ == "__main__": main()
