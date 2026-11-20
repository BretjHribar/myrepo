"""Off-shell larger-basis conditional functional, with bounded resumable state."""
from .runtime import configure, torch_setup
ROOT = configure(gpu_only_compatible=True)

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import torch
from .gpu_spin_response import SpinChain
from .intrinsic_density_frame import spatial_product_values, product_value_frame
from .tensor_auxiliary_model import UniversalTensorOperators


def cpu_payload(value):
    if isinstance(value, torch.Tensor): return value.detach().cpu()
    if isinstance(value, dict): return {key: cpu_payload(item) for key, item in value.items()}
    if isinstance(value, list): return [cpu_payload(item) for item in value]
    if isinstance(value, tuple): return tuple(cpu_payload(item) for item in value)
    return value


def atomic_save(value, destination):
    temporary = destination.with_suffix(destination.suffix+".tmp")
    torch.save(cpu_payload(value), temporary)
    temporary.replace(destination)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default="research/tensor_water_conditional_protocol_20261117.json")
    parser.add_argument("--output", default="research/tensor_water_conditional_gpu_20261117.json")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    protocol = json.loads(Path(args.protocol).read_text())
    output = Path(args.output)
    if protocol.get("status", "").startswith("closed"): raise RuntimeError("This conditional protocol is closed")
    if output.exists():
        report = json.loads(output.read_text())
        if not args.resume or report["status"] not in ("running", "interrupted"):
            raise RuntimeError("A terminal report exists, or resume was not requested; do not repeat this experiment")
        if report["protocol"] != protocol: raise RuntimeError("Resume must retain the original physical protocol")
    else:
        if args.resume: raise RuntimeError("No interrupted conditional report exists")
        report = {"created_utc": datetime.now(timezone.utc).isoformat(), "device": "cuda", "protocol": protocol,
                  "status": "running", "history": [], "failures": []}
    def log(phase, **values):
        row = {"phase": phase, **values}; report["history"].append(row)
        print(json.dumps(row), flush=True); output.write_text(json.dumps(report, indent=2)+"\n")
    device = torch_setup(protocol["frame_construction_seed"])
    cache = ROOT / protocol["cache_folder"]; cache.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2)+"\n")
    try:
        gate = json.loads(Path("research/local_tensor_functional_checks_gpu_20261117.json").read_text())
        recovery_gate = json.loads(Path("research/resolvent_checkpoint_checks_gpu_20261117.json").read_text())
        if not gate["passed"] or not recovery_gate["passed"]: raise RuntimeError("Conditional tensor or optimizer recovery gate has not passed")
        data = torch.load(ROOT / protocol["source_input"], map_location=device, weights_only=True)
        saved = torch.load(ROOT / protocol["source_state"], map_location=device, weights_only=True)
        density = torch.load(ROOT / protocol["source_density"], map_location=device, weights_only=True)["density"]
        context = SpinChain(saved["cores"], saved["labels"], saved["electrons"], saved["twice_spin"])
        frame_path = cache / "spatial_frame.pt"
        if frame_path.exists():
            frame = torch.load(frame_path, map_location=device, weights_only=True)
            log("restored_spatial_frame", operators=len(frame["operators"]))
        else:
            log("building_spatial_frame")
            values = spatial_product_values(data["case"], data["basis"], device, protocol["frame_construction_seed"]) @ data["reference_orbitals"]
            frame = product_value_frame(values)
            independent = spatial_product_values(data["case"], data["basis"], device, protocol["frame_validation_seed"]) @ data["reference_orbitals"]
            independent /= independent.norm(dim=1, keepdim=True).clamp_min(1e-300)
            products = (independent[:, :, None]*independent[:, None, :]).flatten(1)
            modes = frame["operators"].flatten(1)
            error = float((products-(products @ modes.T) @ modes).abs().max())
            if error > protocol["spatial_span_tolerance"]: raise RuntimeError(f"New spatial product frame failed independent points: {error}")
            frame["independent_point_span_error"] = error
            atomic_save(frame, frame_path)
            del values, independent, products
            log("spatial_frame_ready", operators=len(frame["operators"]), independent_point_span_error=error)
        moments = frame["operators"][1:]
        external = data["h"]-data["kinetic"]
        constant = external.trace()/len(external)
        potential = torch.einsum("aij,ij->a", moments, external)
        span_error = float((external-constant*torch.eye(len(external), device=device)-torch.einsum("a,aij->ij", potential, moments)).norm())
        if span_error > protocol["external_span_tolerance_hartree"]: raise RuntimeError(f"The fixed spatial frame misses the physical potential: {span_error}")
        auxiliary = torch.einsum("aij,ij->a", moments, density)-protocol["epsilon_inverse_hartree"]*potential
        universal = {"kinetic": data["kinetic"], "eri": data["eri"], "operators": frame["operators"],
                     "electrons": protocol["electrons"], "twice_spin": protocol["twice_spin"]}
        log("building_universal_tensor_operators", external_span_error_hartree=span_error)
        domain = UniversalTensorOperators(universal)
        compiled = None
        base_path = cache / "compiled_base.pt"
        if base_path.exists():
            if not args.resume: raise RuntimeError("A compiled conditional checkpoint exists; resume explicitly")
            compiled = torch.load(base_path, map_location=device, weights_only=True)
            compiled["density_chunks"] = [torch.load(path, map_location=device, weights_only=True) for path in sorted(cache.glob("density_*.pt"))]
            log("restored_compiled_section", chunks=len(compiled["density_chunks"]))
        def persist_compiler(value):
            if value["phase"] == "base":
                destination = base_path
            else:
                destination = cache / f"density_{value['start']:04d}_{value['stop']:04d}.pt"
            if destination.exists(): raise RuntimeError("Do not overwrite an immutable compiled operator chunk")
            atomic_save(value, destination)
        def compiler_progress(value):
            value = dict(value); phase = value.pop("phase"); log(phase, **value)
        section = domain.section(context, protocol["center"], protocol, compiler_progress, compiled, persist_compiler)
        def solve_at(point, label, initial=None):
            destination = cache / f"{label}.optimizer.pt"
            checkpoint = torch.load(destination, map_location=device, weights_only=True) if args.resume and destination.exists() else None
            log("conditional_search", point=label)
            return section.evaluate(point, initial=initial, checkpoint=checkpoint,
                checkpoint_callback=lambda value: atomic_save(value, destination),
                callback=lambda row: log("conditional_progress", point=label, **row))
        solved = solve_at(auxiliary, "central")
        atomic_save(solved, cache / "central_result.pt")
        if not solved["converged"]: raise RuntimeError(f"Conditional chemical search failed its original bounded gap: {solved['gap']}")
        installed, installation = section.install(solved)
        old_universal = (section.ensemble*section.matrix).sum()
        old_moments = torch.einsum("aij,ij->a", moments, density)
        old_energy = old_universal+potential @ old_moments+constant*domain.number+data["nuclear"]
        physical_energy = solved["universal_energy"]+potential @ solved["moments"]+constant*domain.number+data["nuclear"]
        recovered = solved["functional"]+potential @ auxiliary+protocol["epsilon_inverse_hartree"]*potential.square().sum()/2+constant*domain.number+data["nuclear"]
        mismatch = solved["moments"]-auxiliary-protocol["epsilon_inverse_hartree"]*potential
        identity_error = float((recovered-physical_energy-mismatch.square().sum()/(2*protocol["epsilon_inverse_hartree"])).abs())
        if identity_error > protocol["physical_energy_identity_tolerance_hartree"] or float(physical_energy) > float(old_energy)+1e-7:
            raise RuntimeError("Conditional update failed physical energy recovery or variational monotonicity")
        kinetic = section.kinetic(auxiliary, density)
        xc = section.xc(auxiliary, solved, kinetic)
        atomic_save({"cores": installed.cores, "labels": installed.labels, "electrons": installed.electrons,
                     "twice_spin": installed.twice_spin, "interacting": solved, "kinetic": kinetic,
                     "auxiliary": auxiliary, "installation": installation}, cache / "installed_state.pt")
        log("conditional_central_result", physical_total_energy_hartree=float(physical_energy),
            original_total_energy_hartree=float(old_energy), local_gap_hartree=solved["gap"],
            physical_energy_identity_error_hartree=identity_error, **installation)
        generator = torch.Generator(device=device).manual_seed(protocol["derivative_seed"])
        direction = torch.randn(auxiliary.shape, device=device, generator=generator); direction /= direction.norm()
        values = []
        for sign, label in ((-1, "minus"), (1, "plus")):
            changed = auxiliary+sign*protocol["derivative_step"]*direction
            interacting = solve_at(changed, label, solved["state"])
            kinetic_changed = section.kinetic(changed, kinetic["state"])
            changed_xc = section.xc(changed, interacting, kinetic_changed)
            values.append(changed_xc["energy"])
        finite = (values[1]-values[0])/(2*protocol["derivative_step"])
        analytic = xc["gradient"] @ direction
        derivative_error = float((finite-analytic).abs())
        result = {"conditional_functional_passed": derivative_error < protocol["derivative_tolerance_hartree"],
                  "local_physical_dimension": section.space["physical_dimension"], "density_coordinate_count": len(moments),
                  "conditional_gap_hartree": solved["gap"], "kinetic_gap_hartree": kinetic["gap"],
                  "physical_total_energy_hartree": float(physical_energy), "original_total_energy_hartree": float(old_energy),
                  "physical_energy_identity_error_hartree": identity_error, "derivative_error_hartree": derivative_error,
                  "finite_difference": float(finite), "analytic_directional_derivative": float(analytic),
                  "installation": installation, "external_potential_passed_to_universal_evaluator": False,
                  "global_tensor_stationarity_certified": False, "self_consistent_global_xc_model": False,
                  "chemical_accuracy_measured": False, "new_reference_generated_or_loaded": False,
                  "wtmad2_measured": False, "goal_accuracy_achieved": False}
        report["result"] = result; log("conditional_derivative_result", **result)
    except Exception as exc:
        report["failures"].append({"error": f"{type(exc).__name__}: {exc}"}); log("failure", **report["failures"][-1])
    report["status"] = "completed"
    report["passed"] = not report["failures"] and report.get("result", {}).get("conditional_functional_passed", False)
    report["peak_cuda_bytes"] = torch.cuda.max_memory_allocated()
    output.write_text(json.dumps(report, indent=2)+"\n")
    if not report["passed"]: raise RuntimeError("Larger-basis conditional functional did not pass its declared gates")


if __name__ == "__main__": main()
