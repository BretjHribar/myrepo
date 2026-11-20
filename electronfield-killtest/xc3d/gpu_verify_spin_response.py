"""Independent exact-image and spin-sector checks for tensor response."""
from .runtime import configure, torch_setup
ROOT = configure(gpu_only_compatible=True)

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import torch
from .gpu_fermion_mpo import FermionMPO
from .gpu_purified_spin import PurifiedSpinState, purified_norm
from .gpu_spin_response import SpinChain, apply_physical, overlap
from .gpu_manybody_ops import molecular_words, dense_operator, spin_words
from .gpu_verify_purified_operator import random_hamiltonian, randomize, exhaustive_purification


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="research/spin_response_checks_gpu_20261117.json")
    parser.add_argument("--finish-test-capacity", action="store_true",
                        help="Complete only unfinished checks after the nominal untruncated test cap proved too small")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists(): raise RuntimeError("This bounded response check already has a report")
    protocol = json.loads(Path("research/spin_response_protocol_20261117.json").read_text())
    device = torch_setup(protocol["seed"])
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "device": device, "protocol": protocol,
              "results": [], "failures": [], "status": "running"}
    if args.finish_test_capacity:
        previous = json.loads(Path("research/spin_response_checks_gpu_20261117.json").read_text())
        failed = [row for row in previous["results"] if not row["passed"]]
        if (len(failed) != 1 or failed[0]["case"] != protocol["state_cases"][1]
                or failed[0]["variant"] != "untruncated" or not failed[0]["capacity_limited"]):
            raise RuntimeError("The original report does not match the verified test-capacity failure")
        report["results"] = [row for row in previous["results"] if row["passed"]]
        report["previous_report"] = "spin_response_checks_gpu_20261117.json"
        report["previous_failed_test_case"] = failed[0]
        report["repair"] = "Use the algebraic full-prefix row-space bound (4*ancilla)^(sites-1), rather than calling a cap of 256 untruncated; no molecule, objective, seed, algorithm or tolerance changed"
    completed = {(row["case"]["electrons"], row["case"]["twice_spin"], row["variant"]) for row in report["results"]}
    try:
        sites = protocol["sites"]
        h, eri = random_hamiltonian(sites, device)
        words, coefficient = molecular_words(h, eri, threshold=0.)
        mpo = FermionMPO(words, coefficient, sites)
        bits = torch.arange(4**sites, device=device)
        dense = dense_operator(words, coefficient, bits, sites)
        sw, sc = spin_words(sites, device)
        spin_matrix = dense_operator(sw, sc, bits, sites)
        occupations = ((bits[:, None] >> torch.arange(2*sites, device=device)) & 1)
        number_label = occupations.sum(1)
        magnetic = (occupations[:, 0::2].sum(1)-occupations[:, 1::2].sum(1))/2
        for case in protocol["state_cases"]:
            number, spin = case["electrons"], case["twice_spin"]
            state = PurifiedSpinState(sites, number, spin, channels=protocol["channels_per_sector"], ancilla=protocol["ancilla"], device=device)
            randomize(state)
            # Reconstruct the identical RNG sequence for the unfinished case,
            # without repeating any completed image or physical check.
            variants = [(label, maximum) for label, maximum in (("untruncated", protocol["untruncated_maximum_bond"]), ("bounded", protocol["bounded_maximum_bond"]))
                        if (number, spin, label) not in completed]
            if not variants: continue
            chain = SpinChain.from_state(state).normalized()
            _, _, wave = exhaustive_purification(chain.cores)
            exact_image = dense @ wave
            for label, maximum in variants:
                if label == "untruncated" and args.finish_test_capacity:
                    maximum = (4*protocol["ancilla"])**(sites-1)
                result, diagnostics = apply_physical(mpo, chain, maximum_bond=maximum,
                                                      relative_tolerance=protocol["relative_singular_tolerance"],
                                                      maximum_raw_bytes=protocol["maximum_raw_intermediate_bytes"])
                _, _, image = exhaustive_purification(result.cores)
                magnitude = image.norm()
                image_error = float((image-exact_image).norm()/exact_image.norm())
                number_error = float(((number_label-number)[:, None]*image).norm()/magnitude)
                magnetic_error = float(((magnetic-spin/2)[:, None]*image).norm()/magnitude)
                spin_image = spin_matrix @ image+magnetic[:, None].square()*image
                spin_error = float((spin_image-(spin/2)*(spin/2+1)*image).norm()/magnitude)
                norm_error = float((purified_norm(result.cores)-image.square().sum()).abs())
                cross_error = float((mpo.contract(result.cores, bra=chain.cores)-(wave*(dense @ image)).sum()).abs())
                overlap_error = float((overlap(chain, result)-(wave*image).sum()).abs())
                valid = max(number_error, magnetic_error, spin_error, norm_error, cross_error, overlap_error) < protocol["operator_and_physical_tolerance"]
                if label == "untruncated": valid = valid and image_error < protocol["operator_and_physical_tolerance"]
                row = {"case": case, "variant": label, "passed": valid, "relative_image_difference": image_error,
                       "number_sector_error": number_error, "magnetic_sector_error": magnetic_error,
                       "total_spin_sector_error": spin_error, "norm_difference": norm_error,
                       "different_bond_cross_hamiltonian_error_hartree": cross_error, "cross_overlap_error": overlap_error,
                       "maximum_state_bond": diagnostics["maximum_bond"], "capacity_limited": diagnostics["capacity_limited"],
                       "maximum_isometry_error": max(item["isometry_error"] for item in diagnostics["history"]),
                       "compression_indicator_is_global_error_bound": False}
                report["results"].append(row); print(json.dumps(row), flush=True)
                output.write_text(json.dumps(report, indent=2)+"\n")
                if not valid: raise RuntimeError("Physical tensor response failed its exact-image or sector check")
    except Exception as exc:
        report["failures"].append({"error": f"{type(exc).__name__}: {exc}"}); print(json.dumps(report["failures"][-1]), flush=True)
    report["status"] = "completed"
    report["passed"] = not report["failures"] and len(report["results"]) == 4 and all(row["passed"] for row in report["results"])
    report["peak_cuda_bytes"] = torch.cuda.max_memory_allocated()
    output.write_text(json.dumps(report, indent=2)+"\n")
    if not report["passed"]: raise RuntimeError("Spin response is not approved for molecular use")


if __name__ == "__main__": main()
