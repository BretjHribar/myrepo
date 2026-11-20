"""Check the new physical metric against full small-system images and errors."""
from .runtime import configure, torch_setup
ROOT = configure(gpu_only_compatible=True)

from datetime import datetime, timezone
import json
from pathlib import Path
import torch
from .gpu_fermion_mpo import FermionMPO
from .gpu_spin_response import product_reference
from .gpu_physical_response_metric import physical_product_response
from .gpu_manybody_ops import molecular_words, dense_operator, spin_words
from .gpu_verify_purified_operator import random_hamiltonian, exhaustive_purification


@torch.no_grad()
def main():
    output = Path("research/physical_response_metric_checks_gpu_20261117.json")
    if output.exists(): raise RuntimeError("This changed physical-metric test already has a report")
    protocol = json.loads(Path("research/physical_response_metric_protocol_20261117.json").read_text())
    device = torch_setup(protocol["seed"])
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "device": device, "protocol": protocol,
              "status": "running", "results": [], "failures": []}
    try:
        sites = protocol["sites"]
        h, eri = random_hamiltonian(sites, device)
        words, coefficients = molecular_words(h, eri, threshold=0.)
        mpo = FermionMPO(words, coefficients, sites)
        bits = torch.arange(4**sites, device=device)
        dense = dense_operator(words, coefficients, bits, sites)
        sw, sc = spin_words(sites, device)
        spin_matrix = dense_operator(sw, sc, bits, sites)
        occupations = ((bits[:, None] >> torch.arange(2*sites, device=device)) & 1)
        n = occupations.sum(1)
        m = (occupations[:, 0::2].sum(1)-occupations[:, 1::2].sum(1))/2
        for case in protocol["states"]:
            number, spin = case["electrons"], case["twice_spin"]
            source = product_reference(sites, number, spin, device)
            _, _, wave = exhaustive_purification(source.cores)
            expected = dense @ wave
            for name, maximum in (("untruncated", 4**(sites-1)), ("bounded", protocol["bounded_bond"])):
                result, check = physical_product_response(mpo, source, maximum_bond=maximum,
                                                          singular_tolerance=protocol["singular_tolerance"],
                                                          frame_null_tolerance=protocol["right_frame_null_tolerance"],
                                                          maximum_gram_bytes=protocol["maximum_gram_bytes"],
                                                          maximum_frame_bytes=protocol["maximum_frame_bytes"])
                _, _, actual = exhaustive_purification(result.cores)
                difference = (actual-expected).norm()
                measured_squared = check["full_image_difference_squared_norm"]
                squared_error = float((difference.square()-measured_squared).abs())
                norm_error = float((expected.square().sum()-check["exact_full_image_squared_norm"]).abs())
                spin_image = spin_matrix @ actual+m[:, None].square()*actual
                spin_error = float((spin_image-(spin/2)*(spin/2+1)*actual).norm()/actual.norm())
                number_error = float(((n-number)[:, None]*actual).norm()/actual.norm())
                magnetic_error = float(((m-spin/2)[:, None]*actual).norm()/actual.norm())
                bound_valid = float(difference) <= check["sequential_projection_error_bound"]+protocol["absolute_identity_tolerance"]
                valid = max(squared_error, norm_error, spin_error, number_error, magnetic_error) < protocol["absolute_identity_tolerance"] and bound_valid
                if name == "untruncated": valid = valid and float(difference) < protocol["absolute_identity_tolerance"]
                row = {"case": case, "variant": name, "passed": valid,
                       "full_image_error_norm": float(difference), "image_squared_error_identity_difference": squared_error,
                       "exact_full_norm_identity_difference": norm_error, "spin_sector_error": spin_error,
                       "number_sector_error": number_error, "magnetic_sector_error": magnetic_error,
                       "projection_bound_valid": bound_valid, "diagnostics": check}
                report["results"].append(row)
                print(json.dumps({key: value for key, value in row.items() if key != "diagnostics"}), flush=True)
                output.write_text(json.dumps(report, indent=2)+"\n")
                if not valid: raise RuntimeError("Physical-norm response metric failed its independent image/sector test")
    except Exception as exc:
        report["failures"].append({"error": f"{type(exc).__name__}: {exc}"}); print(json.dumps(report["failures"][-1]), flush=True)
    report["status"] = "completed"
    report["passed"] = len(report["results"]) == 4 and not report["failures"] and all(row["passed"] for row in report["results"])
    report["peak_cuda_bytes"] = torch.cuda.max_memory_allocated()
    output.write_text(json.dumps(report, indent=2)+"\n")
    if not report["passed"]: raise RuntimeError("Physical response metric is not approved for molecular use")


if __name__ == "__main__": main()
