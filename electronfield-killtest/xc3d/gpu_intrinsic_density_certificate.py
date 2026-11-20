"""Intrinsic density-frame and kinetic completion checks on saved new states."""
from .runtime import configure, torch_setup
ROOT = configure(gpu_only_compatible=True)

from datetime import datetime, timezone
import argparse
import json
import math
from pathlib import Path
import torch
from .intrinsic_density_frame import density_frame, product_value_frame, spatial_product_values, project_operator, one_body_density
from .gpu_four_electron import transform_eri
from .gpu_ensemble_dual import ensemble_dual


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spatial", action="store_true")
    args = parser.parse_args()
    device = torch_setup()
    stem = "spatial" if args.spatial else "intrinsic"
    specification = json.loads(Path(f"research/{stem}_density_protocol_20261117.json").read_text())
    output = Path(f"research/{stem}_density_certificate_gpu_20261117.json")
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "device": device,
              "protocol": specification, "results": [], "failures": []}
    folder = ROOT / "data/charge_spin_accuracy"
    for name in specification["cases"]:
        try:
            data = torch.load(folder / f"{name}.pt", map_location=device, weights_only=True)
            saved = torch.load(folder / f"{name}.larger.pt", map_location=device, weights_only=True)
            validation_error = None
            if args.spatial:
                values = spatial_product_values(data["case"], data["basis"], device, specification["construction_seed"],
                                                specification["radial_count"], specification["directions_count"])
                frame = product_value_frame(values, specification["relative_null_threshold"])
                validation = spatial_product_values(data["case"], data["basis"], device, specification["validation_seed"],
                                                    specification["radial_count"], specification["directions_count"])
                validation /= validation.norm(dim=1, keepdim=True).clamp_min(1e-150)
                products = validation[:, :, None]*validation[:, None, :]
                coefficients = torch.einsum("aij,gij->ga", frame["operators"], products)
                residual = products-torch.einsum("ga,aij->gij", coefficients, frame["operators"])
                validation_error = float(residual.flatten(1).norm(dim=1).max())
            else:
                frame = density_frame(data["eri"], specification["relative_null_threshold"])
            operators = frame["operators"]
            density = one_body_density(saved["state"], data["bits"], len(data["h"]))
            target = torch.einsum("aij,ij->a", operators, density)
            external = data["h"]-data["kinetic"]
            coefficients, _, external_error = project_operator(external, frame)
            # The random rotation is a coordinate check of the same saved state
            # data, not a refit or a new chemical accuracy experiment.
            rotation, _ = torch.linalg.qr(torch.randn_like(data["h"]))
            rotated = (product_value_frame(values @ rotation, specification["relative_null_threshold"]) if args.spatial else
                       density_frame(transform_eri(data["eri"], rotation), specification["relative_null_threshold"]))
            rotated_back = rotation @ rotated["operators"] @ rotation.T
            overlap = torch.einsum("aij,bij->ab", operators, rotated_back)
            covariance = (torch.eye(len(operators), device=device)-overlap @ overlap.T).norm()
            beta = 2*len(density)*math.log(2)/specification["kinetic_entropy_allowance_hartree"]
            occupations, natural = torch.linalg.eigh(density)
            effective = (natural*((2/occupations.clamp(1e-14, 2-1e-14)-1).log()/beta)) @ natural.T
            initial = project_operator(effective-data["kinetic"], frame)[0]
            kinetic = ensemble_dual(data["kinetic"], operators, target, beta,
                                    statistics="onebody", initial=initial)
            ks_density = (kinetic["vectors"]*kinetic["population"]) @ kinetic["vectors"].T
            interaction_functional = (saved["state"].new_tensor(saved["report"]["total_energy_hartree"])
                                      -data["nuclear"]-(external*density).sum())
            hartree_matrix = torch.einsum("ijkl,kl->ij", data["eri"], ks_density)
            hartree = .5*(ks_density*hartree_matrix).sum()
            xc = interaction_functional-kinetic["dual"]-hartree
            total = ((data["kinetic"]*ks_density).sum()+hartree+xc
                     +(external*ks_density).sum()+data["nuclear"])
            same_density_total_error = total-saved["report"]["total_energy_hartree"]
            potential = torch.einsum("a,aij->ij", kinetic["potential"]-coefficients, operators)-hartree_matrix
            fock = data["h"]+hartree_matrix+potential
            commutator = (fock @ ks_density-ks_density @ fock).norm()
            valid = (float(external_error) < specification["external_span_tolerance_hartree"]
                     and kinetic["residual"] < specification["kinetic_density_residual_tolerance"]
                     and float(covariance) < 1e-5 and len(rotated["operators"]) == len(operators)
                     and float(commutator) < 1e-7
                     and (validation_error is None or validation_error < specification["density_product_validation_tolerance"])
                     and abs(float(same_density_total_error)) < specification["kinetic_entropy_allowance_hartree"]
                     and float((density.trace()-data["case"]["electrons"]).abs()) < 1e-8
                     and float(occupations.min()) > -1e-8 and float(occupations.max()) < 2+1e-8)
            row = {"id": name, "certificate_passed": valid, "density_frame_dimension": len(operators),
                   "symmetric_matrix_dimension": frame["symmetric_matrix_dimension"],
                   "retained_coulomb_modes": frame.get("retained_coulomb_modes"),
                   "retained_product_modes": frame.get("retained_product_modes"), "null_threshold": frame["null_threshold"],
                   "independent_density_product_validation_error": validation_error,
                   "external_span_error_hartree": float(external_error), "orbital_covariance_projector_error": float(covariance),
                   "rotated_frame_dimension": len(rotated["operators"]),
                   "number_error": float((density.trace()-data["case"]["electrons"]).abs()),
                   "minimum_occupation": float(occupations.min()), "maximum_occupation": float(occupations.max()),
                   "kinetic_density_residual": kinetic["residual"], "ks_commutator_residual": float(commutator),
                   "same_density_total_minus_state_hartree": float(same_density_total_error),
                   "kinetic_entropy_allowance_hartree": specification["kinetic_entropy_allowance_hartree"],
                   "xc_energy_hartree": float(xc), "interacting_search_repeated": False,
                   "new_reference_generated": False, "off_shell_derivative_tested": False,
                   "kinetic_history": kinetic["history"]}
            report["results"].append(row)
            torch.save({"frame": {key: value.cpu() if isinstance(value, torch.Tensor) else value for key, value in frame.items()},
                        "density": density.cpu(), "noninteracting_density": ks_density.cpu(),
                        "xc_potential": potential.cpu(), "report": row}, folder / f"{name}.{stem}_density.pt")
            print(json.dumps({key: value for key, value in row.items() if key != "kinetic_history"}), flush=True)
        except Exception as exc:
            row = {"id": name, "error": f"{type(exc).__name__}: {exc}"}
            report["failures"].append(row); print(json.dumps(row), flush=True)
        output.write_text(json.dumps(report, indent=2)+"\n")
    report["peak_cuda_bytes"] = torch.cuda.max_memory_allocated()
    output.write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    main()
