"""Accuracy-only test of charge/spin-complete compositional state blocks."""
from .runtime import configure, torch_setup
ROOT = configure(gpu_only_compatible=True)

from datetime import datetime, timezone
import json
from pathlib import Path
import time
import torch
from .gpu_chemistry_transfer import cuda_operators
from .gpu_four_electron import transform_eri
from .gpu_manybody_ops import fixed_sector, molecular_words, dense_operator
from .gpu_canonical_density import slater_vector
from .gpu_four_particle import FourParticleOperator, cached_ritz
from .gpu_six_particle import SixParticleOperator, SixParticleFockAction
from .gpu_fock_blocks import SectorCache, ProductGroups, allowed_multiplets, MultipletBlock, ChargeSpinProductProjector, spin_lower_norm

KCAL = 627.509474


@torch.no_grad()
def local_gauge(data):
    if "local_h" in data:
        return data["local_h"], data["local_eri"], data["reference_orbitals"]
    mapping = data["orbital_map"]
    values, vectors = torch.linalg.eigh(mapping @ mapping.T)
    orth = (vectors*values.sqrt()) @ vectors.T
    rotation = torch.linalg.solve(orth, mapping)
    return rotation @ data["h"] @ rotation.T, transform_eri(data["eri"], rotation.T), rotation


@torch.no_grad()
def prepare(case, device):
    destination = ROOT / "data/charge_spin_accuracy" / f"{case['id']}.pt"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return torch.load(destination, map_location=device, weights_only=True)
    if "source_folder" in case:
        source = torch.load(ROOT / "data" / case["source_folder"] / f"{case['id']}.pt", map_location=device, weights_only=True)
    else:
        physical = {key: case[key] for key in ("id", "atom", "charge", "spin", "electrons")}
        source = cuda_operators(physical, case["basis"])
    h, eri, orbitals = local_gauge(source)
    physical = source["case"]
    slices = source.get("atom_ao_slices")
    if slices is None:
        from pyscf import gto
        mol = gto.M(atom=physical["atom"], unit="Angstrom", basis=source["basis"],
                    charge=physical["charge"], spin=physical["spin"], verbose=0)
        slices = mol.aoslice_by_atom()[:, 2:4].tolist()  # Basis metadata only; no CPU integral call.
    split = slices[case["first_block_atoms"]-1][1]
    electrons = physical["electrons"]
    bits, _ = fixed_sector(len(h), electrons//2, electrons//2, device)
    data = {"h": h, "eri": eri, "orbitals": orbitals, "nuclear": source["nuclear"],
            "case": physical, "basis": source["basis"], "split": split, "bits": bits,
            "input_reference_status": source["reference_status"]}
    if "kinetic" in source:
        data["kinetic"] = orbitals @ source["kinetic"] @ orbitals.T
    if min(electrons, 2*len(h)-electrons) not in (4, 6):
        words, coefficients = molecular_words(h, eri)
        data["matrix"] = dense_operator(words, coefficients, bits, len(h))+source["nuclear"]*torch.eye(len(bits), device=device)
    torch.save({key: value.cpu() if isinstance(value, torch.Tensor) else value for key, value in data.items()}, destination)
    return data


class PhysicalAction:
    def __init__(self, data):
        self.data = data
        self.six_action = None
        if "matrix" in data:
            self.operator = None
        elif min(data["case"]["electrons"], 2*len(data["h"])-data["case"]["electrons"]) == 6:
            self.operator = SixParticleOperator(data["h"], data["eri"], data["case"]["electrons"], data["nuclear"])
            self.six_action = SixParticleFockAction(self.operator, data["bits"])
        else:
            self.operator = FourParticleOperator(data["h"], data["eri"], data["case"]["electrons"], data["nuclear"])

    def __call__(self, vector):
        if self.operator is None:
            return self.data["matrix"] @ vector
        if self.six_action is not None:
            return self.six_action(vector)
        tensor = self.operator.from_fock(vector, self.data["bits"])
        return self.operator.to_fock(self.operator(tensor), self.data["bits"])


@torch.no_grad()
def model_prediction(data, protocol, warm_state=None, block_checkpoints=None):
    h, eri, bits, split = data["h"], data["eri"], data["bits"], data["split"]
    electrons, pairs = data["case"]["electrons"], data["case"]["electrons"]//2
    first_cache, second_cache = SectorCache(split, h.device), SectorCache(len(h)-split, h.device)
    groups = ProductGroups(bits, first_cache, second_cache, pairs, pairs)
    reference_orbitals = data["orbitals"][:, :pairs]
    determinant = slater_vector(reference_orbitals, torch.arange(pairs, device=h.device), bits)
    density = 2*reference_orbitals @ reference_orbitals.T
    first_environment, second_environment = torch.zeros_like(density), torch.zeros_like(density)
    first_environment[split:, split:] = density[split:, split:]
    second_environment[:split, :split] = density[:split, :split]
    def embedded(environment):
        return h+torch.einsum("ijkl,kl->ij", eri, environment)-.5*torch.einsum("ikjl,kl->ij", eri, environment)
    if block_checkpoints is not None:
        if warm_state is None:
            raise ValueError("Checkpoint continuation requires its saved physical state")
        first = MultipletBlock.from_checkpoint(h[:split, :split], eri[:split, :split, :split, :split], first_cache, block_checkpoints[0])
        second = MultipletBlock.from_checkpoint(h[split:, split:], eri[split:, split:, split:, split:], second_cache, block_checkpoints[1])
    else:
        multiplets = allowed_multiplets(split, len(h)-split, electrons)
        support = [groups.matrices(determinant, 0), groups.matrices(determinant, 1)]
        if warm_state is not None:
            for side in range(2):
                previous_support = groups.matrices(warm_state, side)
                support[side] = {key: torch.cat((value, previous_support[key]), -1) for key, value in support[side].items()}
        first = MultipletBlock(embedded(first_environment)[:split, :split], eri[:split, :split, :split, :split],
                               first_cache, multiplets[0], support[0], protocol["initial_low_energy_multiplets_per_sector"])
        second = MultipletBlock(embedded(second_environment)[split:, split:], eri[split:, split:, split:, split:],
                                second_cache, multiplets[1], support[1], protocol["initial_low_energy_multiplets_per_sector"])
    projector = ChargeSpinProductProjector(groups, first, second)
    retention = float((projector(determinant)-determinant).norm())
    if retention > 1e-8:
        raise RuntimeError(f"Charge/spin blocks did not retain their input determinant: {retention}")
    generic, _ = torch.linalg.qr(torch.randn(len(h), pairs, device=h.device), mode="reduced")
    generic_state = slater_vector(generic, torch.arange(pairs, device=h.device), bits)
    raw_seed = determinant+.03*generic_state
    raw_seed /= raw_seed.norm()
    recovery_error = None
    if warm_state is not None:
        recovery_error = float((projector(warm_state)-warm_state).norm())
        if recovery_error > 1e-8:
            raise RuntimeError(f"Schmidt recovery did not retain the saved physical state: {recovery_error}")
    seed = projector(raw_seed if warm_state is None else warm_state); seed /= seed.norm()
    initial_spin = float(spin_lower_norm(seed, bits, len(h), pairs, pairs))
    if initial_spin > 1e-7:
        raise RuntimeError(f"Local multiplet projector did not preserve the input singlet: {initial_spin}")
    physical = PhysicalAction(data)
    history = []
    for update in range(protocol["maximum_adaptive_updates"]+1):
        result = cached_ritz(lambda value: projector(physical(projector(value))), seed,
                             maximum=protocol["maximum_projected_ritz_dimension"],
                             tolerance=protocol["projected_state_residual_tolerance_hartree"], log_every=100)
        state = projector(result["state"]); state /= state.norm()
        image = physical(state)
        energy = state @ image
        full_residual = image-energy*state
        size = float(full_residual.norm())
        row = {"adaptive_update": update, "energy_hartree": float(energy), "full_state_residual": size,
               "projected_state_residual": result["residual"], "projected_dimension": projector.dimension()}
        history.append(row); print(json.dumps(row), flush=True)
        if size < protocol["full_state_residual_target_hartree"] or update == protocol["maximum_adaptive_updates"]:
            break
        leakage = full_residual-projector(full_residual)
        added_a = first.enrich(groups.matrices(leakage, 0), protocol["residual_directions_per_sector_per_update"], protocol["residual_source_threshold_hartree"])
        added_b = second.enrich(groups.matrices(leakage, 1), protocol["residual_directions_per_sector_per_update"], protocol["residual_source_threshold_hartree"])
        row["added_local_multiplets"] = [added_a, added_b]
        if not added_a and not added_b:
            break
        seed = state
    spin = float(spin_lower_norm(state, bits, len(h), pairs, pairs))
    reference_not_degraded = float(energy) <= float(determinant @ physical(determinant))+1e-8
    validity = retention < 1e-8 and spin < 1e-7 and reference_not_degraded and result["converged"]
    checkpoints = []
    for block in (first, second):
        checkpoints.append([{ "electrons": number, "twice_spin": spin,
                              "spin_basis": entry["spin_basis"].cpu(), "coordinates": entry["coordinates"].cpu(),
                              "source_support_added": entry["source_support_added"], "source_energies": entry["source_energies"].cpu()}
                            for (number, spin), entry in block.entries.items()])
    return {"state": state, "energy": energy, "physical_validity_passed": validity,
            "full_residual": size, "projected_residual": result["residual"], "input_retention_error": retention,
            "spin_lower_norm": spin, "initial_spin_lower_norm": initial_spin, "reference_not_degraded": reference_not_degraded,
            "projected_dimension": projector.dimension(), "charge_weights": projector.charge_weights(state),
            "local_multiplets": [first.summary(), second.summary()], "history": history,
            "unprojected_reference_seed": raw_seed, "saved_state_recovery_error": recovery_error,
            "block_checkpoints": checkpoints}


def main():
    device = torch_setup()
    protocol = json.loads(Path("research/charge_spin_accuracy_protocol_20261117.json").read_text())
    output = Path("research/charge_spin_accuracy_gpu_20261117.json")
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "device": device, "protocol": protocol,
              "results": [], "failures": []}
    for case in protocol["cases"]:
        started = time.perf_counter()
        try:
            data = prepare(case, device)
            print(json.dumps({"id": case["id"], "phase": "charge_spin_prediction", "electrons": data["case"]["electrons"]}), flush=True)
            prediction = model_prediction(data, protocol)
            # Reference energy is not accessed until the fixed prediction has stopped.
            if "source_report" in case:
                previous = json.loads(Path(case["source_report"]).read_text())
                reference_row = next(value for value in previous["results"] if value["id"] == case["id"])
                exact = reference_row["reference_energy_hartree"]
                reference_converged = reference_row.get("reference_converged", True)
                new_reference = False
            else:
                reference = cached_ritz(PhysicalAction(data), prediction["unprojected_reference_seed"], maximum=192, tolerance=1e-8)
                exact, reference_converged, new_reference = float(reference["energy"]), reference["converged"], True
            error = float((prediction["energy"]-exact)*KCAL) if reference_converged else None
            row = {"id": case["id"], "electrons": data["case"]["electrons"], "basis": data["basis"],
                   "spatial_orbitals": len(data["h"]), "full_fixed_sz_dimension": len(data["bits"]),
                   "energy_error_kcal": error, "below_one_kcal": error is not None and abs(error) < 1,
                   "total_energy_hartree": float(prediction["energy"]), "reference_energy_hartree": exact,
                   "reference_converged": reference_converged, "new_reference_generated": new_reference,
                   "reference_energy_fitted": False, "input_reference_status": data["input_reference_status"],
                   "seconds": time.perf_counter()-started}
            row.update({key: value for key, value in prediction.items() if key not in ("state", "energy", "unprojected_reference_seed", "block_checkpoints")})
            report["results"].append(row)
            torch.save({"state": prediction["state"].cpu(), "block_checkpoints": prediction["block_checkpoints"], "report": row}, ROOT / "data/charge_spin_accuracy" / f"{case['id']}.model.pt")
            print(json.dumps({key: value for key, value in row.items() if key not in ("history", "local_multiplets")}), flush=True)
        except Exception as exc:
            row = {"id": case["id"], "error": f"{type(exc).__name__}: {exc}"}
            report["failures"].append(row); print(json.dumps(row), flush=True)
        output.write_text(json.dumps(report, indent=2)+"\n")
    lookup = {row["id"]: row for row in report["results"]}
    report["balanced_contrast_errors_kcal"] = {}
    for label, first, second in (("H4_geometries", "H4_parallel_far", "H4_square_close"),
                                 ("BeH2_stretch", "BeH2_neutral_1.33", "BeH2_neutral_2.10"),
                                 ("NH3_geometries", "NH3_pyramidal", "NH3_planar")):
        if first in lookup and second in lookup and lookup[first]["energy_error_kcal"] is not None and lookup[second]["energy_error_kcal"] is not None:
            report["balanced_contrast_errors_kcal"][label] = lookup[second]["energy_error_kcal"]-lookup[first]["energy_error_kcal"]
    report["rejection_panel_passed"] = (not report["failures"] and len(report["results"]) == len(protocol["cases"])
                                        and all(row["below_one_kcal"] and row["physical_validity_passed"] for row in report["results"])
                                        and all(abs(value) < 1 for value in report["balanced_contrast_errors_kcal"].values()))
    report["peak_cuda_bytes"] = torch.cuda.max_memory_allocated()
    output.write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    main()
