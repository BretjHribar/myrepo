"""Actual public target-basis validation/probes of the bounded GPU factors."""
from .runtime import configure, torch_setup
ROOT = configure(gpu_only_compatible=True)

from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import torch
from .gpu_coulomb_factors import GPUCoulombFactorSource, cuda_integrals_only
from .gpu_verify_gmtkn55_coordinates import read_coord
from .gpu_four_electron import transform_eri


def metadata_molecule(structure, protocol, pinned):
    from pyscf import gto
    relative = f"{structure['id']}/coord"
    path = ROOT/"data/gmtkn55_official_v1/data"/relative
    if hashlib.sha256(path.read_bytes()).hexdigest() != pinned[relative]["sha256"]:
        raise RuntimeError("Pinned public geometry changed")
    symbols, coordinates, units = read_coord(path)
    values = torch.tensor(coordinates, device="cuda")
    if units == "angstrom": values /= values.new_tensor(.52917721092)
    atoms = list(zip([symbol.capitalize() for symbol in symbols], values.tolist()))
    with cuda_integrals_only():
        return gto.M(atom=atoms, unit="Bohr", charge=structure["charge"], spin=structure["unpaired_electrons"], basis=protocol["basis"], verbose=0)


def source_for(mol, protocol):
    names = ("pair_batch_size", "auxiliary_batch_size", "factor_chunk_size", "maximum_working_bytes", "screening_cutoff", "metric_reconstruction_relative_tolerance")
    return GPUCoulombFactorSource(mol, protocol["auxiliary_basis"], **{name: protocol[name] for name in names})


@torch.no_grad()
def main():
    import cupy as cp
    from gpu4pyscf.df import int3c2e_bdiv
    from gpu4pyscf.scf.int4c2e import get_int4c2e
    from gpu4pyscf.scf.hf import get_hcore
    from gpu4pyscf.pbc.gto.int1e import int1e_ovlp
    output = Path("research/coulomb_factor_source_checks_gpu_20261117.json")
    if output.exists(): raise RuntimeError("Preserve terminal factor-source evidence; do not repeat")
    protocol = json.loads(Path("research/coulomb_factor_source_protocol_20261117.json").read_text())
    device = torch_setup(protocol["seed"])
    inputs = json.loads((ROOT/"data/gmtkn55_pipeline/prediction_inputs.json").read_text())
    manifest = json.loads((ROOT/"data/gmtkn55_official_v1/manifest.json").read_text())
    pinned = {row["path"]: row for row in manifest["files"]}
    structures = {row["id"]: row for row in inputs["structures"]}
    domain = json.loads(Path("research/gmtkn55_model_domain_audit_20261117.json").read_text())
    largest = sorted(domain["structures"], key=lambda row: (-row["spatial_basis_functions"], row["id"]))[0]
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "device": device, "protocol": protocol,
              "status": "running", "history": [], "failures": [], "probes": [], "results": {}}
    def log(row):
        report["history"].append(row); print(json.dumps(row), flush=True)
        output.write_text(json.dumps(report, indent=2)+"\n")
    folder = ROOT/"data/coulomb_factor_source"; folder.mkdir(exist_ok=True)
    try:
        name = protocol["small_public_validation_id"]
        log({"id": name, "phase": "bounded_factor_source_setup"})
        mol = metadata_molecule(structures[name], protocol, pinned)
        source = source_for(mol, protocol)
        log({"id": name, "phase": "factor_source_metadata", **source.summary()})
        # Separate dense three-center implementation is confined to this
        # prespecified small validation, never used by the production source.
        with cuda_integrals_only():
            dense_b = torch.from_dlpack(int3c2e_bdiv.aux_e2(mol, source.auxmol))
            metric = torch.from_dlpack(int3c2e_bdiv.int2c2e(source.auxmol))
            exact_eri = torch.from_dlpack(get_int4c2e(mol))
            h = torch.from_dlpack(get_hcore(mol)); overlap = torch.from_dlpack(int1e_ovlp(mol))
        direct_root = torch.linalg.cholesky(metric)
        dense_f = torch.linalg.solve_triangular(direct_root, dense_b.reshape(source.sites**2, source.count).T, upper=False)
        factor_error, symmetry_error = h.new_zeros(()), h.new_zeros(())
        df_eri = torch.zeros_like(exact_eri)
        factors_checked = 0
        for start, values in source.chunks():
            expected = dense_f[start:start+len(values)].reshape_as(values)
            factor_error = torch.maximum(factor_error, (values-expected).abs().max())
            symmetry_error = torch.maximum(symmetry_error, (values-values.transpose(-1, -2)).abs().max())
            df_eri += torch.einsum("pij,pkl->ijkl", values, values)
            factors_checked += len(values)
            if start % 40 == 0: log({"id": name, "phase": "all_factor_comparison", "factors_checked": factors_checked, "total": source.count})
        eigenvalues, orbitals = torch.linalg.eigh(overlap)
        if bool(eigenvalues.min() <= 1e-10): raise RuntimeError("The full physical basis is singular; do not truncate")
        orth = (orbitals*eigenvalues.rsqrt()) @ orbitals.T
        _, core = torch.linalg.eigh(orth.T @ h @ orth)
        occupied = structures[name]["physical_electrons"]//2
        coefficients = orth @ core[:, :occupied]
        density = 2*coefficients @ coefficients.T
        def interaction(eri):
            return .5*torch.einsum("ijkl,ij,kl->", eri, density, density)-.25*torch.einsum("ikjl,ij,kl->", eri, density, density)
        energy_difference = (interaction(df_eri)-interaction(exact_eri))*protocol["hartree_to_kcal_per_mol"]
        orth_error = transform_eri(exact_eri-df_eri, orth)
        pairs = structures[name]["physical_electrons"]*(structures[name]["physical_electrons"]-1)//2
        bound = pairs*orth_error.norm()*protocol["hartree_to_kcal_per_mol"]
        passed = bool((factor_error < protocol["factor_comparison_tolerance"]) & (symmetry_error < protocol["factor_symmetry_tolerance"])) and factors_checked == source.count
        report["results"] = {"id": name, "physical_electrons": structures[name]["physical_electrons"],
            "basis_functions": source.sites, "all_factors_checked": factors_checked,
            "independent_factor_maximum_difference": float(factor_error), "factor_symmetry_error": float(symmetry_error),
            "producer_implementation_passed": passed, "provided_determinant_fitting_energy_difference_kcal_per_mol": float(energy_difference),
            "all_N_electron_states_conservative_fitting_energy_bound_kcal_per_mol": float(bound),
            "bound_definition": "N(N-1)/2 times the Frobenius norm of the two-particle fitting error in the full orthonormal physical basis; this also bounds every normalized N-electron state's energy error from this fitting approximation",
            "source": source.summary(), "correlated_molecular_accuracy_measured": False, "full_suite_accuracy_estimate_available": False}
        log({"phase": "small_molecular_input_check_completed", **report["results"]})
        if not passed: raise RuntimeError("The blocked factor producer differs from independent dense physical factors")
        del mol, source, dense_b, metric, exact_eri, h, overlap, dense_f, df_eri, orth_error, values
        gc.collect(); torch.cuda.empty_cache(); cp.get_default_memory_pool().free_all_blocks()
        for name in protocol["additional_probe_ids"]+[largest["id"]]:
            log({"id": name, "phase": "larger_public_input_probe_setup"})
            try:
                mol = metadata_molecule(structures[name], protocol, pinned)
                source = source_for(mol, protocol)
                log({"id": name, "phase": "larger_source_ready", **source.summary()})
                source.progress_callback = lambda row: log({"id": name, "phase": "streaming_global_factor", **row})
                for start, values in source.chunks(source.count-1, source.count):
                    norm = values.norm()
                    if not bool(torch.isfinite(norm)) or not bool(norm > 0): raise RuntimeError("Invalid/nonphysical global factor")
                    path = folder/f"{name.replace('/', '__')}.final_factor.pt"
                    torch.save({"structure_id": name, "factor_index": start, "values": values.cpu(),
                                "source": source.summary(), "scope": "One actual input factor, not a molecular energy/prediction"}, path)
                report["probes"].append({"id": name, "completed": True, "physical_electrons": structures[name]["physical_electrons"],
                    "basis_functions": source.sites, "factor_count": source.count,
                    "factors_generated": 1, "full_molecular_energy_evaluated": False, "source": source.summary()})
                log({"id": name, "phase": "single_global_factor_probe_completed", "basis_functions": source.sites})
                del mol, source, values
                gc.collect(); torch.cuda.empty_cache(); cp.get_default_memory_pool().free_all_blocks()
            except Exception as exc:
                failure = {"id": name, "error": f"{type(exc).__name__}: {exc}"}
                report["failures"].append(failure); log(failure)
    except Exception as exc:
        failure = {"phase": "producer_validation", "error": f"{type(exc).__name__}: {exc}"}
        report["failures"].append(failure); log(failure)
    report.update(status="completed", passed=not report["failures"] and report["results"].get("producer_implementation_passed", False) and len(report["probes"]) == 3,
        peak_torch_cuda_bytes=torch.cuda.max_memory_allocated(), wtmad2_measured=False, full_public_model_predictions=0)
    output.write_text(json.dumps(report, indent=2)+"\n")
    if not report["passed"]: raise RuntimeError("Full frozen factor-source validation/probes incomplete; retain every failure")


if __name__ == "__main__": main()
