"""First actual public predictions for the exact N=1 physical component."""
from .runtime import configure, torch_setup
ROOT = configure(gpu_only_compatible=True)

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import torch
from .gpu_verify_gmtkn55_coordinates import read_coord


@torch.no_grad()
def predict(structure, protocol, pinned_files):
    import cupy as cp
    from pyscf import gto
    from gpu4pyscf.scf.hf import get_hcore
    from gpu4pyscf.pbc.gto.int1e import int1e_ovlp
    relative = f"{structure['id']}/coord"
    path = ROOT/"data/gmtkn55_official_v1/data"/relative
    if hashlib.sha256(path.read_bytes()).hexdigest() != pinned_files[relative]["sha256"]:
        raise ValueError("Pinned model geometry changed")
    symbols, coordinates, units = read_coord(path)
    coords = torch.tensor(coordinates, device="cuda")
    if units == "angstrom": coords /= coords.new_tensor(.52917721092)
    # All unit arithmetic is CUDA. Copying coordinates back supplies molecule
    # metadata to the GPU integral engine; it is not a CPU energy calculation.
    atoms = list(zip([name.capitalize() for name in symbols], coords.tolist()))
    mol = gto.M(atom=atoms, unit="Bohr", basis=protocol["basis"], charge=structure["charge"], spin=1, verbose=0)
    if mol.nelectron != 1 or mol._ecp: raise ValueError("Unexpected physical electron/ECP domain")
    def forbidden(*args, **kwargs):
        raise RuntimeError("CPU molecular integrals/energy are forbidden")
    old = gto.Mole.intor, gto.Mole.intor_symmetric, gto.Mole.intor_asymmetric, gto.Mole.energy_nuc
    gto.Mole.intor = gto.Mole.intor_symmetric = gto.Mole.intor_asymmetric = gto.Mole.energy_nuc = forbidden
    try:
        arrays = [get_hcore(mol), int1e_ovlp(mol)]
        if not all(isinstance(array, cp.ndarray) for array in arrays): raise RuntimeError("Non-CUDA integral output")
        h, overlap = [torch.from_dlpack(array) for array in arrays]
    finally:
        gto.Mole.intor, gto.Mole.intor_symmetric, gto.Mole.intor_asymmetric, gto.Mole.energy_nuc = old
    eigenvalues, vectors = torch.linalg.eigh(overlap)
    if bool(eigenvalues.min() <= protocol["minimum_overlap_eigenvalue"]):
        raise RuntimeError("Full target basis failed the unchanged overlap guard; no basis truncation")
    orth = (vectors*eigenvalues.rsqrt()) @ vectors.T
    physical_h = orth.T @ h @ orth
    levels, orbitals = torch.linalg.eigh((physical_h+physical_h.T)/2)
    orbital = orbitals[:, 0]
    density = orbital[:, None]*orbital[None]
    residual = (physical_h @ orbital-levels[0]*orbital).norm()
    metric = (orth.T @ overlap @ orth-torch.eye(len(h), device=h.device)).abs().max()
    occupations = torch.linalg.eigvalsh(density)
    number_error = (density.trace()-1).abs()
    independent_scalar_error = ((density*physical_h).sum()-levels[0]).abs()
    charges = torch.tensor(mol.atom_charges().tolist(), device=h.device, dtype=h.dtype)
    distance = torch.cdist(coords, coords, compute_mode="donot_use_mm_for_euclid_dist").clamp_min(1e-30)
    nuclear = torch.triu(charges[:, None]*charges[None]/distance, diagonal=1).sum()
    energy = levels[0]+nuclear
    physical = bool(torch.isfinite(energy) & (metric < protocol["physical_tolerance"])
                    & (number_error < protocol["physical_tolerance"]) & (independent_scalar_error < protocol["physical_tolerance"])
                    & (occupations.min() >= -protocol["physical_tolerance"]) & (occupations.max() <= 1+protocol["physical_tolerance"]))
    converged = bool(residual < protocol["orbital_residual_tolerance_hartree"])
    diagnostics = {"physical_electrons": 1, "physical_twice_spin": 1,
        "physical_spin_squared": .75, "spin_justification": "Exact one-electron sector has only spin 1/2",
        "number_error": float(number_error), "orthonormal_metric_error": float(metric),
        "orbital_residual_hartree": float(residual), "independent_density_energy_error_hartree": float(independent_scalar_error),
        "minimum_density_occupation": float(occupations.min()), "maximum_density_occupation": float(occupations.max()),
        "basis_functions": len(h), "basis_truncation_performed": False, "cpu_integrals_disabled": True,
        "nuclear_energy_hartree": float(nuclear), "reference_energy_used_for_prediction": False}
    if not physical or not converged:
        return {"id": structure["id"], "status": "failed", "reason": "Frozen one-electron physical/convergence gate failed", "diagnostics": diagnostics}
    return {"id": structure["id"], "status": "accepted", "total_energy_hartree": float(energy),
            "converged": True, "physical_checks_passed": True, "diagnostics": diagnostics}


def main():
    output = Path("research/gmtkn55_n1_predictions_gpu_20261117.json")
    destination = ROOT/"data/gmtkn55_pipeline/N1_component_predictions.json"
    if output.exists() or destination.exists(): raise RuntimeError("Do not repeat completed public predictions")
    protocol_path = Path("research/gmtkn55_n1_protocol_20261117.json")
    protocol = json.loads(protocol_path.read_text())
    device = torch_setup(protocol["seed"])
    inputs = json.loads((ROOT/protocol["inputs"]).read_text())
    document = json.loads((ROOT/protocol["coverage_inputs"]).read_text())
    manifest_path = ROOT/"data/gmtkn55_official_v1/manifest.json"
    if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != inputs["manifest_sha256"]:
        raise ValueError("Input manifest changed")
    manifest = json.loads(manifest_path.read_text())
    pinned = {row["path"]: row for row in manifest["files"]}
    source_hashes = {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in protocol["source_files"]}
    document["candidate"] = {"id": protocol["candidate_id"], "basis": protocol["basis"],
        "protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256(json.dumps(source_hashes, sort_keys=True).encode()).hexdigest()}
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "device": device, "protocol": protocol,
        "candidate": document["candidate"], "source_files_sha256": source_hashes, "status": "running", "results": [], "failures": []}
    positions = {row["id"]: i for i, row in enumerate(document["predictions"])}
    for structure in inputs["structures"]:
        if structure["physical_electrons"] != 1: continue
        try:
            result = predict(structure, protocol, pinned)
        except Exception as exc:
            result = {"id": structure["id"], "status": "failed", "reason": f"{type(exc).__name__}: {exc}"}
        report["results"].append(result)
        document["predictions"][positions[structure["id"]]] = result
        if result["status"] != "accepted": report["failures"].append(result)
        destination.write_text(json.dumps(document, indent=2)+"\n")
        output.write_text(json.dumps(report, indent=2)+"\n")
        print(json.dumps(result), flush=True)
    report.update(status="completed", accepted_predictions=sum(row["status"] == "accepted" for row in report["results"]),
        peak_cuda_bytes=torch.cuda.max_memory_allocated(), full_public_suite_score_available=False,
        full_suite_point_estimate_kcal_per_mol=None, private_tests_passed=False, speed_work_performed=False)
    output.write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__": main()
