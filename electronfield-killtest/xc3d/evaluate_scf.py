"""Fresh self-consistent evaluation of our model, with every failure retained."""
from .runtime import configure, torch_setup
ROOT = configure()

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import types
import numpy as np
import torch
from pyscf import dft, gto, lib
from .model import CollectiveXC, SpatialTemplate, rho_from_dm
from .pair_state import PairStateXC
from .collective_pair import CollectivePairXC
from .screened_state import ScreenedStateXC, tau_from_dm
from .conserved_hole import ConservedHoleXC
from .quantum_features import overlap_roots, quantum_torch
from .ensemble_screened import PBEDensity, whitened_integrals, regional_hartree

KCAL = 627.509474


class Correction:
    def __init__(self, model, mf, device):
        self.model, self.mf, self.device = model, mf, device
        self.template = self.ao = None
        self.grid_id = None
        self.calls = 0
        self.timings = {"prepare": 0., "density_fields": 0., "energy_and_dm_gradient": 0.}

    def prepare(self):
        grids = self.mf.grids
        if self.grid_id != id(grids.coords):
            ao = dft.numint.eval_ao(self.mf.mol, grids.coords, deriv=1)
            self.ao = torch.tensor(ao, device=self.device)
            self.template = SpatialTemplate(*[
                torch.tensor(array, device=self.device)
                for array in (grids.coords, grids.weights, self.mf.mol.atom_coords())
            ])
            self.grid_id = id(grids.coords)

    def value(self, dm):
        self.prepare()
        array = np.asarray(dm)
        restricted = array.ndim == 2
        spin = np.stack((array / 2, array / 2)) if restricted else array
        tensor = torch.tensor(spin, device=self.device, requires_grad=True)
        rho = rho_from_dm(tensor, self.ao)
        energy = self.model(rho, self.template)
        potential = torch.autograd.grad(energy, tensor)[0]
        potential = (potential + potential.transpose(-1, -2)) / 2
        if restricted:
            potential = potential.mean(0)
        self.calls += 1
        return float(energy.detach()), potential.detach().cpu().numpy()


class PairCorrection(Correction):
    def prepare(self):
        previous = self.grid_id
        super().prepare()
        if previous != self.grid_id:
            self.eri = torch.tensor(self.mf.mol.intor("int2e", aosym="s1"), device=self.device)
            self.prepared = self.model.prepare(self.ao, self.template)

    def value(self, dm):
        self.prepare()
        array = np.asarray(dm)
        restricted = array.ndim == 2
        spin = np.stack((array / 2, array / 2)) if restricted else array
        tensor = torch.tensor(spin, device=self.device, requires_grad=True)
        energy = self.model(tensor, self.ao, self.template, self.eri, self.prepared)
        potential = torch.autograd.grad(energy, tensor)[0]
        potential = (potential + potential.transpose(-1, -2)) / 2
        if restricted:
            potential = potential.mean(0)
        self.calls += 1
        return float(energy.detach()), potential.detach().cpu().numpy()


class ScreenedCorrection(Correction):
    def prepare(self):
        previous = self.grid_id
        super().prepare()
        if previous != self.grid_id:
            self.prepared = self.model.prepare(self.template)
            if self.model.quantum_enabled:
                root, inverse = overlap_roots(self.mf.get_ovlp())
                self.overlap_root = torch.tensor(root, device=self.device)
                self.slices = self.mf.mol.aoslice_by_atom()[:, 2:4].tolist()
                if getattr(self.model, "needs_ensemble", False):
                    self.ensemble_integrals = torch.tensor(whitened_integrals(self.mf.mol, inverse), device=self.device)

    def value(self, dm):
        started = time.perf_counter()
        self.prepare()
        prepared_at = time.perf_counter()
        array = np.asarray(dm)
        restricted = array.ndim == 2
        spin = np.stack((array / 2, array / 2)) if restricted else array
        tensor = torch.tensor(spin, device=self.device, requires_grad=True)
        rho = rho_from_dm(tensor, self.ao)
        tau = tau_from_dm(tensor, self.ao)
        quantum = quantum_torch(tensor, self.overlap_root, self.slices) if self.model.quantum_enabled else None
        if getattr(self.model, "needs_ensemble", False):
            quantum["hartree_matrix"] = regional_hartree(quantum["projected"], self.ensemble_integrals, self.slices)
            quantum["pbe_density"] = PBEDensity.apply(rho)
        fields_at = time.perf_counter()
        energy = self.model(rho, tau, self.template, self.prepared, quantum=quantum)
        potential = torch.autograd.grad(energy, tensor)[0]
        potential = (potential + potential.transpose(-1, -2)) / 2
        if restricted:
            potential = potential.mean(0)
        self.calls += 1
        self.timings["prepare"] += prepared_at-started
        self.timings["density_fields"] += fields_at-prepared_at
        self.timings["energy_and_dm_gradient"] += time.perf_counter()-fields_at
        return float(energy.detach()), potential.detach().cpu().numpy()


def attach(mf, model, device):
    original = mf.get_veff
    correction_class = (PairCorrection if isinstance(model, PairStateXC) else
                        ScreenedCorrection if isinstance(model, ScreenedStateXC) else Correction)
    correction = correction_class(model, mf, device)

    def get_veff(this, mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1):
        mol = this.mol if mol is None else mol
        dm = this.make_rdm1() if dm is None else dm
        baseline = original(mol, dm, dm_last, vhf_last, hermi)
        ec, potential = correction.value(dm)
        return lib.tag_array(np.asarray(baseline) + potential,
                             ecoul=baseline.ecoul, exc=baseline.exc + ec,
                             vj=getattr(baseline, "vj", None), vk=getattr(baseline, "vk", None))

    mf.get_veff = types.MethodType(get_veff, mf)
    return correction


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=ROOT / "data/pilot_3d_ccpvdz")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=24)
    parser.add_argument("--family", default=None)
    args = parser.parse_args()
    device = torch_setup()
    lib.num_threads(1)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    pair_state = checkpoint.get("architecture") in ("pair_state", "collective_pair", "conserved_hole")
    pair_class = {"pair_state": PairStateXC, "collective_pair": CollectivePairXC,
                  "conserved_hole": ConservedHoleXC}.get(checkpoint.get("architecture"), PairStateXC)
    model = (pair_class(checkpoint["width"]) if pair_state else
             CollectiveXC(checkpoint["width"], nonlocal_state=checkpoint["nonlocal_state"])).to(device)
    model.load_state_dict(checkpoint["model"])
    model.requires_grad_(False)
    report = {"created_utc": datetime.now(timezone.utc).isoformat(),
              "status": "Fresh SCF, finite-basis tiny-system pilot; not GMTKN55 or challenge-suite evidence",
              "checkpoint": str(args.checkpoint), "device": device, "cases": []}
    evaluated = 0
    for path in sorted(args.data.glob("*.npz")):
        with np.load(path) as data:
            metadata = json.loads(str(data["metadata"]))
            exact = float(data["energy"][0])
            baseline = float(data["energy_pbe"][0])
        if args.family is not None and metadata["family"] != args.family:
            continue
        if evaluated >= args.limit:
            break
        evaluated += 1
        start = time.perf_counter()
        row = {"id": metadata["id"], "split": metadata["split"], "reference_hartree": exact,
               "pbe_error_kcal": (baseline - exact) * KCAL}
        try:
            mol = gto.M(atom=metadata["atom"], basis=metadata["basis"],
                        charge=metadata["charge"], spin=metadata["spin"], verbose=0, max_memory=1800)
            mf = dft.UKS(mol) if mol.spin else dft.RKS(mol)
            mf.xc = "0*LDA_X" if pair_state else "pbe"
            mf.grids.level = metadata["grid_level"]
            mf.conv_tol = 1e-9
            mf.max_cycle = 80
            mf.damp = .25
            correction = attach(mf, model, device)
            energy = mf.kernel()
            first_converged = bool(mf.converged)
            if not first_converged:
                mf.damp = .6
                mf.diis_start_cycle = 8
                energy = mf.kernel(dm0=mf.make_rdm1())
            dm = mf.make_rdm1()
            fock = mf.get_hcore() + np.asarray(mf.get_veff(mol, dm))
            overlap = mf.get_ovlp()
            commutator = fock @ dm @ overlap - overlap @ dm @ fock
            row.update({"energy_hartree": float(energy), "error_kcal": (float(energy) - exact) * KCAL,
                        "converged": bool(mf.converged), "first_attempt_converged": first_converged,
                        "correction_calls": correction.calls,
                        "commutator_norm": float(np.linalg.norm(commutator)),
                        "scf_cycles_last_attempt": getattr(mf, "cycles", None)})
        except Exception as exc:
            row.update({"converged": False, "exception": f"{type(exc).__name__}: {exc}"})
        if device == "cuda":
            torch.cuda.synchronize()
        row["seconds_including_setup_and_retries"] = time.perf_counter() - start
        report["cases"].append(row)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(row), flush=True)
    summary = {}
    for split in ("train", "val", "test"):
        rows = [row for row in report["cases"] if row["split"] == split]
        if rows:
            summary[split] = {
                "cases": len(rows), "failures": sum(not row["converged"] for row in rows),
                "all_returned_energy_mae_kcal": float(np.mean([abs(row["error_kcal"]) for row in rows]))
                    if all("error_kcal" in row for row in rows) else None,
                "pbe_mae_kcal": float(np.mean([abs(row["pbe_error_kcal"]) for row in rows])),
                "median_seconds": float(np.median([row["seconds_including_setup_and_retries"] for row in rows])),
            }
    report["summary"] = summary
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"summary": summary}), flush=True)


if __name__ == "__main__":
    main()
