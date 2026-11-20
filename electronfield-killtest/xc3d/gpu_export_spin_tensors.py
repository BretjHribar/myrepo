"""Export physical state tensors without refitting or reference generation."""
from .runtime import configure, torch_setup
ROOT = configure(gpu_only_compatible=True)

from datetime import datetime, timezone
import json
from pathlib import Path
import torch
from .gpu_fock_blocks import SectorCache, ProductGroups, MultipletBlock, spin_lower_norm
from .gpu_spin_pair_tensor import SpinCoupledPairTensor
from .gpu_charge_spin_accuracy import PhysicalAction


@torch.no_grad()
def main():
    device = torch_setup()
    cases = [("H4_square_close", ".continued.pt"), ("H2O2_skew_new", ".model.pt"),
             ("H2F_plus_new", ".stationary.pt")]
    folder = ROOT / "data/charge_spin_accuracy"
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "device": device,
              "scope": "Exact two-block charge/spin tensor representation; no accuracy fitting, reference generation or timing benchmark",
              "results": [], "failures": []}
    output = Path("research/spin_tensor_state_checks_gpu_20261117.json")
    for name, suffix in cases:
        try:
            data = torch.load(folder / f"{name}.pt", map_location=device, weights_only=True)
            saved = torch.load(folder / f"{name}{suffix}", map_location=device, weights_only=True)
            split, h, eri = data["split"], data["h"], data["eri"]
            left_cache, right_cache = SectorCache(split, device), SectorCache(len(h)-split, device)
            left = MultipletBlock.from_checkpoint(h[:split, :split], eri[:split, :split, :split, :split], left_cache, saved["block_checkpoints"][0])
            right = MultipletBlock.from_checkpoint(h[split:, split:], eri[split:, split:, split:, split:], right_cache, saved["block_checkpoints"][1])
            particles = data["case"]["electrons"]//2
            groups = ProductGroups(data["bits"], left_cache, right_cache, particles, particles)
            representation = SpinCoupledPairTensor.from_fock(saved["state"], groups, left, right)
            reconstructed = representation.to_fock(groups)
            norm_error = float((representation.norm_squared()-1).abs())
            state_error = float((reconstructed-saved["state"]).norm())
            action = PhysicalAction(data)
            original_energy = saved["state"] @ action(saved["state"])
            tensor_energy = reconstructed @ action(reconstructed)/reconstructed.square().sum()
            energy_error = float((original_energy-tensor_energy).abs())
            spin = float(spin_lower_norm(reconstructed, data["bits"], len(h), particles, particles))
            valid = norm_error < 1e-8 and state_error < 1e-8 and energy_error < 1e-8 and spin < 1e-7
            parameters = sum(value["first_highest"].numel()+value["second_highest"].numel()+value["singular_values"].numel()
                             for value in representation.sectors)
            row = {"id": name, "electrons": data["case"]["electrons"], "representation_valid": valid,
                   "norm_error": norm_error, "state_l2_error": state_error, "energy_difference_hartree": energy_error,
                   "spin_lower_norm": spin, "global_fock_coefficients": len(saved["state"]),
                   "highest_multiplet_and_bond_parameters": parameters, "sectors": representation.summary()}
            report["results"].append(row)
            torch.save({"state_tensor": representation.payload(), "report": row}, folder / f"{name}.spin_tensor.pt")
            print(json.dumps(row), flush=True)
        except Exception as exc:
            row = {"id": name, "error": f"{type(exc).__name__}: {exc}"}
            report["failures"].append(row); print(json.dumps(row), flush=True)
        output.write_text(json.dumps(report, indent=2)+"\n")
    report["peak_cuda_bytes"] = torch.cuda.max_memory_allocated()
    output.write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    main()
