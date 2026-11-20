"""Our local charge-and-spin correlation flow, with exact spin-exchange gates.

Charge-only gates cannot recouple spins in a singly occupied region. This
changes the physical operator family, not a loss weight: every gate includes
sum_{p<q} Q(p)Q(q) S_p.S_q alongside its charge operator. For three electrons
the spin gate is exponentiated exactly in 3x3 permutation blocks on CUDA.
"""
from .runtime import configure, torch_setup
ROOT = configure(gpu_only_compatible=True)

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import torch
from .gpu_sector_flow import SectorFlow, rotate_block, optimize


class SpinChargeFlow(SectorFlow):
    def __init__(self, data):
        super().__init__(data, layers=3)
        eigenvalues, prefactors = [], []
        for spectrum in self.orbital_spectrum:
            first, second, third = spectrum[:, None, None], spectrum[None, :, None], spectrum[None, None, :]
            ab, ac, bc = torch.broadcast_tensors(first*second, first*third, second*third)
            diagonal = -.25*(ab+ac+bc)
            matrix = torch.stack((torch.stack((diagonal+.5*ab, -.5*bc, .5*ac), -1),
                                  torch.stack((-.5*bc, diagonal+.5*ac, -.5*ab), -1),
                                  torch.stack((.5*ac, -.5*ab, diagonal+.5*bc), -1)), -2)
            values, vectors = torch.linalg.eigh(matrix)
            eigenvalues.append(values)
            prefactors.append(vectors[..., 0, None, :]*vectors)
        self.register_buffer("spin_spectrum", torch.stack(eigenvalues))
        self.register_buffer("spin_prefactor", torch.stack(prefactors))
        self.spin_raw = torch.nn.Parameter(torch.zeros_like(self.raw))

    def forward(self):
        states = self.seed
        charge_angles = 3*torch.tanh(self.raw)*self.mode_weight
        # Spin entanglement can remain finite between degenerate separated
        # spin fragments; it must not be suppressed as if it were charge transfer.
        spin_angles = 2*torch.tanh(self.spin_raw)
        for layer_index, (charge_layer, spin_layer) in enumerate(zip(charge_angles, spin_angles)):
            for index, (theta, eta, rotation, diagonal) in enumerate(zip(charge_layer, spin_layer, self.rotation, self.pair_diagonal)):
                states = rotate_block(states, rotation.T)
                states = states*torch.exp(-theta*diagonal)
                exponential = torch.exp(-eta*self.spin_spectrum[index])
                row = (self.spin_prefactor[index]*exponential[..., None, :]).sum(-1)
                states = (row[..., 0]*states+row[..., 1]*states.transpose(2, 3)
                          +row[..., 2]*states.permute(0, 3, 1, 2))
                states = rotate_block(states, rotation)
                states = self.orbital_update(states, layer_index, index)
            states = states/states.square().sum((1, 2, 3), keepdim=True).sqrt()
        return states

    def orbital_update(self, states, layer, mode):
        return states


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--output", type=Path, default=Path("research/spin_charge_flow_gpu_20261116.json"))
    args = parser.parse_args()
    device = torch_setup()
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "device": device,
              "scope": "Original connected spin/charge operator family, first three-electron finite-basis tests; broad OOS accuracy unproven",
              "training_targets": "Variational Hamiltonian expectation only, no reference-energy fit",
              "steps": args.steps, "cases": [], "failures": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for path in sorted((ROOT / "data/gpu_odd_operators").glob("*.flow_seed.pt")):
        try:
            report["cases"].append(optimize(path, args, device, model_class=SpinChargeFlow))
        except Exception as exc:
            failure = {"id": path.stem, "exception": f"{type(exc).__name__}: {exc}"}
            report["failures"].append(failure)
            print(json.dumps(failure), flush=True)
        args.output.write_text(json.dumps(report, indent=2)+"\n")
    if report["cases"]:
        errors = torch.tensor([row["error_kcal"] for row in report["cases"]], device=device)
        report["summary"] = {"cases": len(errors), "mae_kcal": float(errors.abs().mean()),
                             "maximum_error_kcal": float(errors.abs().max())}
    report["peak_cuda_bytes"] = torch.cuda.max_memory_allocated()
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({"summary": report.get("summary"), "failures": report["failures"]}), flush=True)


if __name__ == "__main__":
    main()
