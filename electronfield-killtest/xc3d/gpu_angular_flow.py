"""Rotation-scalar dipole correlation on top of the reference-complete state.

The new operator is sum_{p<q} sum_d Q_d(p)Q_d(q). It couples angular
fluctuations without singling out a laboratory axis and preserves all particle
permutations and spin. Its eighth-order exponential action has a conservative
per-gate truncation bound below 8e-11 after local operator normalization.
"""
from .runtime import configure, torch_setup
ROOT = configure(gpu_only_compatible=True)

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import numpy as np
import torch
from .gpu_sector_flow import optimize


def dipole_action(states, operator):
    first = torch.einsum("dil,bljk->dbijk", operator, states)
    pair12 = torch.einsum("djm,dbimk->bijk", operator, first)
    pair13 = torch.einsum("dkm,dbijm->bijk", operator, first)
    second = torch.einsum("djm,bimk->dbijk", operator, states)
    pair23 = torch.einsum("dkn,dbijn->bijk", operator, second)
    return pair12+pair13+pair23


class AngularFlow(torch.nn.Module):
    def __init__(self, data):
        super().__init__()
        self.register_buffer("seed", data["sector_block"])
        self.register_buffer("dipoles", data["dipole_operators"])
        self.raw = torch.nn.Parameter(self.seed.new_zeros(3, len(self.dipoles)))

    def forward(self):
        states = self.seed
        for layer in .1*torch.tanh(self.raw):
            for angle, operator in zip(layer, self.dipoles):
                term = states
                result = states
                for order in range(1, 9):
                    term = (-angle/order)*dipole_action(term, operator)
                    result = result+term
                states = result
            states = states/states.square().sum((1, 2, 3), keepdim=True).sqrt()
        return states


def prepare(source, device):
    data = torch.load(source, map_location=device, weights_only=True)
    name = data["metadata"]["id"]
    output = source.with_name(name+".angular_seed.pt")
    if output.exists():
        return output
    checkpoint = torch.load(Path("research") / f"reference_complete_flow_gpu_20261116_{name}.pt", map_location=device, weights_only=True)
    block = torch.stack((checkpoint["best_state"], data["sector_block"][0]))
    gram = torch.einsum("aijk,bijk->ab", block, block)
    values, vectors = torch.linalg.eigh(gram)
    active = values > values.max()*1e-10
    block = torch.einsum("ar,aijk->rijk", vectors[:, active]/values[active].sqrt(), block)
    with np.load(ROOT / "data/pilot_3d_ccpvdz" / f"{name}.npz") as saved:
        coords, weights, landmarks, ao, orth = [torch.from_numpy(saved[key]).to(device)
                                               for key in ("coords", "weights", "landmarks", "ao", "orth")]
    displacement = coords[:, None]-landmarks[None]
    radius = displacement.square().sum(-1).clamp_min(1e-20).sqrt()
    partition = torch.softmax(-radius.square()/2, -1)
    fields = partition[..., None]*displacement/(1+radius[..., None])
    operators = []
    for atom in range(len(landmarks)):
        matrix = ao[0].T[None] @ ((weights[:, None]*fields[:, atom]).T[:, :, None]*ao[0][None])
        matrix = orth.T @ matrix @ orth
        matrix = (matrix+matrix.transpose(-1, -2))/2
        bound = torch.linalg.eigvalsh((matrix @ matrix).sum(0)).max().clamp_min(1e-12)
        operators.append(matrix/bound.sqrt())
    data["sector_block"] = block
    data["dipole_operators"] = torch.stack(operators)
    data["base_state_source"] = "Best reference-complete physical state plus retained input determinant"
    torch.save({key: value.cpu() if isinstance(value, torch.Tensor) else value for key, value in data.items()}, output)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--output", type=Path, default=Path("research/angular_connected_flow_gpu_20261116.json"))
    args = parser.parse_args()
    device = torch_setup()
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "device": device,
              "scope": "New rotation-scalar angular operator, after reference-complete spin/charge/orbital flow; three-electron development evidence only",
              "training_targets": "Physical Hamiltonian expectation; reference energies used for reporting only",
              "gate_error_bound": "exp(0.3)*0.3^9/9! < 8e-11 per gate", "steps": args.steps,
              "cases": [], "failures": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for source in sorted((ROOT / "data/gpu_odd_operators").glob("*.reference_seed.pt")):
        try:
            prepared = prepare(source, device)
            report["cases"].append(optimize(prepared, args, device, model_class=AngularFlow))
        except Exception as exc:
            failure = {"id": source.stem, "exception": f"{type(exc).__name__}: {exc}"}
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
