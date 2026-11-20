"""Connected three-electron density correlations, on CUDA.

For a one-particle region operator Q, the normal-ordered third moment is
sum_{p<q<r} Q(p)Q(q)Q(r). It is zero for N<3 without an empirical N switch.
Here its exact exponential is evaluated for three electrons. Local and
distance-damped two-region operators define a connected spatial hierarchy.
No correlated reference energy enters optimization.
"""
from .runtime import configure, torch_setup
ROOT = configure(gpu_only_compatible=True)

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import torch
from .gpu_sector_flow import optimize, rotate_block


class TripleFlow(torch.nn.Module):
    def __init__(self, data):
        super().__init__()
        self.register_buffer("seed", data["sector_block"])
        regions, landmarks = data["regions"], data["landmarks"]
        atomic = regions.reshape(len(regions)//2, 2, *regions.shape[-2:]).sum(1)
        modes = [region for region in regions]+[region for region in atomic]
        weights = [self.seed.new_ones(()) for _ in modes]
        for a in range(len(atomic)):
            for b in range(a):
                modes.append(atomic[a]+atomic[b])
                distance2 = (landmarks[a]-landmarks[b]).square().sum()
                weights.append((1+distance2/4).pow(-1.5))
        values, vectors = torch.linalg.eigh(torch.stack(modes))
        self.register_buffer("rotation", vectors)
        self.register_buffer("triple_diagonal", values[:, :, None, None]*values[:, None, :, None]*values[:, None, None, :])
        self.register_buffer("mode_weight", torch.stack(weights))
        self.raw = torch.nn.Parameter(self.seed.new_zeros(3, len(modes)))

    def forward(self):
        states = self.seed
        for layer in self.raw*self.mode_weight:
            for angle, rotation, diagonal in zip(layer, self.rotation, self.triple_diagonal):
                states = rotate_block(states, rotation.T)
                states = states*torch.exp(-angle*diagonal)
                states = rotate_block(states, rotation)
            states = states/states.square().sum((1, 2, 3), keepdim=True).sqrt()
        return states


def prepare(source, device):
    data = torch.load(source, map_location=device, weights_only=True)
    name = data["metadata"]["id"]
    output = source.with_name(name+".triple_seed.pt")
    if output.exists():
        return output
    checkpoint = torch.load(Path("research") / f"quadrupole_connected_flow_gpu_20261116_{name}.pt", map_location=device, weights_only=True)
    block = torch.stack((checkpoint["best_state"], data["sector_block"][0]))
    gram = torch.einsum("aijk,bijk->ab", block, block)
    values, vectors = torch.linalg.eigh(gram)
    active = values > values.max()*1e-10
    data["sector_block"] = torch.einsum("ar,aijk->rijk", vectors[:, active]/values[active].sqrt(), block)
    data["base_state_source"] = "Best rank-two correlated state plus retained input determinant"
    torch.save({key: value.cpu() if isinstance(value, torch.Tensor) else value for key, value in data.items()}, output)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--output", type=Path, default=Path("research/connected_triple_flow_gpu_20261116.json"))
    args = parser.parse_args()
    device = torch_setup()
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "device": device,
              "scope": "New connected third-moment correlation operator; three-electron development test only",
              "training_targets": "Variational physical energy only; references used for reporting",
              "steps": args.steps, "cases": [], "failures": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for source in sorted((ROOT / "data/gpu_odd_operators").glob("*.reference_seed.pt")):
        try:
            prepared = prepare(source, device)
            report["cases"].append(optimize(prepared, args, device, model_class=TripleFlow))
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
