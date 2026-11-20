"""Complete the local angular moment family through rank two, CUDA only.

The five Cartesian traceless components are an orthonormal real rank-two
tensor basis. Summing their pair products is a rotational scalar. The numerical
gate and physical energy checks are the same as for the dipole state; this
changes the operator content, not training seeds, loss weights or depth.
"""
from .runtime import configure, torch_setup
ROOT = configure(gpu_only_compatible=True)

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import numpy as np
import torch
from .gpu_angular_flow import AngularFlow
from .gpu_sector_flow import optimize


def prepare(source, device):
    data = torch.load(source, map_location=device, weights_only=True)
    name = data["metadata"]["id"]
    output = source.with_name(name+".quadrupole_seed.pt")
    if output.exists():
        return output
    checkpoint = torch.load(Path("research") / f"angular_connected_flow_gpu_20261116_{name}.pt", map_location=device, weights_only=True)
    block = torch.stack((checkpoint["best_state"], data["sector_block"][0]))
    gram = torch.einsum("aijk,bijk->ab", block, block)
    values, vectors = torch.linalg.eigh(gram)
    active = values > values.max()*1e-10
    data["sector_block"] = torch.einsum("ar,aijk->rijk", vectors[:, active]/values[active].sqrt(), block)
    with np.load(ROOT / "data/pilot_3d_ccpvdz" / f"{name}.npz") as saved:
        coords, weights, landmarks, ao, orth = [torch.from_numpy(saved[key]).to(device)
                                               for key in ("coords", "weights", "landmarks", "ao", "orth")]
    displacement = coords[:, None]-landmarks[None]
    x, y, z = displacement.unbind(-1)
    r2 = displacement.square().sum(-1)
    partition = torch.softmax(-r2/2, -1)
    components = torch.stack(((x.square()-y.square())/2**.5,
                               (2*z.square()-x.square()-y.square())/6**.5,
                               2**.5*x*y, 2**.5*x*z, 2**.5*y*z), -1)
    fields = partition[..., None]*components/(1+r2[..., None])
    operators = []
    for atom in range(len(landmarks)):
        matrix = ao[0].T[None] @ ((weights[:, None]*fields[:, atom]).T[:, :, None]*ao[0][None])
        matrix = orth.T @ matrix @ orth
        matrix = (matrix+matrix.transpose(-1, -2))/2
        bound = torch.linalg.eigvalsh((matrix @ matrix).sum(0)).max().clamp_min(1e-12)
        operators.append(matrix/bound.sqrt())
    # AngularFlow's action sums over an arbitrary number of tensor components.
    data["dipole_operators"] = torch.stack(operators)
    data["angular_rank"] = 2
    data["base_state_source"] = "Best dipole-correlated state plus retained input determinant"
    torch.save({key: value.cpu() if isinstance(value, torch.Tensor) else value for key, value in data.items()}, output)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--output", type=Path, default=Path("research/quadrupole_connected_flow_gpu_20261116.json"))
    args = parser.parse_args()
    device = torch_setup()
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "device": device,
              "scope": "Connected angular moment family through rank two; three-electron development test, not broad chemical validation",
              "training_targets": "Hamiltonian expectation only; reference energies used for reporting",
              "steps": args.steps, "cases": [], "failures": []}
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
