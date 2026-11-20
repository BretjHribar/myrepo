"""Our noncommuting local-charge correlation flow, CUDA only.

Each gate is exp[-theta * sum_{p<q} Q(p)Q(q)], with Q a density-region
operator. The same definition applies at any electron count; it is identity
for one electron and factorizes for truly disjoint local fragments. This
experiment evaluates three-electron states, with an exact Hamiltonian
expectation and no reference-energy loss. Broad transfer remains unproven.
"""
from .runtime import configure, torch_setup
ROOT = configure(gpu_only_compatible=True)

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import torch

KCAL = 627.509474


def h3_action(state, h, eri):
    result = (torch.einsum("il,ljk->ijk", h, state)
              +torch.einsum("jl,ilk->ijk", h, state)
              +torch.einsum("kl,ijl->ijk", h, state))
    return (result+torch.einsum("iljm,lmk->ijk", eri, state)
            +torch.einsum("ilkm,ljm->ijk", eri, state)
            +torch.einsum("jlkm,ilm->ijk", eri, state))


def rotate_all(state, matrix):
    for axis in range(state.ndim):
        state = torch.tensordot(matrix, state, dims=([1], [axis])).movedim(0, axis)
    return state


class ConnectedFlow(torch.nn.Module):
    def __init__(self, seed, regions, layers=2):
        super().__init__()
        self.register_buffer("seed", seed)
        modes = [region for region in regions]
        atomic = regions.reshape(len(regions)//2, 2, *regions.shape[-2:]).sum(1)
        for a in range(len(atomic)):
            for b in range(a):
                modes.append(atomic[a]-atomic[b])
        operator = torch.stack(modes)
        values, vectors = torch.linalg.eigh((operator+operator.transpose(-1, -2))/2)
        self.register_buffer("rotation", vectors)
        pair_diagonals = []
        for spectrum in values:
            total = torch.zeros_like(seed)
            squares = torch.zeros_like(seed)
            for axis in range(seed.ndim):
                shape = [1]*seed.ndim
                shape[axis] = len(spectrum)
                entry = spectrum.reshape(shape)
                total = total+entry
                squares = squares+entry.square()
            pair_diagonals.append((total.square()-squares)/2)
        self.register_buffer("pair_diagonal", torch.stack(pair_diagonals))
        self.raw = torch.nn.Parameter(torch.zeros(layers, len(modes), device=seed.device, dtype=seed.dtype))

    def forward(self):
        state = self.seed
        angles = 3*torch.tanh(self.raw)
        for layer in angles:
            for theta, rotation, diagonal in zip(layer, self.rotation, self.pair_diagonal):
                state = rotate_all(state, rotation.T)
                state = state*torch.exp(-theta*diagonal)
                state = rotate_all(state, rotation)
            state = state/state.square().sum().sqrt()
        return state


def optimize(seed_path, device, steps, output_checkpoint):
    data = torch.load(seed_path, map_location=device, weights_only=True)
    start = time.perf_counter()
    model = ConnectedFlow(data["seed"], data["regions"]).to(device)
    h, eri = data["h"], data["eri"]
    optimizer = torch.optim.Adam(model.parameters(), lr=.04)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps, eta_min=.0005)
    history, best_energy, best_state = [], None, None
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        state = model()
        energy = (state*h3_action(state, h, eri)).sum()/state.square().sum()+data["nuclear"]
        energy.backward()
        if not bool(torch.isfinite(energy)) or not bool(torch.isfinite(model.raw.grad).all()):
            raise RuntimeError("Nonfinite connected-flow energy or gradient")
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        value = float(energy.detach())
        if best_energy is None or value < best_energy:
            best_energy = value
            best_state = state.detach().clone()
        optimizer.step()
        scheduler.step()
        if step == 0 or (step+1) % 100 == 0 or step+1 == steps:
            error = float((energy.detach()-data["reference"])*KCAL)
            row = {"step": step+1, "energy_hartree": value, "error_kcal": error,
                   "gradient_norm": float(model.raw.grad.norm()), "seconds": time.perf_counter()-start}
            history.append(row)
            print(json.dumps({"id": data["metadata"]["id"], **row}), flush=True)
    antisymmetry = float((best_state+best_state.transpose(0, 1)).abs().max())
    action = h3_action(best_state, h, eri)
    electronic = (best_state*action).sum()/best_state.square().sum()
    residual = (action-electronic*best_state).square().sum().sqrt()/best_state.square().sum().sqrt()
    error = (best_energy-float(data["reference"]))*KCAL
    torch.save({"architecture": "connected_local_charge_flow", "model": model.state_dict(),
                "best_state": best_state.cpu(), "metadata": data["metadata"]}, output_checkpoint)
    return {"id": data["metadata"]["id"], "nelec": data["metadata"]["nelec"],
            "initial_error_kcal": float((data["initial_energy"]-data["reference"])*KCAL),
            "error_kcal": error, "energy_hartree": best_energy, "reference_hartree": float(data["reference"]),
            "antisymmetry_residual": antisymmetry, "hamiltonian_residual_hartree": float(residual),
            "variational_bound_satisfied": error > -1e-3, "history": history,
            "parameters": model.raw.numel(), "seconds": time.perf_counter()-start}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--output", type=Path, default=Path("research/connected_flow_gpu_20261116.json"))
    args = parser.parse_args()
    device = torch_setup()
    seed_dir = ROOT / "data/gpu_odd_operators"
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "device": device,
              "scope": "New number-consistent correlation-flow hypothesis; first three-electron finite-basis test",
              "training_targets": "Physical variational energy only; references used solely for error reporting",
              "priority": "Out-of-sample accuracy and valid many-electron structure before speed",
              "cases": [], "failures": [], "steps": args.steps}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for path in sorted((ROOT / "data/pilot_3d_ccpvdz").glob("H3_neutral_*.npz")):
        seed_path = seed_dir / f"{path.stem}.flow_seed.pt"
        try:
            if not seed_path.exists():
                # Generate the state needed for this new experiment, reusing
                # the GPU ERI cache. This is initialization, not a repeated benchmark.
                from .gpu_odd_geminal import calculate
                calculate(path, seed_dir, device)
            row = optimize(seed_path, device, args.steps, args.output.with_name(path.stem+"_connected_flow.pt"))
            report["cases"].append(row)
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
