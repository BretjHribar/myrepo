"""Symmetry-complete local-charge flow in a small physical state block.

The single-state flow can be trapped in a wrong spatial symmetry sector.
This changed ansatz transforms eight low variational states together, then
chooses the lowest generalized Rayleigh energy. These are physical sectors,
not stochastic training seeds. All numerical work is on CUDA.
"""
from .runtime import configure, torch_setup
ROOT = configure(gpu_only_compatible=True)

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import torch
from .gpu_connected_flow import rotate_all

KCAL = 627.509474


def h3_action_block(state, h, eri):
    result = (torch.einsum("il,aljk->aijk", h, state)
              +torch.einsum("jl,ailk->aijk", h, state)
              +torch.einsum("kl,aijl->aijk", h, state))
    return (result+torch.einsum("iljm,almk->aijk", eri, state)
            +torch.einsum("ilkm,aljm->aijk", eri, state)
            +torch.einsum("jlkm,ailm->aijk", eri, state))


def rotate_block(state, matrix):
    for axis in range(1, state.ndim):
        state = torch.tensordot(matrix, state, dims=([1], [axis])).movedim(0, axis)
    return state


class SectorFlow(torch.nn.Module):
    def __init__(self, data, layers=3):
        super().__init__()
        seed, regions, landmarks = data["sector_block"], data["regions"], data["landmarks"]
        self.register_buffer("seed", seed)
        modes = [region for region in regions]
        weights = [seed.new_ones(()) for _ in modes]
        atomic = regions.reshape(len(regions)//2, 2, *regions.shape[-2:]).sum(1)
        for a in range(len(atomic)):
            for b in range(a):
                modes.append(atomic[a]-atomic[b])
                # Bounded contrast gates vanish at infinite fragment separation.
                distance2 = (landmarks[a]-landmarks[b]).square().sum()
                weights.append((1+distance2/4).pow(-1.5))
        values, vectors = torch.linalg.eigh(torch.stack(modes))
        self.register_buffer("rotation", vectors)
        self.register_buffer("orbital_spectrum", values)
        self.register_buffer("mode_weight", torch.stack(weights))
        diagonals = []
        for spectrum in values:
            r1, r2, r3 = spectrum[:, None, None], spectrum[None, :, None], spectrum[None, None, :]
            diagonals.append(r1*r2+r1*r3+r2*r3)
        self.register_buffer("pair_diagonal", torch.stack(diagonals))
        self.raw = torch.nn.Parameter(seed.new_zeros(layers, len(modes)))

    def forward(self):
        states = self.seed
        for layer in 3*torch.tanh(self.raw)*self.mode_weight:
            for theta, rotation, diagonal in zip(layer, self.rotation, self.pair_diagonal):
                states = rotate_block(states, rotation.T)
                states = states*torch.exp(-theta*diagonal)
                states = rotate_block(states, rotation)
            states = states/states.square().sum((1, 2, 3), keepdim=True).sqrt()
        return states


def variational_energy(states, h, eri):
    acted = h3_action_block(states, h, eri)
    metric = torch.einsum("aijk,bijk->ab", states, states)
    hmatrix = torch.einsum("aijk,bijk->ab", states, acted)
    chol = torch.linalg.cholesky((metric+metric.T)/2+torch.eye(len(metric), device=metric.device)*1e-12)
    inverse = torch.linalg.solve_triangular(chol, torch.eye(len(metric), device=metric.device), upper=False)
    projected = inverse @ ((hmatrix+hmatrix.T)/2) @ inverse.T
    values, vectors = torch.linalg.eigh((projected+projected.T)/2)
    coefficients = inverse.T @ vectors[:, 0]
    state = torch.einsum("a,aijk->ijk", coefficients, states)
    return values[0], state


def optimize(path, args, device, model_class=SectorFlow):
    data = torch.load(path, map_location=device, weights_only=True)
    start = time.perf_counter()
    model = model_class(data)
    optimizer = torch.optim.Adam(model.parameters(), lr=.025)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.steps, eta_min=.0003)
    best_energy, best_state, history = None, None, []
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        electronic, state = variational_energy(model(), data["h"], data["eri"])
        electronic.backward()
        if not bool(torch.isfinite(electronic)) or not bool(torch.isfinite(model.raw.grad).all()):
            raise RuntimeError("Nonfinite physical state-block optimization")
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        energy = float(electronic.detach()+data["nuclear"])
        if best_energy is None or energy < best_energy:
            best_energy, best_state = energy, state.detach().clone()
        optimizer.step()
        scheduler.step()
        if step == 0 or (step+1) % 100 == 0 or step+1 == args.steps:
            row = {"step": step+1, "error_kcal": (energy-float(data["reference"]))*KCAL,
                   "gradient_norm": float(model.raw.grad.norm()), "seconds": time.perf_counter()-start}
            history.append(row)
            print(json.dumps({"id": data["metadata"]["id"], **row}), flush=True)
    error = (best_energy-float(data["reference"]))*KCAL
    # Report the physical Rayleigh quotient of the stored state, independently
    # of the tiny regularization used in the generalized eigenproblem.
    actual = (best_state*h3_action_block(best_state[None], data["h"], data["eri"])[0]).sum()/best_state.square().sum()+data["nuclear"]
    error = float((actual-data["reference"])*KCAL)
    antisymmetry = float((best_state+best_state.transpose(0, 1)).abs().max())
    spin_identity = float((best_state-best_state.transpose(1, 2)+best_state.permute(2, 0, 1)).abs().max())
    if antisymmetry > 1e-8 or spin_identity > 1e-8:
        raise RuntimeError("The trial state violated fermionic antisymmetry or the doublet spin identity")
    torch.save({"architecture": model_class.__name__, "model": model.state_dict(),
                "best_state": best_state.cpu(), "metadata": data["metadata"]},
               args.output.with_name(args.output.stem+"_"+data["metadata"]["id"]+".pt"))
    return {"id": data["metadata"]["id"], "nelec": data["metadata"]["nelec"],
            "energy_hartree": float(actual), "reference_hartree": float(data["reference"]), "error_kcal": error,
            "antisymmetry_residual": antisymmetry, "doublet_spin_residual": spin_identity,
            "variational_bound_satisfied": error > -1e-3, "physical_state_block": len(model.seed),
            "parameters": model.raw.numel(), "history": history, "seconds": time.perf_counter()-start}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--output", type=Path, default=Path("research/sector_flow_gpu_20261116.json"))
    args = parser.parse_args()
    device = torch_setup()
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "device": device,
              "scope": "Changed multi-sector physical correlation flow; three-electron development evidence only",
              "training_targets": "Variational Hamiltonian expectation, no reference-energy fit",
              "cases": [], "failures": [], "steps": args.steps}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    seed_dir = ROOT / "data/gpu_odd_operators"
    for source in sorted((ROOT / "data/pilot_3d_ccpvdz").glob("H3_neutral_*.npz")):
        path = seed_dir / f"{source.stem}.flow_seed.pt"
        try:
            previous = torch.load(path, map_location="cpu", weights_only=True) if path.exists() else {}
            if "sector_block" not in previous:
                from .gpu_odd_geminal import calculate
                calculate(source, seed_dir, device)
            row = optimize(path, args, device)
            report["cases"].append(row)
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
