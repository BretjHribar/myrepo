"""One bounded CUDA experiment for our shared pair probability state."""
from .runtime import configure, torch_setup
ROOT = configure(gpu_only_compatible=True)

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import torch
from .joint_transport import JointTransport

KCAL = 627.509474


def load_batch(data, device):
    manifest = json.loads((data / "manifest.json").read_text())
    cases = [torch.load(data / f"{row['id']}.pt", map_location=device, weights_only=True)
             for row in sorted(manifest["cases"], key=lambda row: row["id"])]
    if manifest["failures"]:
        raise RuntimeError("GPU preprocessing has unresolved failures; do not silently drop them")
    nodes = max(case["mass"].shape[-1] for case in cases)
    batch = {}
    for name in ("node_features", "mass", "edges", "interaction", "distance"):
        values = []
        for case in cases:
            value = case[name]
            pad = nodes-value.shape[-2 if name == "node_features" else -1]
            if name == "node_features":
                value = torch.nn.functional.pad(value, (0, 0, 0, pad))
            elif name == "mass":
                value = torch.nn.functional.pad(value, (0, pad))
            else:
                value = torch.nn.functional.pad(value, (0, pad, 0, pad))
            if name == "distance":
                value = value[None].expand(len(case["energy"]), -1, -1)
            values.append(value)
        batch["features" if name == "node_features" else name] = torch.cat(values)
    for name in ("energy_hf_input", "energy_pbe", "energy"):
        batch[name] = torch.cat([case[name] for case in cases])
    states = len(cases[0]["energy"])
    return manifest, cases, batch, states


def prediction(model, batch):
    correction, diagnostics = model(*[batch[key] for key in ("features", "mass", "edges", "interaction", "distance")])
    return batch["energy_hf_input"]+correction, diagnostics


@torch.no_grad()
def measure(model, cases, batch, states, include_test=False):
    predicted, diagnostic = prediction(model, batch)
    errors = (predicted-batch["energy"]).reshape(len(cases), states)[:, 0]*KCAL
    base = (batch["energy_pbe"]-batch["energy"]).reshape(len(cases), states)[:, 0]*KCAL
    result = {}
    for split in (("train", "val", "test") if include_test else ("train", "val")):
        indices = [i for i, case in enumerate(cases) if case["metadata"]["split"] == split]
        if not indices:
            continue
        index = torch.tensor(indices, device=errors.device)
        result[split] = {"cases": len(indices), "energy_mae_kcal": float(errors[index].abs().mean()),
                         "pbe_input_mae_kcal": float(base[index].abs().mean()),
                         "by_case": [{"id": cases[i]["metadata"]["id"], "error_kcal": float(errors[i])} for i in indices]}
    result["state_diagnostics"] = {"maximum_marginal_residual": float(diagnostic["marginal_residual"].max()),
                                   "minimum_kinetic_cost_hartree": float(diagnostic["kinetic_cost"].min()),
                                   "positive_candidates": int(diagnostic["candidate_positive"].sum()),
                                   "kappa_min": float(diagnostic["kappa_min"].min()),
                                   "kappa_max": float(diagnostic["kappa_max"].max())}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/gpu_pair_state")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=500)
    args = parser.parse_args()
    device = torch_setup()
    manifest, cases, batch, states = load_batch(args.data, device)
    model = JointTransport().to(device)
    train_cases = [i for i, case in enumerate(cases) if case["metadata"]["split"] == "train"]
    training = torch.tensor(train_cases, device=device)
    groups = [[i for i in train_cases if cases[i]["metadata"]["family"] == family]
              for family in sorted({cases[i]["metadata"]["family"] for i in train_cases})]
    optimizer = torch.optim.Adam(model.parameters(), lr=.008)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.steps, eta_min=.0002)
    initial = measure(model, cases, batch, states)
    history = []
    start = time.perf_counter()
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        predicted, diagnostic = prediction(model, batch)
        if float(diagnostic["marginal_residual"].max()) > 1e-9:
            raise RuntimeError("Pair-state density marginals are not satisfied")
        errors = (predicted-batch["energy"]).reshape(len(cases), states)
        loss = (errors[training].mean(-1)/.03).square().mean()
        for positions in groups:
            if len(positions) > 1:
                curve = errors[positions].mean(-1)
                loss = loss+((curve[1:]-curve[0])/.03).square().mean()
        field_errors = (errors[training, 1:]-errors[training, :1])/.002
        loss = loss+.1*field_errors.square().mean()
        loss.backward()
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("Nonfinite new-state loss")
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
        optimizer.step()
        scheduler.step()
        if step == 0 or (step+1) % 100 == 0 or step+1 == args.steps:
            current = measure(model, cases, batch, states)
            row = {"step": step+1, "loss": float(loss.detach()), "metrics": current,
                   "seconds": time.perf_counter()-start}
            history.append(row)
            print(json.dumps({"step": row["step"], "loss": row["loss"],
                              "energy_mae": {key: value["energy_mae_kcal"] for key, value in current.items() if key != "state_diagnostics"},
                              "marginal_residual": current["state_diagnostics"]["maximum_marginal_residual"],
                              "seconds": round(row["seconds"], 2)}), flush=True)
    final = measure(model, cases, batch, states, include_test=True)
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "device": device,
              "projection": "Finite Sinkhorn followed by exact nonnegative deficit rounding",
              "scope": "Original two-electron constrained-pair mechanism experiment. Frozen inputs, finite-basis labels; not broad chemistry or challenge performance.",
              "accuracy_priority": "Sub-kcal transfer first; speed optimization deferred by user",
              "initial": initial, "history": history, "final": final,
              "outside_scope": manifest["outside_scope"], "runtime_seconds": time.perf_counter()-start,
              "peak_cuda_bytes": torch.cuda.max_memory_allocated(), "steps": args.steps}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    torch.save({"architecture": "joint_transport", "width": model.width,
                "iterations": model.iterations, "model": model.state_dict()}, args.output.with_suffix(".pt"))
    print(json.dumps({"final": {key: value if key == "state_diagnostics" else {k: v for k, v in value.items() if k != "by_case"}
                                for key, value in final.items()}}), flush=True)


if __name__ == "__main__":
    main()
