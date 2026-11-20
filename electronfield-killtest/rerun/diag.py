"""Diagnostics for the electron-field kill test. Imports the user's prototype unchanged.

Usage: python3 diag.py <path-to-electronfield-killtest> <experiment> [args]
"""
import sys, json, time, math
import numpy as np
import torch

REPO = sys.argv[1]
sys.path.insert(0, REPO)
import prototype as P  # noqa: E402

torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)
DX = 0.45


def setup(seed, n_train=None):
    data = P.generate_dataset(seed=seed)
    x, t, w, _ = P.grid_setup(35, DX)
    sets = {s: P.make_tensors(data, s) for s in ("train", "test_id", "test_ood")}
    if n_train is not None:
        sets["train"] = {k: v[:n_train] for k, v in sets["train"].items()}
    return x, t, w, sets


def evaluate(model, sets):
    out = {}
    for split, b in sets.items():
        with torch.enable_grad():
            ee, nn, conv = P.scf(model, b["v"], steps=250, mix=0.15)
            n_in = b["n"].clone().requires_grad_(True)
            vc = P.centered_potential(model.corr_potential(n_in, create_graph=False), n_in, DX)
            vref = P.centered_potential(b["vc"], n_in, DX)
            err = (ee - b["e"]).detach()
            dm = DX / 2 * (nn.detach() - b["n"]).abs().sum(-1)
            vr = (DX / 2 * (b["n"] * (vc.detach() - vref).square()).sum(-1)).sqrt()
        out[split] = dict(e_mae=float(err.abs().mean()), dens=float(dm.mean()),
                          v_rmse=float(vr.mean()), unconv=int((conv.detach() > 1e-4).sum()))
    return out


def field_losses(model, tr):
    n_ref = tr["n"].clone().requires_grad_(True)
    vc = P.centered_potential(model.corr_potential(n_ref, create_graph=True), n_ref, DX)
    vref = P.centered_potential(tr["vc"], n_ref, DX)
    pot = (DX / 2 * (n_ref * (vc - vref).square()).sum(-1)).mean() / 0.12 ** 2
    vh = DX * (n_ref @ model.w)
    heff = model.t[None] + torch.diag_embed(tr["v"] + 0.5 * vh + vc)
    _, vec = torch.linalg.eigh(heff)
    resp = 2 / DX * vec[:, :, 0].square()
    dens = (DX / 2 * (resp - n_ref).square().sum(-1)).mean() / 0.12 ** 2
    return pot, dens


def train(arm, seed, steps, lr=0.003, pw=0.2, dw=0.2, n_train=None, cosine=False,
          checkpoints=(), init=None, quiet=True):
    torch.manual_seed(seed); np.random.seed(seed)
    x, t, w, sets = setup(seed, n_train)
    model = P.XCFunctional(x, w, t, DX, 24)
    if init is not None:
        model.load_state_dict(init)
    init_state = {k: v.clone() for k, v in model.state_dict().items()}
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=lr * 0.02) if cosine else None
    tr = sets["train"]
    log = {}
    for ep in range(steps):
        opt.zero_grad()
        loss = torch.zeros(())
        if arm != "potential_only":
            e_scf, _, _ = P.scf(model, tr["v"], steps=24)
            e_loss = (((e_scf - tr["e"]) / 0.04) ** 2).mean()
            loss = loss + e_loss
        if arm in ("fields", "potential_only"):
            pot, dens = field_losses(model, tr)
            loss = loss + (pot if arm == "potential_only" else pw * pot + dw * dens)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        if sched: sched.step()
        if (ep + 1) in checkpoints or ep == steps - 1:
            log[ep + 1] = evaluate(model, sets)
            if not quiet:
                print(arm, seed, ep + 1, json.dumps(log[ep + 1]), flush=True)
    return log, init_state


def baseline(seed):
    """Zero-correlation functional (exact exchange only) and untrained net."""
    x, t, w, sets = setup(seed)
    torch.manual_seed(seed)
    m = P.XCFunctional(x, w, t, DX, 24)
    untrained = evaluate(m, sets)
    for p in m.parameters():
        torch.nn.init.zeros_(p)
    zero = evaluate(m, sets)
    data = P.generate_dataset(seed=seed)
    ec = {s: float(np.mean([abs(r["e_c"]) for r in data if r["split"] == s])) for s in sets}
    return dict(zero=zero, untrained=untrained, mean_abs_ec=ec)


if __name__ == "__main__":
    exp = sys.argv[2]
    if exp == "baseline":
        for seed in range(20261116, 20261121):
            print(seed, json.dumps(baseline(seed)), flush=True)
    elif exp == "repro":
        t0 = time.time()
        for arm in ("energy_only", "fields"):
            log, _ = train(arm, 20261116, 130)
            print(arm, json.dumps(log), round(time.time() - t0, 1), flush=True)
    elif exp == "run":
        # python3 diag.py REPO run <arm> <seed> <steps> <cosine 0/1> <pw> <dw> <n_train or 0> <out>
        arm, seed, steps, cosine = sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), bool(int(sys.argv[6]))
        pw, dw, ntr, out = float(sys.argv[7]), float(sys.argv[8]), int(sys.argv[9]), sys.argv[10]
        cps = [c for c in (130, 500, 1000, 2000, 4000) if c <= steps]
        log, _ = train(arm, seed, steps, pw=pw, dw=dw, n_train=ntr or None, cosine=cosine, checkpoints=cps)
        json.dump(dict(arm=arm, seed=seed, steps=steps, cosine=cosine, pw=pw, dw=dw, n_train=ntr, log=log),
                  open(out, "w"))
        print("done", out, flush=True)
