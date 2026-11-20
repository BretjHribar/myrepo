"""Per-case long-bond check with a more careful SCF. Usage: python3 diag2.py REPO seed [seed...]"""
import sys, json
import numpy as np, torch
import diag as D
P = D.P
DX = D.DX


def fit(arm, seed, steps=1000, lr=0.003, w=5.0):
    torch.manual_seed(seed); np.random.seed(seed)
    x, t, wm, sets = D.setup(seed)
    model = P.XCFunctional(x, wm, t, DX, 24)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=lr * 0.02)
    tr = sets["train"]
    for ep in range(steps):
        opt.zero_grad()
        e_scf, _, _ = P.scf(model, tr["v"], steps=24)
        loss = (((e_scf - tr["e"]) / 0.04) ** 2).mean()
        if arm == "fields":
            pot, dens = D.field_losses(model, tr)
            loss = loss + w * pot + w * dens
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step(); sched.step()
    return model, sets


for seed in map(int, sys.argv[2:]):
    data = [r for r in P.generate_dataset(seed=seed) if r["split"] == "test_ood"]
    R = [round(r["r"], 2) for r in data]; Z = [round(r["z1"], 2) for r in data]
    print("seed", seed, "R", R, "z1", Z, flush=True)
    for arm in ("energy_only", "fields"):
        model, sets = fit(arm, seed)
        b = sets["test_ood"]
        for steps, mix in ((250, 0.15), (3000, 0.03)):
            with torch.enable_grad():
                ee, nn, conv = P.scf(model, b["v"], steps=steps, mix=mix)
            err = (ee - b["e"]).detach().numpy()
            # energy evaluated non-self-consistently on the exact density
            print(f"  {arm:12s} scf({steps},{mix}) err_mHa={np.round(1000*err,1).tolist()} "
                  f"resid={['%.0e' % c for c in conv.detach().numpy()]} MAE={np.abs(err).mean():.4f}", flush=True)
        with torch.enable_grad():
            n = b["n"].clone().requires_grad_(True)
            phi = torch.sqrt(n * DX / 2)
            ts = 2 * torch.einsum("bi,ij,bi->b", phi, model.t, phi) if False else 2 * torch.einsum("bi,ij,bj->b", phi, model.t, phi)
            eh = 0.5 * DX ** 2 * torch.einsum("bi,ij,bj->b", n, model.w, n)
            e_exactn = ts + DX * (b["v"] * n).sum(-1) + 0.5 * eh + model.corr_energy(n)
        err2 = (e_exactn - b["e"]).detach().numpy()
        print(f"  {arm:12s} on exact density err_mHa={np.round(1000*err2,1).tolist()} MAE={np.abs(err2).mean():.4f}", flush=True)
