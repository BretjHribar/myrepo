"""Exact 1D two-electron benchmark for energy-only vs field-supervised XC models.

Research *prototype*, not evidence of chemical accuracy/competition performance.
Only dependencies: NumPy, SciPy, PyTorch; CPU-only is sufficient.
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh
import torch
from torch import nn

HARTREE_TO_KCAL = 627.509474


def grid_setup(n: int = 35, dx: float = 0.45, interaction_soft: float = 1.0):
    x = (np.arange(n) - (n - 1) / 2) * dx
    t = sparse.diags((-np.ones(n - 1), 2 * np.ones(n), -np.ones(n - 1)), (-1, 0, 1), format="csr") / (2 * dx**2)
    w = 1 / np.sqrt((x[:, None] - x[None, :]) ** 2 + interaction_soft**2)
    two_body_kinetic = sparse.kron(t, sparse.eye(n), format="csr") + sparse.kron(sparse.eye(n), t, format="csr")
    return x, t.toarray(), w, two_body_kinetic


def external_potential(x, z1, z2, r, nuclear_soft=0.8):
    return -z1 / np.sqrt((x + r / 2) ** 2 + nuclear_soft**2) - z2 / np.sqrt((x - r / 2) ** 2 + nuclear_soft**2)


def exact_system(x, t, w, hkin, dx, z1, z2, r):
    """Exact diagonalization of opposite-spin two-electron Hamiltonian.

    Discrete wavefunction normalized via sum |psi_ij|² = 1 (grid quadrature
    factors absorbed in the coefficients). Density is 2 sum_j |psi_ij|² / dx.
    """
    v = external_potential(x, z1, z2, r)
    h = hkin + sparse.diags((v[:, None] + v[None, :] + w).ravel(), format="csr")
    evals, evecs = eigsh(h, k=1, which="SA", tol=1e-10)
    psi = evecs[:, 0].reshape(len(x), len(x))
    dens = (2 / dx) * (psi**2).sum(axis=1)
    phi = np.sqrt(np.maximum(dens * dx / 2, 1e-20))
    vh = dx * (w @ dens)
    # Exact two-electron spin-singlet exchange: E_x = -E_H/2.
    # Inverted exact KS correlation potential is determined up to a constant.
    vc = -(t @ phi) / np.maximum(phi, 1e-12) - v - 0.5 * vh
    vc -= dx / 2 * np.dot(dens, vc)  # weighted-mean gauge
    ts = 2 * np.dot(phi, t @ phi)
    eh = dx**2 / 2 * (dens @ w @ dens)
    eext = dx * np.dot(v, dens)
    ec = float(evals[0] - ts - eext - eh / 2)
    return dict(z1=z1, z2=z2, r=r, v=v.tolist(), n=dens.tolist(),
                e=float(evals[0]), e_c=ec, vc=vc.tolist())


def generate_dataset(seed=20261116, ngrid=35, dx=0.45):
    rng = np.random.default_rng(seed)
    x, t, w, hkin = grid_setup(ngrid, dx)
    cases = []
    # Two-electron neutral and slightly asymmetric diatomic model potentials.
    # The long-bond split is a meaningful out-of-distribution stress test.
    for split, k, r_range in [("train", 28, (0.9, 3.1)),
                               ("test_id", 8, (0.9, 3.1)),
                               ("test_ood", 8, (3.2, 4.6))]:
        for i in range(k):
            r = float(rng.uniform(*r_range))
            z1 = float(rng.uniform(0.65, 1.35))
            z2 = float(2.0 - z1)
            row = exact_system(x, t, w, hkin, dx, z1, z2, r)
            row["id"] = f"{split}_{i:02d}"
            row["split"] = split
            cases.append(row)
    return cases


class XCFunctional(nn.Module):
    """E_xc = -E_H/2 + integral n(x) f_theta(local/nonlocal descriptors) dx."""
    def __init__(self, x, w, t, dx, width=24):
        super().__init__()
        self.dx = dx
        self.register_buffer("w", torch.tensor(w, dtype=torch.float64))
        self.register_buffer("t", torch.tensor(t, dtype=torch.float64))
        distances = x[:, None] - x[None, :]
        g1 = np.exp(-0.5 * (distances / 0.7) ** 2)
        g2 = np.exp(-0.5 * (distances / 1.6) ** 2)
        g1 /= g1.sum(axis=1, keepdims=True)
        g2 /= g2.sum(axis=1, keepdims=True)
        self.register_buffer("g1", torch.tensor(g1, dtype=torch.float64))
        self.register_buffer("g2", torch.tensor(g2, dtype=torch.float64))
        self.net = nn.Sequential(nn.Linear(5, width), nn.Tanh(),
                                 nn.Linear(width, width), nn.Tanh(), nn.Linear(width, 1))

    def corr_energy(self, n):
        smooth1 = n @ self.g1.T
        smooth2 = n @ self.g2.T
        vh = self.dx * (n @ self.w)
        # Shift/scale descriptors to keep fitting well conditioned.
        feats = torch.stack((n, smooth1, smooth2, vh / 2,
                             (smooth1 - smooth2)), dim=-1)
        f = self.net(feats).squeeze(-1)
        return self.dx * (n * f).sum(dim=-1)

    def corr_potential(self, n, create_graph=True):
        if not n.requires_grad:
            n = n.detach().requires_grad_(True)
        e = self.corr_energy(n)
        # E=sum_i dx n_i epsilon_i, hence derivative /dx is potential.
        dc = torch.autograd.grad(e.sum(), n, create_graph=create_graph)[0] / self.dx
        return dc


def make_tensors(data, split, dtype=torch.float64):
    xs = [r for r in data if r["split"] == split]
    return {k: torch.tensor([r[k] for r in xs], dtype=dtype)
            for k in ("v", "n", "vc", "e")}


def noninteracting_density(t, v, dx):
    e, eigv = torch.linalg.eigh(t[None] + torch.diag_embed(v))
    return 2 * eigv[:, :, 0].square() / dx


def scf(model, v, steps=35, mix=0.35, differentiable=False):
    """KS SCF with a fixed mixing schedule; differentiable=True unrolls SCF.

    If differentiable=False, SCF state is detached each step. This makes
    training energy gradients use the converged-state variational identity.
    """
    n = noninteracting_density(model.t, v, model.dx)
    for _ in range(steps):
        if not differentiable:
            n = n.detach()
        n = n.requires_grad_(True)
        vc = model.corr_potential(n, create_graph=differentiable)
        vh = model.dx * (n @ model.w)
        h = model.t[None] + torch.diag_embed(v + 0.5 * vh + vc)
        evals, vectors = torch.linalg.eigh(h)
        n_new = (2 / model.dx) * vectors[:, :, 0].square()
        n = (1 - mix) * n + mix * n_new
    # Recompute potential at final n for consistency and energy functional.
    if not differentiable:
        n = n.detach()
    n = n.requires_grad_(True)
    vc = model.corr_potential(n, create_graph=differentiable)
    vh = model.dx * (n @ model.w)
    h = model.t[None] + torch.diag_embed(v + 0.5 * vh + vc)
    _, eigv = torch.linalg.eigh(h)
    orbitals = eigv[:, :, 0]
    n_orb = 2 / model.dx * orbitals.square()
    # Evaluate variational KS total energy from current occupied orbital.
    ts = 2 * torch.einsum("bi,ij,bj->b", orbitals, model.t, orbitals)
    eh = 0.5 * model.dx**2 * torch.einsum("bi,ij,bj->b", n_orb, model.w, n_orb)
    ext = model.dx * (v * n_orb).sum(dim=-1)
    ec = model.corr_energy(n_orb)
    energy = ts + ext + 0.5 * eh + ec
    convergence = (model.dx * (n_orb - n).abs().sum(dim=-1) / 2)
    return energy, n_orb, convergence


def centered_potential(p, dens, dx):
    return p - (dx / 2 * (p * dens).sum(dim=-1, keepdim=True))


def run(seed=20261116, steps=130, width=24, lr=0.003,
        density_weight=0.2, potential_weight=0.2,
        ngrid=35, dx=0.45, out="results.json", verbose=True):
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    np.random.seed(seed); torch.manual_seed(seed)
    dataset = generate_dataset(seed=seed, ngrid=ngrid, dx=dx)
    x, t, w, _ = grid_setup(ngrid, dx)
    train = make_tensors(dataset, "train")
    sets = {s: make_tensors(dataset,s) for s in ("train", "test_id", "test_ood")}
    initial_model = XCFunctional(x,w,t,dx,width)
    initial_state = {k: v.clone() for k,v in initial_model.state_dict().items()}
    summary = {"setup": {"seed":seed,"epochs":steps,"width":width,"lr":lr,
                          "density_weight":density_weight,"potential_weight":potential_weight,
                          "ngrid":ngrid,"dx":dx,"train_n":28,"test_id_n":8,"test_ood_n":8,
                          "method":"Exact diagonalized 1D soft-Coulomb two-electron singlets"},
               "arms":{}}
    for arm in ("energy_only", "energy_density_potential"):
        model = XCFunctional(x,w,t,dx,width)
        model.load_state_dict(initial_state)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        history = []
        start = time.monotonic()
        for epoch in range(steps):
            opt.zero_grad()
            # Full self-consistent energy for both arms. Energy gradients
            # assume stationarity; the finite-step convergence is recorded.
            e_scf, n_scf, _ = scf(model,train["v"],steps=24,differentiable=False)
            e_loss = (((e_scf-train["e"])/0.04)**2).mean()
            loss = e_loss
            density_loss = torch.zeros(())
            potential_loss = torch.zeros(())
            if arm != "energy_only":
                n_ref = train["n"].clone().requires_grad_(True)
                vc = model.corr_potential(n_ref,create_graph=True)
                # Gauge-free potential matching; density weights suppress tails.
                vc = centered_potential(vc,n_ref,dx)
                vref = centered_potential(train["vc"],n_ref,dx)
                potential_loss = (dx/2 * (n_ref*(vc-vref).square()).sum(-1)).mean() / 0.12**2
                # Density self-consistency residual at the exact density.
                # Differentiation goes through the KS diagonalization and vc.
                vh_ref = dx * (n_ref @ model.w)
                heff = model.t[None] + torch.diag_embed(train["v"] + .5*vh_ref + vc)
                _, ksvec = torch.linalg.eigh(heff)
                density_response = 2 / dx * ksvec[:,:,0].square()
                density_loss = (dx/2 * ((density_response-n_ref).square()).sum(-1)).mean() / 0.12**2
                loss = loss + potential_weight*potential_loss + density_weight*density_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            if epoch % 10 == 0 or epoch == steps - 1:
                history.append({"epoch":epoch,"loss":round(float(loss.detach()),6),
                                "e_loss":round(float(e_loss.detach()),6),
                                "d_loss":round(float(density_loss.detach()),6),
                                "v_loss":round(float(potential_loss.detach()),6)})
                if verbose and epoch % 40 == 0:
                    print(f"{arm} epoch {epoch:3d}: objective {float(loss.detach()):.5f} E {float(e_loss.detach()):.5f}",flush=True)
        metrics = {}
        for split, batch in sets.items():
            # Deliberately do not call no_grad: KS potentials are functional derivatives.
            with torch.enable_grad():
                ee, nn, convergence = scf(model,batch["v"],steps=250,mix=0.15,differentiable=False)
                nn_ref = batch["n"]
                n_in = nn_ref.clone().requires_grad_(True)
                vc_pred = model.corr_potential(n_in,create_graph=False)
                vc_pred = centered_potential(vc_pred,n_in,dx)
                vc_ref = centered_potential(batch["vc"],n_in,dx)
                e_error = (ee-batch["e"]).detach()
                dm = (dx/2 * (nn.detach() - nn_ref).abs().sum(-1))
                v_rmse = (dx/2 * (nn_ref * (vc_pred.detach() - vc_ref).square()).sum(-1)).sqrt()
            metrics[split] = {"energy_mae_hartree":float(e_error.abs().mean()),
                              "energy_mae_kcal_analog":float(e_error.abs().mean()*HARTREE_TO_KCAL),
                              "energy_rmse_hartree":float(e_error.square().mean().sqrt()),
                              "density_l1_fraction":float(dm.mean()),
                              "potential_weighted_rmse_hartree":float(v_rmse.detach().mean()),
                              "scf_residual_l1_fraction":float(convergence.detach().mean()),
                              "scf_not_converged_1e-4":int((convergence.detach()>1e-4).sum()),
                              "energies_predicted":ee.detach().tolist(),
                              "energy_error_hartree_by_case":e_error.tolist()}
        summary["arms"][arm] = {"metrics":metrics,"history":history,
                                 "fit_runtime_sec":round(time.monotonic()-start,3),
                                 "n_params":sum(p.numel() for p in model.parameters())}
        if verbose:
            print(arm, json.dumps({s:{k:round(v,5) for k,v in m.items() if isinstance(v,float)} for s,m in metrics.items()}),flush=True)
    Path(out).write_text(json.dumps(summary,indent=2)+"\n")
    return summary

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",type=int,default=130)
    p.add_argument("--seed",type=int,default=20261116)
    p.add_argument("--width",type=int,default=24)
    p.add_argument("--lr",type=float,default=.003)
    p.add_argument("--potential-weight",type=float,default=.2)
    p.add_argument("--density-weight",type=float,default=.2)
    p.add_argument("--output",default="results.json")
    a=p.parse_args()
    run(seed=a.seed,steps=a.epochs,width=a.width,lr=a.lr,
        density_weight=a.density_weight,potential_weight=a.potential_weight,out=a.output)
