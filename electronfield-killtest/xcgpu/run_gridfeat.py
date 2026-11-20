"""Regime-resolved semilocal integrals from the stored grid data (GPU; no orbitals needed).

Each feature is an energy-like integral  sum_g w_g e(g) * u^i * b^j [* extra]  where
  e      = LDA exchange energy density of one spin channel, or the same-spin / opposite-spin parts of the
           PW92 correlation energy density (Stoll partition);
  u      = g s^2 / (1 + g s^2), s^2 = |grad rho_s|^2 / rho_s^(8/3), g = 0.004       (inhomogeneity, 0..1)
  b      = (tau - tau_W) / (tau + tau_unif)                                         (0 in one-orbital regions)
  extra  = 1, v = 1/(1 + r_s), or zeta^2 for the correlation channels.
A linear fit over these lets the mixing depend on the local regime while every term stays a
size-extensive functional of the density. No reference energies are read.
"""
import argparse
import json
import time
import traceback

from .common import dataset_dir, dataset_structures, file_stem, pin_one_core

pin_one_core()
import numpy as np
import cupy as cp
from gpu4pyscf.dft import libxc

GAMMA = 0.004
NX, NC = 4, 3            # polynomial orders (exclusive) in u and b for exchange / correlation
CF = 0.3 * (6 * np.pi ** 2) ** (2.0 / 3.0)       # spin-resolved Thomas-Fermi constant: tau_unif = CF * rho_s^(5/3)
CX = -0.75 * (3.0 / np.pi) ** (1.0 / 3.0) * 2 ** (1.0 / 3.0)   # e_x^LDA of one spin channel = CX * rho_s^(4/3)
TINY = 1e-14


def lda_c(ra, rb):
    """PW92 correlation energy density per volume for spin densities (ra, rb)."""
    fun = libxc.XCfun("lda_c_pw", "polarized")
    rho = cp.ascontiguousarray(cp.stack([ra, rb], axis=1))
    zk = fun.compute({"rho": rho}, do_exc=True, do_vxc=False)["zk"].ravel()
    return zk * (ra + rb)


def features(w, rho, sig, tau):
    """rho, tau: (n, 2); sig: (n, 2) same-spin |grad|^2."""
    out = {}
    u, b, ex, mask = [], [], [], []
    for s in (0, 1):
        r = cp.clip(rho[:, s], 0, None)
        m = r > 1e-11
        rs = cp.where(m, r, 1.0)
        s2 = sig[:, s] / rs ** (8.0 / 3.0)
        us = GAMMA * s2 / (1 + GAMMA * s2)
        tw = sig[:, s] / (8 * rs)
        tu = CF * rs ** (5.0 / 3.0)
        t = cp.maximum(tau[:, s], tw)
        bs = cp.clip((t - tw) / (t + tu), 0, 1)
        u.append(cp.where(m, us, 0.0)); b.append(cp.where(m, bs, 0.0)); mask.append(m)
        ex.append(cp.where(m, CX * rs ** (4.0 / 3.0), 0.0))
    # exchange channel
    for i in range(NX):
        for j in range(NX):
            out[f"x_{i}{j}"] = float(sum(cp.dot(w, ex[s] * u[s] ** i * b[s] ** j) for s in (0, 1)))
    # correlation channels
    ra, rb = cp.clip(rho[:, 0], 0, None), cp.clip(rho[:, 1], 0, None)
    zero = cp.zeros_like(ra)
    e_aa, e_bb, e_ab_tot = lda_c(ra, zero), lda_c(zero, rb), lda_c(ra, rb)
    e_aa = cp.where(mask[0], e_aa, 0.0); e_bb = cp.where(mask[1], e_bb, 0.0)
    e_os = cp.where(mask[0] | mask[1], e_ab_tot, 0.0) - e_aa - e_bb
    rt = ra + rb
    mt = rt > 1e-11
    rts = cp.where(mt, rt, 1.0)
    v = cp.where(mt, 1.0 / (1.0 + (3.0 / (4 * np.pi * rts)) ** (1.0 / 3.0)), 0.0)
    z2 = cp.where(mt, ((ra - rb) / rts) ** 2, 0.0)
    for i in range(NC):
        for j in range(NC):
            ss = 0.0
            ssv = 0.0
            for s, e in ((0, e_aa), (1, e_bb)):
                core = e * b[s] * u[s] ** i * b[s] ** j     # leading b removes one-orbital self-correlation
                ss += float(cp.dot(w, core)); ssv += float(cp.dot(w, core * v))
            out[f"css_{i}{j}"] = ss
            out[f"cssv_{i}{j}"] = ssv
            ua, ba = 0.5 * (u[0] + u[1]), 0.5 * (b[0] + b[1])
            core = e_os * ua ** i * ba ** j
            out[f"cos_{i}{j}"] = float(cp.dot(w, core))
            out[f"cosv_{i}{j}"] = float(cp.dot(w, core * v))
            out[f"cosz_{i}{j}"] = float(cp.dot(w, core * z2))
    out["n_elec"] = float(cp.dot(w, rt))
    return out


def run_one(grid_file):
    z = np.load(grid_file)
    w = cp.asarray(z["weights"])
    rho = cp.asarray(z["rho"], dtype=cp.float64)
    sig = cp.asarray(z["sigma"], dtype=cp.float64)
    tau = cp.asarray(z["tau"], dtype=cp.float64)
    if rho.ndim == 1:                      # restricted: one spin channel stored
        rho = cp.stack([rho, rho], axis=1); sig = cp.stack([sig, sig], axis=1); tau = cp.stack([tau, tau], axis=1)
    else:
        sig = cp.stack([sig[:, 0], sig[:, 2]], axis=1)
    return features(w, rho, sig, tau)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="gmtkn55")
    ap.add_argument("--scf-tag", required=True)
    ap.add_argument("--semi-tag", default="semi_v1")
    ap.add_argument("--tag", default="gridfeat_v1")
    args = ap.parse_args()
    semi = dataset_dir(args.dataset) / args.scf_tag / args.semi_tag
    out = semi / args.tag
    out.mkdir(parents=True, exist_ok=True)
    (out / "run_card.json").write_text(json.dumps({"gamma": GAMMA, "nx": NX, "nc": NC, "source": "stored float32 grid data"}))
    t0, new = time.time(), 0
    for row in dataset_structures(args.dataset):
        name = file_stem(row["id"])
        dest, src = out / (name + ".json"), semi / "grid" / (name + ".npz")
        if dest.exists() or not src.exists():
            continue
        try:
            res = {"id": row["id"], "features": run_one(src)}
        except Exception as exc:
            res = {"id": row["id"], "error": repr(exc)[:300], "traceback": traceback.format_exc()[-800:]}
        cp.get_default_memory_pool().free_all_blocks()
        tmp = dest.with_suffix(".tmp")
        tmp.write_text(json.dumps(res))
        tmp.replace(dest)
        new += 1
        if new % 100 == 0:
            print(json.dumps({"done": new, "elapsed_min": round((time.time() - t0) / 60, 1)}), flush=True)
    print(json.dumps({"finished": True, "new": new, "elapsed_min": round((time.time() - t0) / 60, 1)}), flush=True)


if __name__ == "__main__":
    main()
