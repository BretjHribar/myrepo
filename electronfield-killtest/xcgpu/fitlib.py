"""Assemble per-structure ingredients and fit / cross-validate linear-in-parameter functionals.

Model energy of a structure:  E = E_base + sum_j theta_j * F_j
  E_base = PBE total energy - PBE exchange-correlation energy + exact exchange of the PBE determinant
           + direct-RPA correlation energy from the PBE response spectrum        (theta = 0 is "EXX + RPA @ PBE")
  F_j    = size-extensive scalars: semilocal exchange/correlation components at the PBE density,
           shifted second-order terms, single-excitation terms, and integrals of a learned
           spectral function over the response eigenvalues (total and per-spin channels).
All numerics run on CUDA (float64). Reference energies enter only the fit targets.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

from .common import BULK, REPO, RESEARCH

sys.path.insert(0, str(REPO / "electronfield-killtest"))
from xc3d.gmtkn55 import GMTKN55Suite  # noqa: E402

DEV = "cuda"
KCAL = 627.5094740631
X_COMPONENTS = ("lda_x", "gga_x_pbe", "gga_x_b88", "gga_x_rpbe", "gga_x_pbe_sol", "gga_x_pw86",
                "mgga_x_tpss", "mgga_x_scan", "mgga_x_r2scan", "mgga_x_m06_l", "mgga_x_revtpss")
C_COMPONENTS = ("lda_c_pw", "gga_c_pbe", "gga_c_lyp", "gga_c_p86", "gga_c_pbe_sol",
                "mgga_c_tpss", "mgga_c_scan", "mgga_c_r2scan", "mgga_c_m06_l")
# knots in q = ln(1 + sigma) for the piecewise-linear spectral correction
Q_KNOTS = (0.0, 0.03, 0.1, 0.25, 0.5, 0.9, 1.5, 2.5, 4.0)


def suite():
    protocol = json.loads((REPO / "electronfield-killtest/research/gmtkn55_scoring_protocol_20261117.json").read_text())
    return GMTKN55Suite(RESEARCH, protocol)


def hat_basis(q, knots):
    """Piecewise-linear partition of unity on the knots; constant beyond the last knot."""
    k = torch.as_tensor(knots, dtype=q.dtype, device=q.device)
    out = []
    for i in range(len(knots)):
        if i == 0:
            b = torch.clamp((k[1] - q) / (k[1] - k[0]), 0, 1)
        elif i == len(knots) - 1:
            b = torch.clamp((q - k[i - 1]) / (k[i] - k[i - 1]), 0, 1)
        else:
            up = (q - k[i - 1]) / (k[i] - k[i - 1])
            down = (k[i + 1] - q) / (k[i + 1] - k[i])
            b = torch.clamp(torch.minimum(up, down), 0, 1)
        out.append(b)
    return out


POLY_ORDER = 5   # smooth spectral basis: [sigma^2/(1+sigma)] * x^k, x = sigma/(1+sigma), k = 0..POLY_ORDER-1
BAND_KNOTS = (0.1, 0.5, 2.0, 10.0)   # Hartree; frequency bands are hat functions in ln(omega) on these knots
BAND_ORDER = 3
TABLE_VERSION = 4


def spectral_integrals(sig, w, knots=Q_KNOTS):
    """-(1/2pi) sum_m w_m sum_n [sigma^2/(1+sigma)] b(sigma) for two bases b, plus the RPA value.

    Returns (hat-basis integrals over q = ln(1+sigma), polynomial-basis integrals over
    x = sigma/(1+sigma), RPA correlation energy). Every basis function is O(sigma^2) at small sigma
    and O(sigma) at large sigma, like the RPA integrand itself.
    """
    s = torch.clamp(sig, min=0)
    pref = s * s / (1 + s)
    q = torch.log1p(s)
    hat = [-(w[:, None] * pref * b).sum() / (2 * np.pi) for b in hat_basis(q, knots)]
    x = s / (1 + s)
    poly, xp = [], torch.ones_like(x)
    per_freq = []                       # (order, nfreq): sum over eigenvalues at each frequency
    for _ in range(POLY_ORDER):
        f = (pref * xp).sum(1)
        per_freq.append(f)
        poly.append(-(w * f).sum() / (2 * np.pi))
        xp = xp * x
    rpa = (w[:, None] * (q - s)).sum() / (2 * np.pi)
    return torch.stack(hat), torch.stack(poly), rpa, torch.stack(per_freq)


def band_integrals(per_freq, w, omega):
    """Split the first BAND_ORDER polynomial integrals into frequency bands (partition of unity in ln omega)."""
    bands = hat_basis(torch.log(omega), tuple(np.log(BAND_KNOTS)))
    return torch.stack([torch.stack([-(w * g * per_freq[k]).sum() / (2 * np.pi) for k in range(BAND_ORDER)]) for g in bands])


def structure_table(scf_tag="pbe_qzvp_g3", corr_tag="corr_v1", semi_tag="semi_v1", knots=Q_KNOTS, refresh=False):
    """Feature records for every structure with all three passes complete. Cached on the bulk drive."""
    root = BULK / "gmtkn55" / scf_tag
    cache = root / f"table_{corr_tag}_{semi_tag}.json"
    fresh = {"knots": list(knots), "version": TABLE_VERSION, "rows": {}}
    table = json.loads(cache.read_text()) if cache.exists() and not refresh else fresh
    if table.get("knots") != list(knots) or table.get("version") != TABLE_VERSION:
        table = fresh
    rows = table["rows"]
    problems = {}
    for scf_path in sorted((root / "results").glob("*.json")):
        name = scf_path.stem
        sid = name.replace("__", "/", 1)
        if sid in rows and rows[sid].get("grid") is not None:
            continue
        cpath, spath = root / corr_tag / "results" / (name + ".json"), root / semi_tag / "results" / (name + ".json")
        if not cpath.exists() or not spath.exists():
            continue
        scf, corr, semi = (json.loads(p.read_text()) for p in (scf_path, cpath, spath))
        bad = [k for k, r in (("scf", scf), ("corr", corr), ("semi", semi)) if r.get("error")] or \
              ([] if scf.get("converged") else ["scf_not_converged"])
        if bad:
            problems[sid] = bad
            continue
        z = np.load(root / corr_tag / "spectra" / (name + ".npz"))
        w = torch.as_tensor(z["weight"], device=DEV)
        st = torch.as_tensor(z["sigma_total"], device=DEV)
        nb = len(BAND_KNOTS)
        if st.shape[0] == 0:
            tot = torch.zeros(len(knots), dtype=torch.float64, device=DEV)
            spin = tot.clone()
            ptot = torch.zeros(POLY_ORDER, dtype=torch.float64, device=DEV)
            pspin = ptot.clone()
            btot = torch.zeros((nb, BAND_ORDER), dtype=torch.float64, device=DEV)
            bspin = btot.clone()
            rpa = torch.zeros((), dtype=torch.float64, device=DEV)
            sig_max = 0.0
        else:
            om = torch.as_tensor(z["omega"], device=DEV)
            tot, ptot, rpa, pf = spectral_integrals(st, w, knots)
            btot = band_integrals(pf, w, om)
            sig_max = float(st.max())
            if "sigma_spin" in z.files:
                ss = torch.as_tensor(z["sigma_spin"], device=DEV)
                a, b = spectral_integrals(ss[0], w, knots), spectral_integrals(ss[1], w, knots)
                spin, pspin = a[0] + b[0], a[1] + b[1]
                bspin = band_integrals(a[3], w, om) + band_integrals(b[3], w, om)
            else:
                h = spectral_integrals(0.5 * st, w, knots)
                spin, pspin = 2 * h[0], 2 * h[1]
                bspin = 2 * band_integrals(h[3], w, om)
        gpath = root / semi_tag / "gridfeat_v1" / (name + ".json")
        grid = json.loads(gpath.read_text()).get("features") if gpath.exists() else None
        rows[sid] = {
            "e_pbe": scf["e_tot"], "exc_pbe": scf["scf_summary"]["exc"], "e_x_exact": corr["e_x_exact"],
            "e_c_rpa": float(rpa), "pt2": corr.get("pt2"), "singles": corr.get("singles"),
            "spec_tot": tot.tolist(), "spec_spin": spin.tolist(), "semi": semi["energies"],
            "poly_tot": ptot.tolist(), "poly_spin": pspin.tolist(), "sigma_max": sig_max,
            "band_tot": btot.tolist(), "band_spin": bspin.tolist(), "grid": grid,
            "min_gap": corr.get("min_excitation_hartree"), "nao": scf["nao"],
            "s2_dev": (scf.get("s_squared", 0) - scf.get("s_squared_exact", 0)) if scf.get("unpaired") else 0.0,
        }
    cache.write_text(json.dumps(table))
    return rows, problems


FEATURE_SETS = {
    "x_semi": [("x", n) for n in X_COMPONENTS],
    "c_semi": [("c", n) for n in C_COMPONENTS],
    "pt2": [("pt2", "0.3", "os"), ("pt2", "0.3", "ss"), ("pt2", "1", "os"), ("pt2", "1", "ss")],
    "singles": [("singles", "0.3"), ("singles", "1")],
    "spec_tot": [("spec_tot", i) for i in range(len(Q_KNOTS))],
    "spec_spin": [("spec_spin", i) for i in range(len(Q_KNOTS))],
    "rpa_scale": [("rpa",)],
    "poly_tot": [("poly_tot", i) for i in range(POLY_ORDER)],
    "poly_spin": [("poly_spin", i) for i in range(POLY_ORDER)],
    # opposite-spin-coupled part only: total minus the two spin channels; zero for one-electron systems
    "poly_cross": [("poly_cross", i) for i in range(POLY_ORDER)],
    "poly_tot3": [("poly_tot", i) for i in range(3)],
    "poly_cross3": [("poly_cross", i) for i in range(3)],
    "pt2_os": [("pt2", "0.3", "os"), ("pt2", "1", "os")],
    "pt2_ss": [("pt2", "0.3", "ss"), ("pt2", "1", "ss")],
    "band_tot": [("band_tot", l, k) for l in range(len(BAND_KNOTS)) for k in range(BAND_ORDER)],
    "band_cross": [("band_cross", l, k) for l in range(len(BAND_KNOTS)) for k in range(BAND_ORDER)],
    # regime-resolved semilocal integrals from the stored grids
    "gx": [("grid", f"x_{i}{j}") for i in range(4) for j in range(4)],
    "gcss": [("grid", f"{c}_{i}{j}") for c in ("css", "cssv") for i in range(3) for j in range(3)],
    "gcos": [("grid", f"{c}_{i}{j}") for c in ("cos", "cosv", "cosz") for i in range(3) for j in range(3)],
    "x_min": [("x", "gga_x_pbe"), ("x", "gga_x_b88"), ("x", "mgga_x_scan")],
    "c_min": [("c", "gga_c_pbe"), ("c", "gga_c_lyp"), ("c", "mgga_c_scan")],
}


def feature_value(row, key):
    kind = key[0]
    if kind == "x":
        return row["semi"][key[1]] - row["e_x_exact"]
    if kind == "c":
        return row["semi"][key[1]]
    if kind == "pt2":
        return row["pt2"][key[1]][key[2]] if row["pt2"] else 0.0
    if kind == "singles":
        return row["singles"][key[1]] if row["singles"] else float("nan")
    if kind in ("spec_tot", "spec_spin", "poly_tot", "poly_spin"):
        return row[kind][key[1]]
    if kind == "poly_cross":
        return row["poly_tot"][key[1]] - row["poly_spin"][key[1]]
    if kind == "band_tot":
        return row["band_tot"][key[1]][key[2]]
    if kind == "band_cross":
        return row["band_tot"][key[1]][key[2]] - row["band_spin"][key[1]][key[2]]
    if kind == "grid":
        if row.get("grid") is None:
            return float("nan")
        v = row["grid"][key[1]]
        return v - row["e_x_exact"] if key[1] == "x_00" else v
    if kind == "rpa":
        return row["e_c_rpa"]
    raise KeyError(key)


def base_energy(row, base):
    if base == "pbe":
        return row["e_pbe"]
    e = row["e_pbe"] - row["exc_pbe"] + row["e_x_exact"]
    if base == "exx":
        return e
    if base == "exx_rpa":
        return e + row["e_c_rpa"]
    raise KeyError(base)


class Problem:
    """Reaction-level regression problem for the reactions whose structures are all available."""

    def __init__(self, rows, feature_keys, base="exx_rpa"):
        self.suite = suite()
        s = self.suite
        t = s.tensors(DEV)
        ids = s.required_ids
        have = torch.tensor([i in rows for i in ids], device=DEV)
        e0 = torch.tensor([base_energy(rows[i], base) if i in rows else 0.0 for i in ids], dtype=torch.float64, device=DEV)
        f = torch.tensor([[feature_value(rows[i], k) if i in rows else 0.0 for k in feature_keys] for i in ids],
                         dtype=torch.float64, device=DEV).reshape(len(ids), len(feature_keys))
        finite = torch.isfinite(f).all(dim=1) & torch.isfinite(e0)
        have = have & finite
        f = torch.nan_to_num(f)
        missing = (t["matrix"] != 0).to(torch.float64) @ (~have).to(torch.float64)
        self.mask = missing == 0
        self.keys = list(feature_keys)
        self.A = (t["matrix"] @ f * KCAL)[self.mask]
        self.base_pred = (t["matrix"] @ torch.where(have, e0, torch.zeros_like(e0)) * KCAL)[self.mask]
        self.ref = t["references"][self.mask]
        self.w = t["weights"][self.mask]
        self.y = self.ref - self.base_pred
        self.subset = [r["subset"] for r, m in zip(s.reactions, self.mask.tolist()) if m]
        self.rid = [r["id"] for r, m in zip(s.reactions, self.mask.tolist()) if m]
        self.n_total = len(s.reactions)
        names = sorted(set(self.subset))
        self.subset_index = torch.tensor([names.index(x) for x in self.subset], device=DEV)
        self.subset_names = names
        self.complete_subsets = [n for n in names if self.subset.count(n) == s.subsets[n]["count"]]

    def wtmad2(self, pred_err):
        """Weighted mean absolute error over the available reactions (equals WTMAD-2 only at full coverage)."""
        return float((self.w * pred_err.abs()).mean())

    def by_subset(self, err):
        out = {}
        for i, n in enumerate(self.subset_names):
            m = self.subset_index == i
            out[n] = {"n": int(m.sum()), "of": self.suite.subsets[n]["count"], "mad": float(err[m].abs().mean())}
        return out


def solve(A, y, w, lam, masks=None, iters=40, eps=1e-3):
    """Batched robust ridge fit.

    For every row of `masks` (F x n, True = reaction used for training) minimise
        sum_r w_r |y_r - A_r theta| + lam * n_train * |theta / scale|^2
    by iteratively reweighted least squares, all F problems at once on the GPU.
    Columns are scaled by their weighted RMS over all reactions so one lam applies to every feature.
    Returns theta with shape (F, p) in the original feature units.
    """
    n, p = A.shape
    if masks is None:
        masks = torch.ones((1, n), dtype=torch.bool, device=A.device)
    f = masks.shape[0]
    if p == 0:
        return torch.zeros((f, 0), dtype=A.dtype, device=A.device)
    scale = torch.sqrt((w[:, None] * A * A).mean(0)).clamp_min(1e-12)
    As = A / scale
    m = masks.to(A.dtype)
    base_w = m * w[None, :]
    wt = base_w.clone()
    reg = lam * m.sum(1)[:, None, None] * torch.eye(p, dtype=A.dtype, device=A.device)[None]
    theta = torch.zeros((f, p), dtype=A.dtype, device=A.device)
    for _ in range(iters):
        G = torch.einsum("rp,fr,rq->fpq", As, wt, As) + reg
        b = torch.einsum("rp,fr->fp", As, wt * y[None, :])
        new = torch.linalg.solve(G, b)
        r = y[None, :] - new @ As.T
        wt = base_w / torch.sqrt(r * r + eps * eps)
        done = float((new - theta).abs().max()) < 1e-9 * (1 + float(new.abs().max()))
        theta = new
        if done:
            break
    return theta / scale[None, :]


def fold_masks(prob, seeds=(0, 1, 2), folds=5):
    """Training masks: [all], stratified random folds for each seed, leave-one-subset-out."""
    n = len(prob.y)
    sub = prob.subset_index.cpu()
    rows, kinds = [torch.ones(n, dtype=torch.bool)], [("all", None, None)]
    fold_ids = []
    for seed in seeds:
        g = torch.Generator(device="cpu").manual_seed(seed)
        fold = torch.zeros(n, dtype=torch.long)
        for i in range(len(prob.subset_names)):
            idx = torch.nonzero(sub == i).ravel()
            perm = idx[torch.randperm(len(idx), generator=g)]
            fold[perm] = torch.arange(len(perm)) % folds
        fold_ids.append(fold)
        for k in range(folds):
            rows.append(fold != k)
            kinds.append(("random", seed, k))
    for i in range(len(prob.subset_names)):
        rows.append(sub != i)
        kinds.append(("subset", i, None))
    return torch.stack(rows).to(DEV), kinds


def fit_and_validate(prob, lam, seeds=(0, 1, 2), folds=5):
    A, y, w = prob.A, prob.y, prob.w
    n = len(y)
    out = {"n_reactions": n, "of": prob.n_total, "n_features": A.shape[1], "lambda": lam}
    masks, kinds = fold_masks(prob, seeds, folds)
    theta = solve(A, y, w, lam, masks)
    err = theta @ A.T - y[None, :]          # (F, n) errors of every fold model on every reaction
    out["theta"] = theta[0].tolist()
    out["in_sample"] = prob.wtmad2(err[0])
    out["base_only"] = prob.wtmad2(-y)
    out["in_sample_err"] = err[0]
    cv_vals = []
    cv_err = None
    for seed in seeds:
        e = torch.zeros_like(y)
        for j, (kind, s_, k) in enumerate(kinds):
            if kind == "random" and s_ == seed:
                held = ~masks[j]
                e[held] = err[j][held]
        cv_vals.append(prob.wtmad2(e))
        cv_err = e
    out["cv_random"] = float(np.mean(cv_vals))
    out["cv_random_spread"] = float(np.std(cv_vals))
    out["cv_random_err"] = cv_err
    e = torch.zeros_like(y)
    for j, (kind, i, _) in enumerate(kinds):
        if kind == "subset":
            held = ~masks[j]
            e[held] = err[j][held]
    out["cv_leave_subset_out"] = prob.wtmad2(e)
    out["cv_lso_err"] = e
    return out
