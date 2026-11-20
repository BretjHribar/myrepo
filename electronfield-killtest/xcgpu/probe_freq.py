"""One-off: convergence of the frequency quadrature for the response spectra (restricted cases)."""
import sys
import numpy as np
from .common import BULK, GMTKN_DATA, build_mol, gmtkn_structures, read_xyz
from . import run_corr as rc
import cupy as cp
from gpu4pyscf.mp import dfmp2_addons as ad


def gauss_legendre_log(dmin, dmax, n, pad):
    """Gauss-Legendre in ln(omega) between dmin/pad and dmax*pad, with the same end corrections."""
    lo, hi = np.log(dmin / pad), np.log(dmax * pad)
    x, w = np.polynomial.legendre.leggauss(n)
    u = 0.5 * (hi - lo) * x + 0.5 * (hi + lo)
    om = np.exp(u)
    wt = 0.5 * (hi - lo) * w * om
    return om, wt, (np.exp(lo), np.exp(hi))


def main():
    tag, sid = sys.argv[1], sys.argv[2]
    row = {r["id"]: r for r in gmtkn_structures()[0]}[sid]
    mol = build_mol(read_xyz(GMTKN_DATA / row["coordinate_path"]), row["charge"], row["unpaired_electrons"])
    z = np.load(BULK / "gmtkn55" / tag / "orbitals" / (sid.replace("/", "__") + ".npz"))
    ch = rc.channels(z, 0, rc.n_frozen(mol))[0]
    fit = rc.Fitter(mol, rc.AUX_RI)
    b = fit.cderi(ch["c_act"], ch["c_vir"])
    naux = b.shape[2]
    r = ad.get_dfmp2_energy_pair_intra(mol, b, ch["e_act"], ch["e_vir"])
    exact2 = 2 * float(r[0].sum())
    d = (ch["e_vir"][None, :] - ch["e_act"][:, None]).ravel()
    b2 = b.reshape(-1, naux)
    dmin, dmax = float(d.min()), float(d.max())
    print(sid, "naux", naux, "dmin", dmin, "dmax", dmax, "exact second order", exact2)

    def evaluate(om, w, ends=None):
        e2 = rpa = 0.0
        first = last = None
        for o, ww in zip(om, w):
            s = cp.clip(cp.linalg.eigvalsh(rc.response_matrix(b2, d, float(o), 4.0, 100000), UPLO="L"), 0, None)
            f2 = float((-0.5 * s * s).sum()); fr = float((cp.log1p(s) - s).sum())
            e2 += ww * f2; rpa += ww * fr
        if ends is not None:  # add flat-below and omega^-4-above pieces from separately evaluated end points
            for o, fac in ((ends[0], 1.0), (ends[1], 1.0 / 3)):
                s = cp.clip(cp.linalg.eigvalsh(rc.response_matrix(b2, d, float(o), 4.0, 100000), UPLO="L"), 0, None)
                e2 += o * fac * float((-0.5 * s * s).sum()); rpa += o * fac * float((cp.log1p(s) - s).sum())
        return e2 / (2 * np.pi), rpa / (2 * np.pi)

    ref2, refr = evaluate(*rc.frequency_grid(dmin, dmax, 0.15, 300.0))
    print(f"reference trapezoid du=0.15 pad=300: second {ref2:.9f} (err vs exact {ref2-exact2:+.2e}) rpa {refr:.9f}")
    for du, pad in ((0.8, 20), (0.7, 20), (0.6, 20), (0.5, 20), (0.7, 40), (0.6, 40), (0.5, 40), (0.4, 40), (0.5, 10), (0.6, 10)):
        om, w = rc.frequency_grid(dmin, dmax, du, pad)
        e2, er = evaluate(om, w)
        print(f"trapezoid du={du} pad={pad} n={len(om)}: second err {e2-exact2:+.2e}  rpa err {er-refr:+.2e}")
    for n, pad in ():
        om, w, ends = gauss_legendre_log(dmin, dmax, n, pad)
        e2, er = evaluate(om, w, ends)
        print(f"gauss-legendre-in-log n={n}+2 pad={pad}: second err {e2-exact2:+.2e}  rpa err {er-refr:+.2e}")


if __name__ == "__main__":
    main()
