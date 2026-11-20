"""One-off: does a spin-symmetry-broken solution give a proper aufbau state where restricted PBE has none?"""
import sys
import time
import numpy as np
from .common import BULK, dataset_structures, pin_one_core, row_mol
pin_one_core()
import cupy as cp
from gpu4pyscf.dft import rks, uks
from . import run_scf as rs


def report(label, mf, t):
    occ = cp.asnumpy(cp.asarray(mf.mo_occ)); eig = cp.asnumpy(cp.asarray(mf.mo_energy))
    if occ.ndim == 1:
        gaps = [eig[occ == 0].min() - eig[occ > 0].max()]
        s2 = 0.0
    else:
        gaps = [eig[s][occ[s] == 0].min() - eig[s][occ[s] > 0].max() for s in (0, 1)]
        s2 = rs.spin_square(mf, mf.mol)
    print(f"{label:40s} conv {mf.converged}  E {float(mf.e_tot):.8f}  gaps {[round(float(g), 5) for g in gaps]}  <S2> {s2:.4f}  {time.time()-t:.1f}s", flush=True)


def main():
    sid = sys.argv[1]
    row = next(r for r in dataset_structures("gmtkn55") if r["id"] == sid)
    mol = row_mol("gmtkn55", row)
    z = np.load(BULK / "gmtkn55/pbe_qzvp_g3/orbitals" / (sid.replace("/", "__") + ".npz"))
    c, occ = z["mo_coeff"], z["mo_occ"]
    nocc = int((occ > 0).sum())
    ca = c[:, :nocc]
    cb = np.concatenate([c[:, :nocc - 1], c[:, nocc:nocc + 1]], axis=1)   # beta: HOMO -> LUMO
    dm0 = cp.asarray(np.stack([ca @ ca.T, cb @ cb.T]))

    def base():
        mf = uks.UKS(mol, xc="pbe").density_fit(auxbasis=rs.AUX_J)
        mf.grids.level = 3; mf.conv_tol = 1e-9; mf.verbose = 0; mf.max_cycle = 150
        return mf

    t = time.time(); mf = base(); mf.kernel(dm0=dm0); report("UKS, HOMO/LUMO-swapped beta guess, DIIS", mf, t)
    if not mf.converged:
        t = time.time(); d = mf.make_rdm1(); mf = base().newton(); mf.kernel(dm0=d); report("  then second-order", mf, t)
    # mixed guess: rotate the frontier pair by 45 degrees in opposite senses for the two spins
    h, l = c[:, nocc - 1], c[:, nocc]
    a = np.concatenate([c[:, :nocc - 1], ((h + l) / np.sqrt(2))[:, None]], axis=1)
    b = np.concatenate([c[:, :nocc - 1], ((h - l) / np.sqrt(2))[:, None]], axis=1)
    dm1 = cp.asarray(np.stack([a @ a.T, b @ b.T]))
    t = time.time(); mf = base(); mf.kernel(dm0=dm1); report("UKS, 45-degree mixed frontier guess, DIIS", mf, t)
    if not mf.converged:
        t = time.time(); d = mf.make_rdm1(); mf = base().newton(); mf.kernel(dm0=d); report("  then second-order", mf, t)


if __name__ == "__main__":
    main()
