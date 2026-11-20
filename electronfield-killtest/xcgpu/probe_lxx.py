"""One-off: cost and correctness of the exact-exchange energy density on a grid (restricted cases).

e_x(r) = -1/2 sum_spin sum_{mu nu} F_mu(r) A_{mu nu}(r) F_nu(r),  F(r) = D_spin phi(r),
A_{mu nu}(r) = integral chi_mu(r') chi_nu(r') / |r - r'| dr'.  Its grid integral must equal the
exact exchange energy of the determinant.
"""
import sys
import time
import numpy as np
from .common import BULK, dataset_structures, pin_one_core, row_mol
pin_one_core()
import cupy as cp
from gpu4pyscf.dft import gen_grid, numint
from gpu4pyscf.gto.int3c1e import int1e_grids


def main():
    sid = sys.argv[1]
    level = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    row = next(r for r in dataset_structures("gmtkn55") if r["id"] == sid)
    mol = row_mol("gmtkn55", row)
    z = np.load(BULK / "gmtkn55/pbe_qzvp_g3/orbitals" / (sid.replace("/", "__") + ".npz"))
    c, occ = cp.asarray(z["mo_coeff"]), cp.asarray(z["mo_occ"])
    dm_s = (c[:, occ > 0]) @ (c[:, occ > 0]).T          # one spin channel
    grids = gen_grid.Grids(mol); grids.level = level; grids.build()
    coords, w = grids.coords, grids.weights
    n, nao = coords.shape[0], mol.nao
    print(sid, "nao", nao, "grid level", level, "points", n, flush=True)
    batch = max(int(1.0e9 / (8 * nao * nao)), 8)
    ex = cp.zeros(n)
    t0 = time.time(); t_int = 0.0
    for p0 in range(0, n, batch):
        cc = coords[p0:p0 + batch]
        ao = numint.eval_ao(mol, cc, deriv=0)             # (nb, nao)
        if ao.shape[0] != cc.shape[0]:
            ao = ao.T
        f = ao @ dm_s                                     # (nb, nao)
        t = time.time()
        a = int1e_grids(mol, cc)                          # (nb, nao, nao)
        cp.cuda.Stream.null.synchronize(); t_int += time.time() - t
        ex[p0:p0 + batch] = -0.5 * 2 * cp.einsum("gm,gmn,gn->g", f, a, f)   # both spins
    cp.cuda.Stream.null.synchronize()
    total = float(cp.dot(w, ex))
    dt = time.time() - t0
    ref = None
    import json
    cj = BULK / "gmtkn55/pbe_qzvp_g3/corr_v1/results" / (sid.replace("/", "__") + ".json")
    if cj.exists():
        ref = json.loads(cj.read_text())["e_x_exact"]
    print(f"integrated exact exchange {total:.6f}  density-fitted reference {ref}  total {dt:.1f}s (integrals {t_int:.1f}s)  "
          f"{dt/n*1e3:.3f} ms/point  {n*nao*nao/max(t_int,1e-9)/1e9:.2f}e9 integrals/s", flush=True)


if __name__ == "__main__":
    main()
