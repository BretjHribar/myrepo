"""One-off: cost and correctness of the GPU-resident exact-exchange energy density (restricted cases)."""
import json
import sys
import time
import numpy as np
from .common import BULK, dataset_structures, pin_one_core, row_mol
pin_one_core()
import cupy as cp
from gpu4pyscf.dft import gen_grid, numint
from .lxx import LocalExchange


def main():
    sid = sys.argv[1]
    level = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    row = next(r for r in dataset_structures("gmtkn55") if r["id"] == sid)
    mol = row_mol("gmtkn55", row)
    z = np.load(BULK / "gmtkn55/pbe_qzvp_g3/orbitals" / (sid.replace("/", "__") + ".npz"))
    c, occ = cp.asarray(z["mo_coeff"]), cp.asarray(z["mo_occ"])
    dm_s = (c[:, occ > 0]) @ (c[:, occ > 0]).T
    grids = gen_grid.Grids(mol); grids.level = level; grids.build()
    coords, w = grids.coords, grids.weights
    n, nao = coords.shape[0], mol.nao
    t0 = time.time()
    lx = LocalExchange(mol)
    t_build = time.time() - t0
    batch = int(sys.argv[4]) if len(sys.argv) > 4 else 16384
    ex = cp.zeros(n); exlr = cp.zeros(n)
    t1 = time.time(); t_int = 0.0
    for p0 in range(0, n, batch):
        cc = coords[p0:p0 + batch]
        ao = numint.eval_ao(mol, cc, deriv=0)
        if ao.shape[0] != cc.shape[0]:
            ao = ao.T
        f = lx.sort(ao @ dm_s)
        t = time.time()
        ex[p0:p0 + batch] = -lx.density(cc, [f])[0]            # -1/2 per spin, two spins
        if len(sys.argv) > 3:
            exlr[p0:p0 + batch] = -lx.density(cc, [f], exponent=float(sys.argv[3]) ** 2)[0]
        cp.cuda.Stream.null.synchronize(); t_int += time.time() - t
    total = float(cp.dot(w, ex))
    dt = time.time() - t1
    cj = BULK / "gmtkn55/pbe_qzvp_g3/corr_v1/results" / (sid.replace("/", "__") + ".json")
    ref = json.loads(cj.read_text())["e_x_exact"] if cj.exists() else None
    print(f"{sid} nao {nao} level {level} points {n} batch {batch}: integrated {total:.6f}  DF reference {ref}  "
          f"setup {t_build:.1f}s  loop {dt:.1f}s (kernel+contract {t_int:.1f}s)  {dt/n*1e3:.3f} ms/point  "
          f"{n*nao*nao/max(t_int,1e-9)/1e9:.2f}e9 pair-points/s  LR {float(cp.dot(w, exlr)):.6f}", flush=True)


if __name__ == "__main__":
    main()
