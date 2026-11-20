"""One-off timing probe: PBE SCF (density-fitted J vs direct J) and one exact-exchange build per size class."""
import json
import sys
import time
from .common import BULK, GMTKN_DATA, build_mol, gmtkn_structures, pin_one_core, read_xyz
pin_one_core()
import cupy as cp
from gpu4pyscf.dft import rks, uks


def sync():
    cp.cuda.Stream.null.synchronize()


def main():
    targets = [int(x) for x in sys.argv[1].split(",")]
    modes = sys.argv[2].split(",")
    inv = json.loads((BULK / "gmtkn55/inventory.json").read_text())
    rows = {r["id"]: r for r in gmtkn_structures()[0]}
    closed = [r for r in inv if r["unpaired"] == 0 and not r["has_ecp"]]
    for target in targets:
        pick = min(closed, key=lambda r: abs(r["nao"] - target))
        row = rows[pick["id"]]
        mol = build_mol(read_xyz(GMTKN_DATA / row["coordinate_path"]), row["charge"], row["unpaired_electrons"])
        for mode in modes:
            t0 = time.time()
            mf = rks.RKS(mol, xc="pbe")
            if mode == "df":
                mf = mf.density_fit(auxbasis="def2-universal-jfit")
            mf.grids.level = 3
            mf.conv_tol = 1e-9
            mf.max_cycle = 60
            try:
                e = mf.kernel(); sync()
                t_scf = time.time() - t0
                out = {"id": pick["id"], "nao": pick["nao"], "natm": pick["natm"], "mode": mode, "e": float(e),
                       "converged": bool(mf.converged), "cycles": int(getattr(mf, "cycles", -1)),
                       "ngrid": int(mf.grids.coords.shape[0]), "scf_s": round(t_scf, 1),
                       "gpu_peak_gb": round(cp.get_default_memory_pool().total_bytes() / 1e9, 2)}
                if mode == "direct" and "k" in modes[-1:]:
                    pass
            except Exception as exc:  # report, never hide
                out = {"id": pick["id"], "nao": pick["nao"], "mode": mode, "error": repr(exc)[:300], "t": round(time.time() - t0, 1)}
            print(json.dumps(out), flush=True)
            if mode == "direct" and "error" not in out and len(sys.argv) > 3:
                t1 = time.time()
                dm = mf.make_rdm1()
                vk = mf.get_k(mol, dm); sync()
                ex = -0.25 * float(cp.einsum("ij,ji->", vk, dm))
                print(json.dumps({"id": pick["id"], "nao": pick["nao"], "k_build_s": round(time.time() - t1, 1), "ex_hf": ex}), flush=True)
            del mf
            cp.get_default_memory_pool().free_all_blocks()


if __name__ == "__main__":
    main()
