"""Semilocal ingredients at the saved Kohn-Sham density, on the GPU.

For every structure: the spin densities, their gradients' invariants and the kinetic-energy
densities on the SCF integration grid (stored compactly so any semilocal or learned form can be
integrated later), plus the integrated energies of a fixed list of exchange and correlation
components used as basis functions in our own fitted functional. No reference energies are used.
"""
import argparse
import json
import time
import traceback

from .common import BULK, dataset_dir, dataset_structures, pin_one_core, row_mol

pin_one_core()
import numpy as np
import cupy as cp
import gpu4pyscf
from gpu4pyscf.dft import libxc, numint, rks, uks
from . import run_gridfeat as gf

# libxc component names; each is evaluated alone at the PBE self-consistent density.
COMPONENTS = (
    "lda_x", "gga_x_pbe", "gga_x_b88", "gga_x_rpbe", "gga_x_pbe_sol", "gga_x_pw86",
    "mgga_x_tpss", "mgga_x_scan", "mgga_x_r2scan", "mgga_x_m06_l", "mgga_x_revtpss",
    "lda_c_pw", "gga_c_pbe", "gga_c_lyp", "gga_c_p86", "gga_c_pbe_sol",
    "mgga_c_tpss", "mgga_c_scan", "mgga_c_r2scan", "mgga_c_bc95", "mgga_c_m06_l", "mgga_c_revtpss",
)


def free():
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()


def density_on_grid(mol, grids, dms):
    """rho[spin, (rho, dx, dy, dz, tau), point] in the order of grids.weights."""
    ni = numint.NumInt()
    ni.build(mol, grids.coords)
    opt = ni.gdftopt
    smol = opt._sorted_mol
    nao = smol.nao
    dms = [opt.sort_orbitals(cp.asarray(dm), axis=[0, 1]) for dm in dms]
    n = grids.coords.shape[0]
    rho = cp.empty((len(dms), 5, n))
    p1 = 0
    for ao, idx, weight, _ in ni.block_loop(smol, grids, nao, 1, max_memory=None):
        p0, p1 = p1, p1 + weight.size
        for s, dm in enumerate(dms):
            rho[s, :, p0:p1] = numint.eval_rho(smol, ao, dm[idx[:, None], idx], xctype="MGGA", hermi=1, with_lapl=False)
    if p1 != n:
        raise RuntimeError("grid loop did not cover every point")
    return rho


def run_one(dataset, row, scf, scf_dir, store_grid):
    t0 = time.time()
    sid = row["id"]
    mol = row_mol(dataset, row)
    z = np.load(scf_dir / "orbitals" / (sid.replace("/", "__") + ".npz"))
    restricted = z["mo_coeff"].ndim == 2   # the SCF pass may have used the unrestricted fallback
    mf = (rks.RKS if restricted else uks.UKS)(mol, xc="pbe")
    mf.grids.level = scf.get("grid_level", 3)
    mf.grids.build()
    c, o = cp.asarray(z["mo_coeff"]), cp.asarray(z["mo_occ"])
    if restricted:
        dms = [0.5 * (c * o) @ c.T] * 2          # two identical spin channels
        rho_s = density_on_grid(mol, mf.grids, dms[:1])
        rho = cp.concatenate([rho_s, rho_s])
    else:
        rho = density_on_grid(mol, mf.grids, [(c[s] * o[s]) @ c[s].T for s in (0, 1)])
    w = cp.asarray(mf.grids.weights)
    n = w.size
    den = (rho[0, 0] + rho[1, 0]) * w
    res = {"id": sid, "nao": int(mol.nao), "ngrid": int(n), "grid_level": int(mf.grids.level),
           "n_electrons_integrated": float(den.sum()), "n_electrons": int(mol.nelectron)}
    res["electron_count_error"] = abs(res["n_electrons_integrated"] - res["n_electrons"])
    inp = {"rho": cp.ascontiguousarray(cp.stack([rho[0, 0], rho[1, 0]], axis=1)),
           "sigma": cp.ascontiguousarray(cp.stack([(rho[0, 1:4] ** 2).sum(0), (rho[0, 1:4] * rho[1, 1:4]).sum(0),
                                                    (rho[1, 1:4] ** 2).sum(0)], axis=1)),
           "tau": cp.ascontiguousarray(cp.stack([rho[0, 4], rho[1, 4]], axis=1))}
    energies, skipped = {}, {}
    for name in COMPONENTS:
        try:
            fun = libxc.XCfun(name, "polarized")
            if not getattr(fun, "on_gpu", True):
                skipped[name] = "not available in the GPU functional library"
                continue
            family = name.split("_")[0]
            args = {"rho": inp["rho"]}
            if family in ("gga", "mgga"):
                args["sigma"] = inp["sigma"]
            if family == "mgga":
                args["tau"] = inp["tau"]
            zk = fun.compute(args, do_exc=True, do_vxc=False)["zk"].ravel()
            energies[name] = float(cp.dot(den, zk))
        except Exception as exc:  # keep the reason; never fall back to CPU
            skipped[name] = repr(exc)[:160]
    res["energies"], res["skipped"] = energies, skipped
    res["e_xc_pbe_check"] = energies.get("gga_x_pbe", float("nan")) + energies.get("gga_c_pbe", float("nan"))
    res["e_xc_scf"] = scf["scf_summary"].get("exc")
    res["e_xc_mismatch"] = abs(res["e_xc_pbe_check"] - res["e_xc_scf"])
    # regime-resolved semilocal integrals, computed here while the density is on the GPU
    res["gridfeat"] = gf.features(w, inp["rho"], cp.ascontiguousarray(inp["sigma"][:, [0, 2]]), inp["tau"])
    arrays = None
    if store_grid:
        if restricted:
            arrays = {"weights": cp.asnumpy(w), "rho": cp.asnumpy(rho[0, 0]).astype(np.float32),
                      "sigma": cp.asnumpy(inp["sigma"][:, 0]).astype(np.float32),
                      "tau": cp.asnumpy(rho[0, 4]).astype(np.float32)}
        else:
            arrays = {"weights": cp.asnumpy(w), "rho": cp.asnumpy(inp["rho"]).astype(np.float32),
                      "sigma": cp.asnumpy(inp["sigma"]).astype(np.float32),
                      "tau": cp.asnumpy(inp["tau"]).astype(np.float32)}
    res["seconds"] = round(time.time() - t0, 2)
    del rho, inp, mf
    free()
    return res, arrays


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="gmtkn55")
    ap.add_argument("--scf-tag", required=True)
    ap.add_argument("--tag", default="semi_v1")
    ap.add_argument("--limit", type=int, default=0, help="use only the first N rows of the dataset order")
    ap.add_argument("--ids", default="")
    ap.add_argument("--max-nao", type=int, default=10**9)
    ap.add_argument("--no-grid", action="store_true")
    ap.add_argument("--follow", action="store_true")
    args = ap.parse_args()
    scf_dir = dataset_dir(args.dataset) / args.scf_tag
    out = scf_dir / args.tag
    (out / "results").mkdir(parents=True, exist_ok=True)
    (out / "grid").mkdir(parents=True, exist_ok=True)
    rows = dataset_structures(args.dataset)
    if args.limit:
        rows = rows[:args.limit]
    if args.ids:
        wanted = set(args.ids.split(","))
        rows = [r for r in rows if r["id"] in wanted]
    rows = [r for r in rows if r["nao"] <= args.max_nao]
    card = {"tag": args.tag, "scf_tag": args.scf_tag, "components": COMPONENTS,
            "density": "PBE self-consistent density from saved orbitals, SCF integration grid",
            "grid_storage": "float32 rho/sigma/tau per spin, float64 weights",
            "engine": f"gpu4pyscf {gpu4pyscf.__version__} grid evaluation + GPU libxc {libxc.__version__}"}
    (out / "run_card.json").write_text(json.dumps(card, indent=1))
    t_start, new = time.time(), 0
    while True:
        waiting = 0
        for i, row in enumerate(rows):
            name = row["id"].replace("/", "__")
            dest = out / "results" / (name + ".json")
            if dest.exists():
                continue
            scf_json = scf_dir / "results" / (name + ".json")
            if not scf_json.exists():
                waiting += 1
                continue
            scf = json.loads(scf_json.read_text())
            arrays = None
            if not scf.get("converged"):
                res = {"id": row["id"], "error": "SCF input not converged"}
            else:
                try:
                    res, arrays = run_one(args.dataset, row, scf, scf_dir, not args.no_grid)
                except Exception as exc:
                    res = {"id": row["id"], "error": repr(exc)[:400], "traceback": traceback.format_exc()[-1200:],
                           "nao": row["nao"]}
                    free()
            if res.get("gridfeat") is not None:
                gdir = out / "gridfeat_v1"
                gdir.mkdir(exist_ok=True)
                gtmp = gdir / (name + ".tmp")
                gtmp.write_text(json.dumps({"id": row["id"], "features": res.pop("gridfeat")}))
                gtmp.replace(gdir / (name + ".json"))
            if arrays is not None:
                tmp = out / "grid" / (name + ".tmp.npz")
                np.savez(tmp, **arrays)
                tmp.replace(out / "grid" / (name + ".npz"))
            tmp = dest.with_suffix(".tmp")
            tmp.write_text(json.dumps(res, indent=1))
            tmp.replace(dest)
            new += 1
            print(json.dumps({"i": i + 1, "n": len(rows), "id": row["id"], "nao": res.get("nao"),
                              "xc_mismatch": res.get("e_xc_mismatch"), "nelec_err": res.get("electron_count_error"),
                              "skipped": list((res.get("skipped") or {}).keys()), "s": res.get("seconds"),
                              "err": res.get("error"), "elapsed_min": round((time.time() - t_start) / 60, 1)}), flush=True)
        if not waiting or not args.follow:
            break
        time.sleep(45)
    print(json.dumps({"finished": True, "new": new}), flush=True)


if __name__ == "__main__":
    main()
