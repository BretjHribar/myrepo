"""Checkpointed GPU Kohn-Sham runs over GMTKN55 structures (or any structure list).

Run card for every output row: functional, basis, auxiliary basis, grid level, SCF thresholds,
retry stage used, software versions. One structure at a time on one GPU; results and orbitals
are written per structure so an interrupted run resumes without repeating finished work.
"""
import argparse
import json
import time
import traceback
from pathlib import Path

from .common import BASIS, BULK, dataset_dir, dataset_structures, pin_one_core, row_mol

pin_one_core()
import numpy as np
import cupy as cp
import pyscf
import gpu4pyscf
from gpu4pyscf.dft import rks, uks

AUX_J = "def2-universal-jfit"
# Deterministic retry schedule; the stage that converged is recorded with the result.
# Level-shifted stages can land on a non-aufbau state (seen for singlet C2: 0.1 Hartree too high, and
# the package then correctly reports non-convergence), so the second-order solver is tried first.
STAGES = (
    {"name": "diis", "max_cycle": 100, "level_shift": 0.0, "damp": 0.0},
    {"name": "second_order", "max_cycle": 100, "level_shift": 0.0, "damp": 0.0, "newton": True, "max_nao": 700},
    {"name": "diis_shift0.25_damp0.3", "max_cycle": 250, "level_shift": 0.25, "damp": 0.3},
    {"name": "diis_shift0.6_damp0.6", "max_cycle": 500, "level_shift": 0.6, "damp": 0.6},
)


def sync():
    cp.cuda.Stream.null.synchronize()


def free():
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()


def to_np(x):
    return cp.asnumpy(x) if isinstance(x, cp.ndarray) else np.asarray(x)


def make_mf(mol, unpaired, xc, grid_level, conv_tol, force_unrestricted=False):
    mf = (uks.UKS if (unpaired or force_unrestricted) else rks.RKS)(mol, xc=xc)
    mf = mf.density_fit(auxbasis=AUX_J)
    mf.grids.level = grid_level
    mf.conv_tol = conv_tol
    mf.verbose = 0
    return mf


def spin_square(mf, mol):
    mo, occ = mf.mo_coeff, mf.mo_occ
    s = cp.asarray(mol.intor("int1e_ovlp"))
    ca, cb = mo[0][:, occ[0] > 0], mo[1][:, occ[1] > 0]
    na, nb = ca.shape[1], cb.shape[1]
    sab = ca.T @ s @ cb
    sz = 0.5 * (na - nb)
    return float(sz * (sz + 1) + nb - float((sab ** 2).sum()))


def run_one(dataset, row, outdir, xc, grid_level, conv_tol, save_orbitals=True):
    t0 = time.time()
    unpaired = int(row["unpaired_electrons"])
    mol = row_mol(dataset, row)
    result = {"id": row["id"], "xc": xc, "basis": BASIS, "aux_basis_j": AUX_J, "grid_level": grid_level,
              "conv_tol": conv_tol, "method": "UKS" if unpaired else "RKS", "nao": int(mol.nao),
              "natm": int(mol.natm), "nelec": int(mol.nelectron), "charge": int(row["charge"]),
              "unpaired": unpaired, "has_ecp": bool(mol.has_ecp()),
              "software": {"gpu4pyscf": gpu4pyscf.__version__, "pyscf": pyscf.__version__, "cupy": cp.__version__}}
    dm0, attempts = None, []
    mf = None
    for stage in STAGES:
        if mol.nao > stage.get("max_nao", 10**9):
            continue
        mf = make_mf(mol, unpaired, xc, grid_level, conv_tol)
        if stage.get("newton"):
            mf = mf.newton()
        mf.max_cycle, mf.level_shift, mf.damp = stage["max_cycle"], stage["level_shift"], stage["damp"]
        ts = time.time()
        e = mf.kernel(dm0=dm0)
        sync()
        attempts.append({"stage": stage["name"], "converged": bool(mf.converged), "cycles": int(getattr(mf, "cycles", -1)),
                         "e_tot": float(e), "seconds": round(time.time() - ts, 2)})
        if mf.converged:
            break
        dm0 = mf.make_rdm1()
    broken_symmetry = False
    if not unpaired:
        occ_r, eig_r = to_np(mf.mo_occ), to_np(mf.mo_energy)
        gap_r = float(eig_r[occ_r == 0].min() - eig_r[occ_r > 0].max()) if (occ_r == 0).any() else 1.0
        result["restricted_gap_hartree"] = gap_r
        if (not mf.converged) or gap_r <= 0:
            # Restricted PBE has no proper (aufbau) state here. Rule: take the spin-symmetry-broken
            # unrestricted solution started from a 45-degree mix of the frontier pair, if it converges
            # with positive gaps in both spin channels. <S^2> is recorded with the result.
            c_r = to_np(mf.mo_coeff)
            o_idx, v_idx = np.where(occ_r > 0)[0], np.where(occ_r == 0)[0]
            h, l = o_idx[np.argmax(eig_r[o_idx])], v_idx[np.argmin(eig_r[v_idx])]
            keep = c_r[:, [i for i in o_idx if i != h]]
            ca = np.concatenate([keep, ((c_r[:, h] + c_r[:, l]) / np.sqrt(2))[:, None]], axis=1)
            cb = np.concatenate([keep, ((c_r[:, h] - c_r[:, l]) / np.sqrt(2))[:, None]], axis=1)
            dm_bs = cp.asarray(np.stack([ca @ ca.T, cb @ cb.T]))
            for use_newton in (False, True):
                mfu = make_mf(mol, 0, xc, grid_level, conv_tol, force_unrestricted=True)
                if use_newton:
                    mfu = mfu.newton()
                mfu.max_cycle = 150
                ts = time.time()
                e = mfu.kernel(dm0=dm_bs)
                sync()
                ou, eu = to_np(mfu.mo_occ), to_np(mfu.mo_energy)
                gaps = [float(eu[k][ou[k] == 0].min() - eu[k][ou[k] > 0].max()) for k in (0, 1)]
                ok = bool(mfu.converged) and min(gaps) > 0
                attempts.append({"stage": "broken_symmetry_unrestricted" + ("_second_order" if use_newton else ""),
                                 "converged": bool(mfu.converged), "cycles": int(getattr(mfu, "cycles", -1)),
                                 "e_tot": float(e), "gaps": gaps, "seconds": round(time.time() - ts, 2)})
                if ok:
                    mf, broken_symmetry = mfu, True
                    break
                dm_bs = mfu.make_rdm1()
    result["broken_symmetry_fallback"] = broken_symmetry
    if broken_symmetry:
        result["method"] = "UKS (spin-symmetry-broken fallback for a closed-shell structure)"
    result["attempts"] = attempts
    result["converged"] = bool(mf.converged)
    result["e_tot"] = float(mf.e_tot)
    result["ngrid"] = int(mf.grids.coords.shape[0])
    summary = {k: float(v) for k, v in getattr(mf, "scf_summary", {}).items()}
    result["scf_summary"] = summary
    occ = to_np(mf.mo_occ)
    eig = to_np(mf.mo_energy)
    if unpaired or broken_symmetry:
        result["s_squared"] = spin_square(mf, mol)
        result["s_squared_exact"] = 0.25 * unpaired * (unpaired + 2)
        homo = max(float(eig[s][occ[s] > 0].max()) for s in (0, 1) if (occ[s] > 0).any())
        gaps = [float(eig[s][occ[s] == 0].min() - eig[s][occ[s] > 0].max()) for s in (0, 1) if (occ[s] > 0).any()]
        result["min_gap_hartree"] = min(gaps) if gaps else None
    else:
        homo = float(eig[occ > 0].max())
    result["homo_hartree"] = homo
    result["finite"] = bool(np.isfinite(mf.e_tot))
    if save_orbitals:
        orb = outdir / "orbitals" / (row["id"].replace("/", "__") + ".npz")
        orb.parent.mkdir(parents=True, exist_ok=True)
        tmp = orb.with_suffix(".tmp.npz")
        np.savez(tmp, mo_coeff=to_np(mf.mo_coeff), mo_occ=occ, mo_energy=eig)
        tmp.replace(orb)
        result["orbital_file"] = str(orb.relative_to(outdir))
    result["seconds"] = round(time.time() - t0, 2)
    del mf
    free()
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", required=True, help="output folder name under BULK/gmtkn55/")
    ap.add_argument("--dataset", default="gmtkn55")
    ap.add_argument("--xc", default="pbe")
    ap.add_argument("--grid-level", type=int, default=3)
    ap.add_argument("--conv-tol", type=float, default=1e-9)
    ap.add_argument("--ids", default="", help="comma-separated structure ids (default: all required)")
    ap.add_argument("--max-nao", type=int, default=10**9)
    ap.add_argument("--min-nao", type=int, default=0)
    ap.add_argument("--no-orbitals", action="store_true")
    args = ap.parse_args()

    rows = dataset_structures(args.dataset)
    if args.ids:
        wanted = set(args.ids.split(","))
        rows = [r for r in rows if r["id"] in wanted]
    rows = [r for r in rows if args.min_nao <= r["nao"] <= args.max_nao]
    outdir = dataset_dir(args.dataset) / args.tag
    (outdir / "results").mkdir(parents=True, exist_ok=True)
    card = {"tag": args.tag, "dataset": args.dataset, "xc": args.xc, "basis": BASIS, "aux_basis_j": AUX_J, "grid_level": args.grid_level,
            "conv_tol": args.conv_tol, "retry_stages": STAGES, "evidence_level": "full public suite input run",
            "engine": f"gpu4pyscf {gpu4pyscf.__version__} (CUDA), density-fitted J, one GPU, one pinned host core"}
    (outdir / "run_card.json").write_text(json.dumps(card, indent=1))
    done = 0
    t_start = time.time()
    for i, row in enumerate(rows):
        dest = outdir / "results" / (row["id"].replace("/", "__") + ".json")
        if dest.exists():
            continue
        try:
            res = run_one(args.dataset, row, outdir, args.xc, args.grid_level, args.conv_tol, not args.no_orbitals)
        except Exception as exc:  # keep every failure visible
            res = {"id": row["id"], "converged": False, "finite": False, "error": repr(exc)[:500],
                   "traceback": traceback.format_exc()[-1500:], "nao": row["nao"]}
            free()
        tmp = dest.with_suffix(".tmp")
        tmp.write_text(json.dumps(res, indent=1))
        tmp.replace(dest)
        done += 1
        print(json.dumps({"i": i + 1, "n": len(rows), "id": row["id"], "nao": res.get("nao"), "conv": res.get("converged"),
                          "e": res.get("e_tot"), "s": res.get("seconds"), "err": res.get("error"),
                          "elapsed_min": round((time.time() - t_start) / 60, 1)}), flush=True)
    print(json.dumps({"finished": True, "new": done, "total_listed": len(rows)}), flush=True)


if __name__ == "__main__":
    main()
