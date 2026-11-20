"""Response-pass ingredients for structures whose three-index integrals do not fit on the GPU.

Same quantities as run_corr.run_one, restricted (closed-shell) case, computed in blocks of occupied
orbitals. Three-index blocks are staged on the bulk drive, the response matrices are accumulated for
a few frequencies at a time, and the second-order energies use a block-pair kernel.
Precision: FP64 response for structures up to FP64_MAX_NAO basis functions; above that the staged
blocks and the response accumulation are FP32 (noise of order 1e-4 Hartree, acceptable only because
those structures belong to low-weight, large-energy reactions; the result records which was used).
"""
import shutil
import time

import numpy as np
import cupy as cp
from cupy import cublas

from . import run_corr as rc
from .common import BULK

BLOCK_BYTES = 1.6e9
GROUP_BYTES = 2.0e9
FP64_MAX_NAO = 2400


def occupied_blocks(n, per_block):
    return [np.arange(i, min(i + per_block, n)) for i in range(0, n, per_block)]


def pt2_block_pair(bi, bj, ei, ej, ev, shifts, j_chunk=12):
    """sum over i in block I, j in block J of sum_ab g^2/D and g*g^T/D, g = (ia|jb), D > 0, per shift."""
    direct = np.zeros(len(shifts))
    exch = np.zeros(len(shifts))
    dv = (ev[:, None] + ev[None, :]).astype(cp.float32)
    nv, naux = bi.shape[1], bi.shape[2]
    for i in range(bi.shape[0]):
        for j0 in range(0, bj.shape[0], j_chunk):
            bjc = bj[j0:j0 + j_chunk]
            g = (bi[i] @ bjc.reshape(-1, naux).T).reshape(nv, bjc.shape[0], nv).transpose(1, 0, 2)   # (j, a, b)
            eo = (ei[i] + ej[j0:j0 + j_chunk]).astype(cp.float32)
            for k, shift in enumerate(shifts):
                t = g / (dv[None] - eo[:, None, None] + np.float32(shift))
                direct[k] += float((g * t).sum(dtype=cp.float64))
                exch[k] += float((g.transpose(0, 2, 1) * t).sum(dtype=cp.float64))
            del g, t
    return direct, exch


def run_one_large(dataset, row, scf_dir, du, pad, grid_level, force_blocks=None):
    from .common import row_mol
    t0 = time.time()
    sid = row["id"]
    mol = row_mol(dataset, row)
    z = np.load(scf_dir / "orbitals" / (sid.replace("/", "__") + ".npz"))
    if z["mo_coeff"].ndim != 2:
        raise NotImplementedError("large-structure path is restricted-only")
    ch = rc.channels(z, False, mol)[0]
    nfrozen = rc.n_frozen(mol)
    fp64 = mol.nao <= FP64_MAX_NAO
    res = {"id": sid, "nao": int(mol.nao), "n_frozen_orbitals": nfrozen, "aux_jk": rc.AUX_JK, "aux_ri": rc.AUX_RI,
           "core_selection": "element population", "core_differs_from_energy_order": ch["core_differs_from_energy_order"],
           "restricted": True, "n_active_occ": [int(ch["c_act"].shape[1])], "n_vir": [int(ch["c_vir"].shape[1])],
           "pt2_shifts": list(rc.PT2_SHIFTS), "freq_du": du, "freq_pad": pad, "path": "blocked",
           "response_precision": "FP64" if fp64 else "FP32 staged blocks and accumulation"}
    stage = BULK / "tmp" / ("corr_" + sid.replace("/", "__"))
    shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir(parents=True)
    try:
        # exact exchange and singles
        t = time.time()
        vxc, exc_check = rc.pbe_potential(mol, z, True, grid_level)
        res["e_xc_recomputed"] = exc_check
        fit = rc.Fitter(mol, rc.AUX_JK)
        nocc, nv = ch["c_occ"].shape[1], ch["c_vir"].shape[1]
        bij = fit.cderi(ch["c_occ"], ch["c_occ"])
        res["e_x_exact"] = -float((bij * bij).sum())
        per = max(int(BLOCK_BYTES // (nv * fit.aux.nao_cart() * 8)), 1)
        k_ov = cp.zeros((nocc, nv))
        for blk in occupied_blocks(nocc, force_blocks or per):
            bja = fit.cderi(ch["c_occ"][:, blk], ch["c_vir"])
            k_ov += cp.einsum("ijp,jap->ia", bij[:, blk, :], bja)
            del bja
            rc.free()
        f_ov = -k_ov - ch["c_occ"].T @ vxc @ ch["c_vir"]
        d_all = ch["e_vir_all"][None, :] - ch["e_occ_all"][:, None]
        res["singles"] = {f"{s:g}": -2.0 * float((f_ov * f_ov / (d_all + s)).sum()) for s in rc.PT2_SHIFTS}
        res["fock_ov_norm2"] = 2.0 * float((f_ov * f_ov).sum())
        res["seconds_exchange"] = round(time.time() - t, 2)
        del fit, bij, vxc, k_ov, f_ov
        rc.free()

        # staged three-index blocks
        t = time.time()
        fit = rc.Fitter(mol, rc.AUX_RI)
        naux = fit.aux.nao
        res["naux_ri"] = int(naux)
        nact = ch["c_act"].shape[1]
        per = max(int(BLOCK_BYTES // (nv * fit.aux.nao_cart() * 8)), 1)
        blocks = occupied_blocks(nact, force_blocks or per)
        dtype = np.float64 if fp64 else np.float32
        for k, blk in enumerate(blocks):
            b = fit.cderi(ch["c_act"][:, blk], ch["c_vir"])
            np.save(stage / f"b{k}.npy", cp.asnumpy(b).astype(dtype, copy=False))
            del b
            rc.free()
        del fit
        rc.free()
        res["n_blocks"] = len(blocks)
        res["seconds_three_index"] = round(time.time() - t, 2)

        def load(k, as32=False):
            arr = np.load(stage / f"b{k}.npy")
            return cp.asarray(arr, dtype=cp.float32 if (as32 or not fp64) else cp.float64)

        # spectra
        t = time.time()
        e_act, e_vir = ch["e_act"], ch["e_vir"]
        d_full = (e_vir[None, :] - e_act[:, None])
        dmin_raw, dmax = float(d_full.min()), float(d_full.max())
        res["min_excitation_hartree"], res["max_excitation_hartree"] = dmin_raw, dmax
        res["nonpositive_excitation"] = bool(dmin_raw <= 0)
        floor = 1e-7
        om, w = rc.frequency_grid(max(dmin_raw, floor), dmax, du, pad)
        sig = np.zeros((len(om), naux))
        per_group = max(int(GROUP_BYTES // (naux * naux * 8)), 1)
        rows_chunk = max(int(0.6e9 / (8 * naux)), 256)
        for g0 in range(0, len(om), per_group):
            group = list(range(g0, min(g0 + per_group, len(om))))
            qs = [cp.zeros((naux, naux), order="F") for _ in group]
            for k, blk in enumerate(blocks):
                b2 = load(k).reshape(-1, naux)
                d = cp.clip(d_full[blk].ravel(), floor, None)
                for q, m in zip(qs, group):
                    gfac = cp.sqrt(4.0 * d / (d * d + om[m] * om[m]))
                    for p0 in range(0, b2.shape[0], rows_chunk):
                        x = b2[p0:p0 + rows_chunk] * gfac[p0:p0 + rows_chunk, None].astype(b2.dtype)
                        if fp64:
                            cublas.syrk("T", x, out=q, alpha=1.0, beta=1.0, lower=True)
                        else:   # FP32 chunk product, FP64 accumulation
                            q += cp.tril(x.T @ x).astype(cp.float64)
                        del x
                del b2
                rc.free()
            for q, m in zip(qs, group):
                sig[m] = cp.asnumpy(cp.linalg.eigvalsh(q, UPLO="L"))
            del qs
            rc.free()
        res["seconds_spectra"] = round(time.time() - t, 2)
        res["n_freq"] = int(len(om))
        res["most_negative_sigma"] = float(sig.min())
        s = np.clip(sig, 0, None)
        res["e_c_rpa"] = float((w[:, None] * (np.log1p(s) - s)).sum() / (2 * np.pi))
        res["e_c_second_order_direct"] = float((w[:, None] * (-0.5 * s * s)).sum() / (2 * np.pi))
        arrays = {"omega": om, "weight": w, "sigma_total": sig}

        # second-order energies from block pairs (FP32 integrals, FP64 sums)
        t = time.time()
        direct = np.zeros(len(rc.PT2_SHIFTS))
        exch = np.zeros(len(rc.PT2_SHIFTS))
        for ka, blk_a in enumerate(blocks):
            ba = load(ka, as32=True)
            for kb in range(ka, len(blocks)):
                bb = ba if kb == ka else load(kb, as32=True)
                dsum, xsum = pt2_block_pair(ba, bb, e_act[blk_a], e_act[blocks[kb]], e_vir, rc.PT2_SHIFTS)
                fac = 1.0 if kb == ka else 2.0
                direct += fac * dsum
                exch += fac * xsum
                if kb != ka:
                    del bb
                rc.free()
            del ba
            rc.free()
        res["pt2"] = {f"{s:g}": {"os": -float(direct[k]), "ss": -float(direct[k] - exch[k])}
                      for k, s in enumerate(rc.PT2_SHIFTS)}
        res["seconds_pt2"] = round(time.time() - t, 2)
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    res["seconds"] = round(time.time() - t0, 2)
    return res, arrays
