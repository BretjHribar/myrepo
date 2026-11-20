"""Orbital-dependent ingredients from saved Kohn-Sham orbitals, on the GPU.

For every structure this stores size-extensive scalars and spectra that a learned
functional can be built from without repeating any electronic-structure work:
  * exact (Hartree-Fock type) exchange energy of the KS determinant, density-fitted;
  * second-order (PT2) opposite-/same-spin energies at several denominator shifts;
  * eigenvalue spectra of the RI density-response matrix Q(i omega) on a frequency grid,
    for the total response and (open shells) for each spin channel separately.
Nothing here uses benchmark reference energies.
"""
import argparse
import json
import time
import traceback
from pathlib import Path

from .common import BULK, dataset_dir, dataset_structures, pin_one_core, row_mol

pin_one_core()
import numpy as np
import cupy as cp
from cupy import cublas
from pyscf import df as pdf
from pyscf.data import elements as pel
import gpu4pyscf
from gpu4pyscf.df import int3c2e_bdiv
from gpu4pyscf.mp import dfmp2_addons as ad

AUX_JK = "def2-universal-jkfit"
AUX_RI = "def2-qzvp-ri"
PT2_SHIFTS = (0.0, 0.3, 1.0)
GPU_BUDGET_BYTES = 3.0e9  # largest three-index block kept on the 8 GB card (the SCF pass may share the GPU)


def frozen_orbitals_all_electron(z):
    """Chemical core per atom (spatial orbitals). s-block metals keep their outer semicore active."""
    if z <= 4:
        return 0   # H-Be
    if z <= 10:
        return 1   # B-Ne: 1s
    if z <= 12:
        return 1   # Na, Mg: 1s (2s2p correlated)
    if z <= 18:
        return 5   # Al-Ar: 1s2s2p
    if z <= 20:
        return 5   # K, Ca: 3s3p correlated
    if z <= 30:
        return 9   # Sc-Zn
    if z <= 36:
        return 14  # Ga-Kr: [Ar]3d10
    if z <= 38:
        return 14  # Rb, Sr
    if z <= 48:
        return 18
    if z <= 54:
        return 23  # In-Xe: [Kr]4d10
    if z <= 56:
        return 23
    if z <= 80:
        return 34
    if z <= 86:
        return 39  # Tl-Rn: [Xe]4f14 5d10
    raise ValueError(f"no frozen-core rule for Z={z}")


def n_frozen(mol):
    total = 0
    for i in range(mol.natm):
        z = pel.charge(mol.atom_pure_symbol(i))
        total += max(frozen_orbitals_all_electron(z) - mol.atom_nelec_core(i) // 2, 0)
    return total


def frozen_by_element(mol):
    """Number of frozen spatial orbitals owed by each element present (ECP cores already removed)."""
    need = {}
    for i in range(mol.natm):
        sym = mol.atom_pure_symbol(i)
        z = pel.charge(sym)
        need[sym] = need.get(sym, 0) + max(frozen_orbitals_all_electron(z) - mol.atom_nelec_core(i) // 2, 0)
    return need


def select_frozen(mol, c_occ, e_occ, ovlp):
    """Indices (into the occupied set) of the frozen-core orbitals, chosen by element character.

    Energy ordering alone is wrong when a frozen semicore shell of one element lies above an active
    shell of another (Pb 5d above O 2s, Al 2p above Be 1s). For each element, the frozen orbitals are
    the lowest-energy occupied orbitals whose Mulliken population on that element exceeds one half.
    Returns (sorted indices, True if this differs from plain energy ordering).
    """
    nocc = c_occ.shape[1]
    need = frozen_by_element(mol)
    total = sum(need.values())
    if nocc == 0 or total == 0:
        return np.zeros(0, dtype=int), False
    pop = cp.asnumpy(c_occ * (ovlp @ c_occ))                      # (nao, nocc)
    aoslices = mol.aoslice_by_atom()
    by_el = {}
    for i in range(mol.natm):
        sym = mol.atom_pure_symbol(i)
        a0, a1 = aoslices[i, 2], aoslices[i, 3]
        by_el[sym] = by_el.get(sym, 0) + pop[a0:a1].sum(0)
    e = cp.asnumpy(e_occ)
    chosen = []
    for sym, n in need.items():
        if n == 0:
            continue
        cand = [i for i in np.argsort(e) if by_el[sym][i] > 0.5 and i not in chosen]
        if len(cand) < n:
            raise RuntimeError(f"cannot identify {n} core orbitals on {sym}: only {len(cand)} candidates")
        chosen.extend(cand[:n])
    chosen = np.sort(np.array(chosen[:min(total, nocc)], dtype=int))
    lowest = np.sort(np.argsort(e)[:len(chosen)])
    return chosen, not np.array_equal(chosen, lowest)


def sync():
    cp.cuda.Stream.null.synchronize()


def free():
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()


def aux_basis_dict(mol, name):
    """Named auxiliary basis per element; elements it lacks get a generated even-tempered set (l <= 6)."""
    from pyscf.gto.basis import load
    out, generated = {}, None
    for sym in sorted({mol.atom_pure_symbol(i) for i in range(mol.natm)}):
        try:
            load(name, sym)
            out[sym] = name
        except Exception:
            if generated is None:
                generated = pdf.autoaux(mol)
            out[sym] = [shell for shell in generated[sym] if shell[0] <= 6]
    return out


class Fitter:
    """Half-transformed, Cholesky-decomposed three-index integrals for one auxiliary basis."""

    def __init__(self, mol, auxbasis):
        self.mol = mol
        self.aux = pdf.addons.make_auxmol(mol, aux_basis_dict(mol, auxbasis))
        self.intopt = int3c2e_bdiv.Int3c2eOpt(mol, self.aux).build()
        self.dec = ad.get_j2c_decomp_gpu(self.aux, j2c=ad.get_j2c_bdiv(self.intopt), alg="cd")

    def bytes_needed(self, nleft, nright):
        return nleft * nright * self.aux.nao_cart() * 8

    def cderi(self, cl, cr):
        buf = cp.empty([cl.shape[1], cr.shape[1], self.aux.nao_cart()], dtype=np.float64)
        b = ad.get_j3c_ovl_gpu_bdiv(self.mol, self.intopt, [cl], [cr], [buf])[0]
        return ad.decompose_j3c_gpu(self.mol, self.dec, [b])[0]


def pbe_potential(mol, z, restricted, grid_level):
    """PBE exchange-correlation potential matrix and energy at the saved density (one grid pass)."""
    from gpu4pyscf.dft import rks, uks
    mf = (rks.RKS if restricted else uks.UKS)(mol, xc="pbe")
    mf.grids.level = grid_level
    mf.grids.build()
    c, o = cp.asarray(z["mo_coeff"]), cp.asarray(z["mo_occ"])
    if restricted:
        dm = (c * o) @ c.T
        _, exc, vxc = mf._numint.nr_rks(mol, mf.grids, "pbe", dm)
    else:
        dm = cp.stack([(c[s] * o[s]) @ c[s].T for s in (0, 1)])
        _, exc, vxc = mf._numint.nr_uks(mol, mf.grids, "pbe", dm)
    return cp.asarray(vxc), float(exc)


def frequency_grid(dmin, dmax, du, pad):
    """Trapezoid in ln(omega) from dmin/pad to dmax*pad with analytic end corrections.

    Below the lowest node the integrand is flat (weight omega); above the highest it decays as
    omega^-4 (weight omega/3). The du^2 and du^4 terms are the Euler-Maclaurin end corrections of the
    trapezoid rule for those two limiting forms; without them the error is O(du^2 / pad).
    """
    lo, hi = dmin / pad, dmax * pad
    m = int(np.ceil(np.log(hi / lo) / du))
    om = lo * np.exp(du * np.arange(m + 1))
    w = om * du
    w[0] = om[0] * (1.0 + 0.5 * du + du ** 2 / 12 - du ** 4 / 720)
    w[-1] = om[-1] * (1.0 / 3 + 0.5 * du + du ** 2 / 4 - 27 * du ** 4 / 720)
    return om, w


def channels(z, unpaired, mol):
    """Per-spin occupied (all / active) and virtual orbitals with their energies."""
    out = []
    spins = (0, 1) if unpaired else (None,)
    ovlp = cp.asarray(mol.intor("int1e_ovlp"))
    for s in spins:
        c = z["mo_coeff"] if s is None else z["mo_coeff"][s]
        e = z["mo_energy"] if s is None else z["mo_energy"][s]
        o = z["mo_occ"] if s is None else z["mo_occ"][s]
        occ = np.where(o > 0)[0]
        vir = np.where(o == 0)[0]
        frozen, differs = select_frozen(mol, cp.asarray(c[:, occ]), cp.asarray(e[occ]), ovlp)
        act = np.delete(occ, frozen)
        out.append({"core_differs_from_energy_order": bool(differs), "n_frozen": int(len(frozen)),
                    "c_occ": cp.asarray(c[:, occ]), "c_act": cp.asarray(c[:, act]), "c_vir": cp.asarray(c[:, vir]),
                    "e_act": cp.asarray(e[act]), "e_vir": cp.asarray(e[vir]),
                    "e_occ_all": cp.asarray(e[occ]), "e_vir_all": cp.asarray(e[vir])})
    return out


def response_matrix(b2d, d, omega, prefactor, rows_per_chunk):
    """Q = prefactor * B^T diag(d/(d^2+omega^2)) B, lower triangle, FP64."""
    naux = b2d.shape[1]
    q = cp.zeros((naux, naux), order="F")
    g = cp.sqrt(prefactor * d / (d * d + omega * omega))
    for p0 in range(0, b2d.shape[0], rows_per_chunk):
        x = b2d[p0:p0 + rows_per_chunk] * g[p0:p0 + rows_per_chunk, None]
        cublas.syrk("T", x, out=q, alpha=1.0, beta=1.0, lower=True)
    return q


def pt2_energies(mol, bs, chans, restricted):
    out = {}
    b32 = []
    while bs:                      # convert one spin channel at a time and release its FP64 copy
        b = bs.pop(0)
        b32.append(b.astype(np.float32))
        del b
        cp.get_default_memory_pool().free_all_blocks()
    for shift in PT2_SHIFTS:
        ev = [ch["e_vir"] + 0.5 * shift for ch in chans]
        eo = [ch["e_act"] for ch in chans]
        if restricted:
            if b32[0].shape[0] == 0:
                os_, ss = 0.0, 0.0
            else:
                r = ad.get_dfmp2_energy_pair_intra(mol, b32[0], eo[0], ev[0])
                os_ = float(r[0].sum())
                ss = float(r[0].sum() - r[1].sum())
        else:
            ss = 0.0
            for s in (0, 1):
                if b32[s].shape[0] > 1:
                    ss += 0.25 * float(ad.get_dfmp2_energy_pair_intra(mol, b32[s], eo[s], ev[s], ss_only=True).sum())
            os_ = 0.0
            if b32[0].shape[0] and b32[1].shape[0]:
                os_ = float(ad.get_dfump2_energy_pair_intra(mol, b32, eo, ev).sum())
        out[f"{shift:g}"] = {"os": os_, "ss": ss}
    return out


def run_one(dataset, row, scf_dir, du, pad, grid_level):
    t0 = time.time()
    sid = row["id"]
    mol = row_mol(dataset, row)
    z = np.load(scf_dir / "orbitals" / (sid.replace("/", "__") + ".npz"))
    restricted = z["mo_coeff"].ndim == 2   # the SCF pass may have used the unrestricted fallback
    nfrozen = n_frozen(mol)
    chans = channels(z, not restricted, mol)
    res = {"id": sid, "nao": int(mol.nao), "n_frozen_orbitals": nfrozen, "aux_jk": AUX_JK, "aux_ri": AUX_RI,
           "core_selection": "element population",
           "core_differs_from_energy_order": any(c["core_differs_from_energy_order"] for c in chans),
           "restricted": restricted, "n_active_occ": [int(c["c_act"].shape[1]) for c in chans],
           "n_vir": [int(c["c_vir"].shape[1]) for c in chans], "pt2_shifts": list(PT2_SHIFTS),
           "freq_du": du, "freq_pad": pad}
    arrays = {}

    # exact exchange of the KS determinant (all occupied orbitals, no frozen core), and the
    # occupied-virtual block of the exact-exchange Fock operator for single-excitation terms
    t = time.time()
    vxc, exc_check = pbe_potential(mol, z, restricted, grid_level)
    res["e_xc_recomputed"] = exc_check
    fit = Fitter(mol, AUX_JK)
    ex = 0.0
    singles = {f"{shift:g}": 0.0 for shift in PT2_SHIFTS}
    fnorm2 = 0.0
    singles_done = True
    for s_idx, ch in enumerate(chans):
        n = ch["c_occ"].shape[1]
        if n == 0:
            continue
        if fit.bytes_needed(n, n) > GPU_BUDGET_BYTES:
            raise MemoryError("exchange three-index block exceeds the GPU budget")
        bij = fit.cderi(ch["c_occ"], ch["c_occ"])
        ex -= (1.0 if restricted else 0.5) * float((bij * bij).sum())
        nv = ch["c_vir"].shape[1]
        if nv and fit.bytes_needed(n, nv) <= GPU_BUDGET_BYTES:
            bja = fit.cderi(ch["c_occ"], ch["c_vir"])
            k_ov = cp.einsum("ijp,jap->ia", bij, bja)
            del bja
            v = vxc if restricted else vxc[s_idx]
            f_ov = -k_ov - ch["c_occ"].T @ v @ ch["c_vir"]
            d = ch["e_vir_all"][None, :] - ch["e_occ_all"][:, None]
            fac = 2.0 if restricted else 1.0
            fnorm2 += fac * float((f_ov * f_ov).sum())
            for shift in PT2_SHIFTS:
                singles[f"{shift:g}"] -= fac * float((f_ov * f_ov / (d + shift)).sum())
        else:
            singles_done = False
        del bij
    res["e_x_exact"] = ex
    res["singles"] = singles if singles_done else None
    res["fock_ov_norm2"] = fnorm2 if singles_done else None
    res["seconds_exchange"] = round(time.time() - t, 2)
    del fit, vxc
    free()

    # RI response
    t = time.time()
    fit = Fitter(mol, AUX_RI)
    naux = fit.aux.nao
    res["naux_ri"] = int(naux)
    for ch in chans:
        if fit.bytes_needed(ch["c_act"].shape[1], ch["c_vir"].shape[1]) > GPU_BUDGET_BYTES:
            raise MemoryError("response three-index block exceeds the GPU budget (large-structure path not built yet)")
    bs = []
    for ch in chans:
        if ch["c_act"].shape[1] == 0:
            bs.append(cp.zeros((0, ch["c_vir"].shape[1], naux)))
        else:
            bs.append(fit.cderi(ch["c_act"], ch["c_vir"]))
    del fit
    free()
    sync()
    res["seconds_three_index"] = round(time.time() - t, 2)
    ds = [(ch["e_vir"][None, :] - ch["e_act"][:, None]).ravel() for ch in chans]
    alld = cp.concatenate(ds) if sum(d.size for d in ds) else cp.zeros(0)
    if alld.size == 0:
        res["no_active_pairs"] = True
        arrays.update(omega=np.zeros(0), weight=np.zeros(0), sigma_total=np.zeros((0, naux)))
    else:
        dmin_raw, dmax = float(alld.min()), float(alld.max())
        res["min_excitation_hartree"], res["max_excitation_hartree"] = dmin_raw, dmax
        res["nonpositive_excitation"] = bool(dmin_raw <= 0)
        floor = 1e-7  # only guards exact degeneracy; near-degenerate pairs keep their true energies
        ds = [cp.clip(d, floor, None) for d in ds]
        om, w = frequency_grid(max(dmin_raw, floor), dmax, du, pad)
        t = time.time()
        pref = 4.0 if restricted else 2.0
        b2 = [b.reshape(-1, naux) for b in bs]
        rows = max(int(1.2e9 / (8 * naux)), 256)
        sig_t = np.zeros((len(om), naux))
        sig_s = None if restricted else np.zeros((2, len(om), naux))
        for k, o in enumerate(om):
            qs = [response_matrix(b2[s], ds[s], o, pref, rows) if b2[s].shape[0] else None for s in range(len(chans))]
            if restricted:
                sig_t[k] = cp.asnumpy(cp.linalg.eigvalsh(qs[0], UPLO="L"))
            else:
                for s in (0, 1):
                    if qs[s] is not None:
                        sig_s[s, k] = cp.asnumpy(cp.linalg.eigvalsh(qs[s], UPLO="L"))
                live = [q for q in qs if q is not None]
                qt = live[0] if len(live) == 1 else live[0] + live[1]
                sig_t[k] = cp.asnumpy(cp.linalg.eigvalsh(qt, UPLO="L"))
            del qs
        sync()
        res["seconds_spectra"] = round(time.time() - t, 2)
        res["n_freq"] = int(len(om))
        res["most_negative_sigma"] = float(sig_t.min())
        s = np.clip(sig_t, 0, None)
        res["e_c_rpa"] = float((w[:, None] * (np.log1p(s) - s)).sum() / (2 * np.pi))
        res["e_c_second_order_direct"] = float((w[:, None] * (-0.5 * s * s)).sum() / (2 * np.pi))
        arrays.update(omega=om, weight=w, sigma_total=sig_t)
        if sig_s is not None:
            arrays["sigma_spin"] = sig_s
        t = time.time()
        del b2
        res["pt2"] = pt2_energies(mol, bs, chans, restricted)   # consumes and frees the FP64 blocks
        res["seconds_pt2"] = round(time.time() - t, 2)
    del bs
    free()
    res["seconds"] = round(time.time() - t0, 2)
    return res, arrays


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="gmtkn55")
    ap.add_argument("--scf-tag", required=True)
    ap.add_argument("--tag", default="corr_v1")
    ap.add_argument("--limit", type=int, default=0, help="use only the first N rows of the dataset order")
    ap.add_argument("--ids", default="")
    ap.add_argument("--max-nao", type=int, default=10**9)
    ap.add_argument("--du", type=float, default=0.6)
    ap.add_argument("--pad", type=float, default=40.0)
    ap.add_argument("--follow", action="store_true", help="keep scanning until every listed structure has an SCF result")
    ap.add_argument("--shard", default="", help="i/n: process only rows whose position in the size-ordered list is i modulo n")
    ap.add_argument("--min-nao", type=int, default=0)
    ap.add_argument("--recheck-core", action="store_true",
                    help="for results made with energy-ordered cores: keep them if the element-based choice is identical, else archive and redo")
    args = ap.parse_args()
    scf_dir = dataset_dir(args.dataset) / args.scf_tag
    out = scf_dir / args.tag
    (out / "results").mkdir(parents=True, exist_ok=True)
    (out / "spectra").mkdir(parents=True, exist_ok=True)
    rows = dataset_structures(args.dataset)
    if args.limit:
        rows = rows[:args.limit]
    if args.ids:
        wanted = set(args.ids.split(","))
        rows = [r for r in rows if r["id"] in wanted]
    rows = [r for r in rows if args.min_nao <= r["nao"] <= args.max_nao]
    if args.shard:
        i, n = (int(x) for x in args.shard.split("/"))
        rows = rows[i::n]
    card = {"tag": args.tag, "scf_tag": args.scf_tag, "aux_jk": AUX_JK, "aux_ri": AUX_RI, "pt2_shifts": PT2_SHIFTS,
            "frequency_grid": {"kind": "trapezoid in ln(omega) with end corrections", "du": args.du, "pad": args.pad},
            "frozen_core": "chemical core; s-block metals keep outer semicore active",
            "engine": f"gpu4pyscf {gpu4pyscf.__version__} three-index integrals + cupy linear algebra (CUDA, FP64 response)"}
    (out / "run_card.json").write_text(json.dumps(card, indent=1))
    if args.recheck_core:
        kept = redo = 0
        arch = scf_dir / "superseded"
        arch.mkdir(exist_ok=True)
        for row in rows:
            name = row["id"].replace("/", "__")
            dest = out / "results" / (name + ".json")
            if not dest.exists():
                continue
            old = json.loads(dest.read_text())
            if old.get("error") or old.get("core_selection") == "element population":
                continue
            mol = row_mol(args.dataset, row)
            z = np.load(scf_dir / "orbitals" / (name + ".npz"))
            differs = any(c["core_differs_from_energy_order"] for c in channels(z, z["mo_coeff"].ndim == 3, mol))
            free()
            if differs:
                dest.replace(arch / (name + f".{args.tag}.energy_ordered_core.json"))
                redo += 1
            else:
                old["core_selection"] = "element population"
                old["core_differs_from_energy_order"] = False
                old["core_rechecked"] = True
                dest.write_text(json.dumps(old, indent=1))
                kept += 1
        print(json.dumps({"core_recheck": {"kept_identical": kept, "archived_for_redo": redo}}), flush=True)
    t_start = time.time()
    new = 0
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
                continue  # SCF not done yet; picked up on the next scan
            scf = json.loads(scf_json.read_text())
            if not scf.get("converged"):
                res, arrays = {"id": row["id"], "error": "SCF input not converged"}, None
            else:
                try:
                    try:
                        res, arrays = run_one(args.dataset, row, scf_dir, args.du, args.pad, scf.get("grid_level", 3))
                    except MemoryError as too_big:
                        if "GPU budget" not in str(too_big):
                            raise
                        free()
                        from . import corr_large   # blocked path for structures that do not fit on the GPU
                        res, arrays = corr_large.run_one_large(args.dataset, row, scf_dir, args.du, args.pad,
                                                               scf.get("grid_level", 3))
                    res["e_xc_scf"] = scf["scf_summary"].get("exc")
                    res["e_xc_mismatch"] = abs(res["e_xc_recomputed"] - res["e_xc_scf"])
                except Exception as exc:
                    res, arrays = {"id": row["id"], "error": repr(exc)[:400], "traceback": traceback.format_exc()[-1200:],
                                   "nao": row["nao"]}, None
                    free()
            if arrays is not None:
                tmp = out / "spectra" / (name + ".tmp.npz")
                np.savez(tmp, **arrays)
                tmp.replace(out / "spectra" / (name + ".npz"))
            tmp = dest.with_suffix(".tmp")
            tmp.write_text(json.dumps(res, indent=1))
            tmp.replace(dest)
            new += 1
            print(json.dumps({"i": i + 1, "n": len(rows), "id": row["id"], "nao": res.get("nao"), "ex": res.get("e_x_exact"),
                              "xc_mismatch": res.get("e_xc_mismatch"), "singles": (res.get("singles") or {}).get("0"),
                              "rpa": res.get("e_c_rpa"), "nfreq": res.get("n_freq"), "s": res.get("seconds"),
                              "err": res.get("error"), "elapsed_min": round((time.time() - t_start) / 60, 1)}), flush=True)
        if not waiting or not args.follow:
            break
        time.sleep(45)
    print(json.dumps({"finished": True, "new": new}), flush=True)


if __name__ == "__main__":
    main()
