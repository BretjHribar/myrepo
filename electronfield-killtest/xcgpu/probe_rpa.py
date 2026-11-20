"""Prototype check: exact exchange, PT2 and RI response spectra from saved PBE orbitals (GPU)."""
import json, sys, time
import numpy as np
from .common import BULK, GMTKN_DATA, build_mol, gmtkn_structures, pin_one_core, read_xyz
pin_one_core()
import cupy as cp
from pyscf import df as pdf
import gpu4pyscf
from gpu4pyscf.df import int3c2e_bdiv
from gpu4pyscf.mp import dfmp2_addons as ad

def sync(): cp.cuda.Stream.null.synchronize()

def cderi(mol, aux, cl, cr, dtype=np.float64):
    intopt = int3c2e_bdiv.Int3c2eOpt(mol, aux).build()
    j2c = ad.get_j2c_bdiv(intopt)
    dec = ad.get_j2c_decomp_gpu(aux, j2c=j2c, alg="cd")
    buf = cp.empty([cl.shape[1], cr.shape[1], aux.nao_cart()], dtype=dtype)
    b = ad.get_j3c_ovl_gpu_bdiv(mol, intopt, [cl], [cr], [buf])[0]
    return ad.decompose_j3c_gpu(mol, dec, [b])[0]

def main():
    tag, sid = sys.argv[1], sys.argv[2]
    row = {r["id"]: r for r in gmtkn_structures()[0]}[sid]
    mol = build_mol(read_xyz(GMTKN_DATA / row["coordinate_path"]), row["charge"], row["unpaired_electrons"])
    mol.verbose = 0
    z = np.load(BULK / "gmtkn55" / tag / "orbitals" / (sid.replace("/", "__") + ".npz"))
    C, occ, eps = cp.asarray(z["mo_coeff"]), z["mo_occ"], cp.asarray(z["mo_energy"])
    nocc = int((occ > 0).sum()); ncore = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    print("nao", mol.nao, "nocc", nocc, "ncore", ncore, flush=True)
    t = time.time()
    auxjk = pdf.addons.make_auxmol(mol, "def2-universal-jkfit")
    bij = cderi(mol, auxjk, C[:, :nocc], C[:, :nocc]); sync()
    ex = -float((bij ** 2).sum())
    print("E_x^HF (DF, RKS)", ex, "naux_jk", auxjk.nao, "%.1fs" % (time.time() - t), flush=True)
    del bij
    t = time.time()
    auxri = pdf.addons.make_auxmol(mol, "def2-qzvp-ri")
    print("ri aux nao", auxri.nao, "lmax", max(auxri.bas_angular(i) for i in range(auxri.nbas)), flush=True)
    co, cv = C[:, ncore:nocc], C[:, nocc:]
    b = cderi(mol, auxri, co, cv); sync()
    no, nv, naux = b.shape
    print("B build %.1fs" % (time.time() - t), b.shape, flush=True)
    eo, ev = eps[ncore:nocc], eps[nocc:]
    d = (ev[None, :] - eo[:, None]).ravel()
    B = b.reshape(no * nv, naux)
    # reference second-order terms
    t = time.time()
    r = ad.get_dfmp2_energy_pair_intra(mol, b, eo, ev)
    e_os = float(r[0].sum()); e_ss = float(r[0].sum() - r[1].sum())
    print("MP2@PBE os", e_os, "ss", e_ss, "%.1fs" % (time.time() - t), flush=True)
    # frequency quadrature, Gauss-Legendre mapped to [0, inf)
    for nfreq in (16, 32):
        t = time.time()
        x, w = np.polynomial.legendre.leggauss(nfreq)
        w0 = 0.5
        om = w0 * (1 + x) / (1 - x); wt = w * 2 * w0 / (1 - x) ** 2
        erpa = 0.0; e2 = 0.0; spectra = []
        for o, ww in zip(om, wt):
            g = d / (d * d + o * o)
            Q = 4.0 * (B.T * g) @ B
            s = cp.linalg.eigvalsh(Q)
            s = cp.clip(s, 0, None)
            erpa += ww * float((cp.log1p(s) - s).sum())
            e2 += ww * float((-0.5 * s * s).sum())
            spectra.append(cp.asnumpy(s))
        sync()
        erpa /= 2 * np.pi; e2 /= 2 * np.pi
        print(f"nfreq {nfreq}: E_c^RPA {erpa:.8f}  second-order {e2:.8f} (expect 2*os = {2*e_os:.8f})  %.1fs" % (time.time() - t), flush=True)

if __name__ == "__main__":
    main()
