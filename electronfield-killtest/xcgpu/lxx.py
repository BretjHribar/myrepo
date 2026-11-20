"""Exact-exchange energy density on grid points, kept entirely on the GPU.

e_x^s(r) = -1/2 sum_{mu nu} F_mu(r) A_{mu nu}(r) F_nu(r),   F(r) = D_s phi(r),
A_{mu nu}(r) = integral chi_mu(r') chi_nu(r') / |r - r'| dr'   (optionally erf-attenuated).

The package routine for A copies every batch to pinned host memory and back; here the same GPU
integral kernel is called per angular-momentum block and each block is contracted with F at once.
"""
import ctypes

import numpy as np
import cupy as cp
from gpu4pyscf.gto import int3c1e
from gpu4pyscf.lib.cupy_helper import cart2sph

libgint = int3c1e.libgint
NULL = ctypes.c_void_p(0)


class LocalExchange:
    def __init__(self, mol, direct_scf_tol=1e-13):
        self.mol = mol
        self.opt = int3c1e.VHFOpt(mol)
        self.opt.build(direct_scf_tol, aosym=True)

    def sort(self, f):
        """(points, nao) in the molecule's AO order -> the kernel's sorted AO order."""
        return self.opt.sort_orbitals(f, axis=[1])

    def density(self, coords, f_sorted_list, exponent=None):
        """Returns one array (points,) per entry of f_sorted_list: sum_{mu nu} F_mu A_{mu nu} F_nu.

        exponent: None for the full Coulomb kernel, or a Gaussian-charge exponent a, which gives the
        long-range kernel erf(sqrt(a) r)/r.
        """
        opt, mol = self.opt, self.mol
        coords = cp.ascontiguousarray(cp.asarray(coords))
        n = coords.shape[0]
        out = [cp.zeros(n) for _ in f_sorted_list]
        expo = None if exponent is None else cp.full(n, float(exponent))
        stream = cp.cuda.get_current_stream()
        for cp_ij_id, log_q in enumerate(opt.log_qs):
            if len(log_q) == 0:
                continue
            cpi, cpj = opt.cp_idx[cp_ij_id], opt.cp_jdx[cp_ij_id]
            li, lj = opt.angular[cpi], opt.angular[cpj]
            i0, i1 = opt.cart_ao_loc[cpi], opt.cart_ao_loc[cpi + 1]
            j0, j1 = opt.cart_ao_loc[cpj], opt.cart_ao_loc[cpj + 1]
            ni, nj = i1 - i0, j1 - j0
            block = cp.zeros([n, nj, ni], order="C")
            bins = np.array([0, len(log_q)], dtype=np.int32)
            offsets = np.array([i0, j0], dtype=np.int32)
            strides = np.array([ni, ni * nj], dtype=np.int32)
            err = libgint.GINTfill_int3c1e(
                ctypes.cast(stream.ptr, ctypes.c_void_p), opt.bpcache,
                ctypes.cast(coords.data.ptr, ctypes.c_void_p),
                NULL if expo is None else ctypes.cast(expo.data.ptr, ctypes.c_void_p),
                ctypes.c_int(n), ctypes.cast(block.data.ptr, ctypes.c_void_p),
                strides.ctypes.data_as(ctypes.c_void_p), offsets.ctypes.data_as(ctypes.c_void_p),
                bins.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(1), ctypes.c_int(cp_ij_id),
                ctypes.c_double(mol.omega))
            if err != 0:
                raise RuntimeError("GINTfill_int3c1e failed")
            if not mol.cart:
                block = cart2sph(block, axis=1, ang=lj)
                block = cart2sph(block, axis=2, ang=li)
            a0, a1 = opt.ao_loc[cpi], opt.ao_loc[cpi + 1]
            b0, b1 = opt.ao_loc[cpj], opt.ao_loc[cpj + 1]
            if cpi == cpj:
                # same shell group: the kernel fills one triangle of the square block; mirror it
                upper = cp.triu(block)
                block = upper + cp.triu(block, 1).transpose(0, 2, 1)
                weight = 1.0
            else:
                weight = 2.0
            for k, f in enumerate(f_sorted_list):
                tmp = cp.einsum("gji,gi->gj", block, f[:, a0:a1])
                out[k] += weight * (tmp * f[:, b0:b1]).sum(1)
            del block
        return out
