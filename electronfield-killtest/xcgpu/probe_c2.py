"""One-off: find an SCF recipe that converges singlet C2 (near-degenerate frontier orbitals) on the GPU."""
import sys
import time
from .common import dataset_structures, pin_one_core, row_mol
pin_one_core()
import cupy as cp
from gpu4pyscf.dft import rks, uks
from . import run_scf as rs


def attempt(label, build):
    t = time.time()
    try:
        mf = build()
        e = mf.kernel() if not isinstance(mf, tuple) else None
        print(f"{label:45s} conv {mf.converged}  E {float(mf.e_tot):.8f}  cycles {getattr(mf, 'cycles', -1)}  {time.time()-t:.1f}s", flush=True)
        return mf
    except Exception as exc:
        print(f"{label:45s} FAILED {exc!r}"[:300], flush=True)
        return None


def main():
    sid = sys.argv[1] if len(sys.argv) > 1 else "W4-11/c2"
    row = next(r for r in dataset_structures("gmtkn55") if r["id"] == sid)
    mol = row_mol("gmtkn55", row)

    def base(cls=rks.RKS):
        mf = cls(mol, xc="pbe").density_fit(auxbasis=rs.AUX_J)
        mf.grids.level = 3
        mf.conv_tol = 1e-9
        mf.verbose = 0
        return mf

    def shifted(shift, damp, cycles):
        def build():
            mf = base(); mf.level_shift = shift; mf.damp = damp; mf.max_cycle = cycles
            return mf
        return build

    attempt("level shift 1.0, damping 0.5, 400 cycles", shifted(1.0, 0.5, 400))

    def smear_then_restart():
        from gpu4pyscf.scf import smearing
        mf = base(); mf.max_cycle = 200
        mf = smearing.smearing(mf, sigma=0.01, method="fermi")
        mf.kernel()
        print("   smeared: conv", mf.converged, "E", float(mf.e_tot), flush=True)
        dm = mf.make_rdm1()
        mf2 = base(); mf2.max_cycle = 300; mf2.level_shift = 0.3
        mf2.kernel(dm0=dm)
        return mf2
    t = time.time()
    try:
        mf2 = smear_then_restart()
        print(f"{'smearing 0.01 then integer restart':45s} conv {mf2.converged}  E {float(mf2.e_tot):.8f}  {time.time()-t:.1f}s", flush=True)
    except Exception as exc:
        print(f"smearing route FAILED {exc!r}"[:300], flush=True)

    def newton():
        mf = base(); mf.max_cycle = 100
        return mf.newton()
    attempt("second-order (newton)", newton)

    def uks_broken():
        mf = base(uks.UKS); mf.max_cycle = 300; mf.level_shift = 0.2
        return mf
    attempt("unrestricted, spin 0 (default guess)", uks_broken)


if __name__ == "__main__":
    main()
