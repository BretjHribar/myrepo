"""One-off: the blocked large-structure path must reproduce the standard path on a structure that fits."""
import json
import sys

from .common import BULK, dataset_structures, pin_one_core

pin_one_core()
import numpy as np
from . import corr_large as cl


def main():
    sid = sys.argv[1]
    nblk = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    row = next(r for r in dataset_structures("gmtkn55") if r["id"] == sid)
    scf_dir = BULK / "gmtkn55/pbe_qzvp_g3"
    name = sid.replace("/", "__")
    ref = json.loads((scf_dir / "corr_v1/results" / (name + ".json")).read_text())
    zref = np.load(scf_dir / "corr_v1/spectra" / (name + ".npz"))
    for label, fp64_max in (("FP64 blocked", 10**9), ("FP32 blocked", 0)):
        cl.FP64_MAX_NAO = fp64_max
        res, arr = cl.run_one_large("gmtkn55", row, scf_dir, ref["freq_du"], ref["freq_pad"], 3, force_blocks=nblk)
        print(f"{label}: blocks {res['n_blocks']}  time {res['seconds']}s")
        for key in ("e_x_exact", "e_c_rpa", "e_c_second_order_direct"):
            print(f"   {key:26s} {res[key]:.9f}  standard {ref[key]:.9f}  diff {res[key]-ref[key]:+.2e}")
        for s in ("0", "0.3", "1"):
            for c in ("os", "ss"):
                print(f"   pt2[{s}][{c}] {res['pt2'][s][c]:.9f}  standard {ref['pt2'][s][c]:.9f}  diff {res['pt2'][s][c]-ref['pt2'][s][c]:+.2e}")
            print(f"   singles[{s}] {res['singles'][s]:.9f}  standard {ref['singles'][s]:.9f}  diff {res['singles'][s]-ref['singles'][s]:+.2e}")
        print(f"   max |sigma difference| {np.abs(arr['sigma_total'] - zref['sigma_total']).max():.2e}")


if __name__ == "__main__":
    main()
