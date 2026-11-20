"""One-off: where does host time go for a small structure in the response pass and the SCF pass?"""
import cProfile
import json
import pstats
import sys

from .common import BULK, dataset_structures, pin_one_core
pin_one_core()
from . import run_corr as rc
from . import run_scf as rs


def main():
    sid, which = sys.argv[1], sys.argv[2]
    row = next(r for r in dataset_structures("gmtkn55") if r["id"] == sid)
    scf_dir = BULK / "gmtkn55" / "pbe_qzvp_g3"
    if which == "corr":
        rc.run_one("gmtkn55", row, scf_dir, 0.6, 40.0, 3)   # warm-up: kernel compilation, caches
        fn = lambda: rc.run_one("gmtkn55", row, scf_dir, 0.6, 40.0, 3)
    else:
        rs.run_one("gmtkn55", row, BULK / "gmtkn55" / "profile_tmp", "pbe", 3, 1e-9, False)
        fn = lambda: rs.run_one("gmtkn55", row, BULK / "gmtkn55" / "profile_tmp", "pbe", 3, 1e-9, False)
    pr = cProfile.Profile()
    pr.enable()
    fn()
    pr.disable()
    st = pstats.Stats(pr)
    st.sort_stats("cumulative")
    st.print_stats(45)


if __name__ == "__main__":
    main()
