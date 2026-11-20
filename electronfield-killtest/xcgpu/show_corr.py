"""Print a compact table of response-pass results (file I/O only)."""
import glob
import json
import sys


def main():
    folder = sys.argv[1]
    for f in sorted(glob.glob(folder + "/*.json")):
        r = json.load(open(f))
        if r.get("error"):
            print(r["id"], "ERR", r["error"][:100])
            continue
        p = r.get("pt2", {}).get("0", {})
        print(f"{r['id']:18s} nao {r['nao']:4d} fc {r['n_frozen_orbitals']} act {r['n_active_occ']} "
              f"Ex {r['e_x_exact']:.6f} RPA {r.get('e_c_rpa', 0):.6f} 2nd {r.get('e_c_second_order_direct', 0):.6f} "
              f"os {p.get('os', 0):.6f} ss {p.get('ss', 0):.6f} nf {r.get('n_freq')} "
              f"dmin {r.get('min_excitation_hartree', 0):.4f} dmax {r.get('max_excitation_hartree', 0):.1f} "
              f"t {r['seconds']} (x {r['seconds_exchange']} 3i {r['seconds_three_index']} "
              f"sp {r.get('seconds_spectra')} pt2 {r.get('seconds_pt2')})")


if __name__ == "__main__":
    main()
