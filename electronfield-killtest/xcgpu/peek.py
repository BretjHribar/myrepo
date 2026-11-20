"""Print energy components of chosen structures and simple combinations (file I/O and arithmetic on a few numbers)."""
import json
import sys

from .common import BULK

KCAL = 627.5094740631


def main():
    table = json.loads((BULK / "gmtkn55/pbe_qzvp_g3/table_corr_v1_semi_v1.json").read_text())["rows"]
    spec = sys.argv[1]  # e.g. "W4-11/h2o:-1,W4-11/o:1,W4-11/h:2"
    total = {"pbe": 0.0, "exx": 0.0, "exx_rpa": 0.0, "rpa_c": 0.0, "pt2os0.3": 0.0, "pt2ss0.3": 0.0, "singles0.3": 0.0}
    for item in spec.split(","):
        sid, coef = item.rsplit(":", 1)
        coef = float(coef)
        r = table[sid]
        exx = r["e_pbe"] - r["exc_pbe"] + r["e_x_exact"]
        print(f"{sid:20s} E_PBE {r['e_pbe']:.6f}  E_EXX {exx:.6f}  E_c^RPA {r['e_c_rpa']:.6f}  gap {r['min_gap']}  sigma_max {r.get('sigma_max')}")
        total["pbe"] += coef * r["e_pbe"]
        total["exx"] += coef * exx
        total["exx_rpa"] += coef * (exx + r["e_c_rpa"])
        total["rpa_c"] += coef * r["e_c_rpa"]
        total["pt2os0.3"] += coef * r["pt2"]["0.3"]["os"]
        total["pt2ss0.3"] += coef * r["pt2"]["0.3"]["ss"]
        total["singles0.3"] += coef * r["singles"]["0.3"]
    print({k: round(v * KCAL, 3) for k, v in total.items()}, "kcal/mol")


if __name__ == "__main__":
    main()
