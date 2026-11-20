"""List structures with the smallest or negative lowest excitation energy (file I/O only)."""
import json
import sys

from .common import BULK


def main():
    table = json.loads((BULK / "gmtkn55/pbe_qzvp_g3/table_corr_v1_semi_v1.json").read_text())["rows"]
    rows = sorted(((r["min_gap"], sid, r.get("sigma_max"), r["s2_dev"]) for sid, r in table.items() if r["min_gap"] is not None))
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    print("structures:", len(table), " negative gap:", sum(g < 0 for g, *_ in rows), " gap < 0.01:", sum(g < 0.01 for g, *_ in rows))
    for gap, sid, smax, s2 in rows[:limit]:
        print(f"{sid:24s} min excitation {gap:+.5f} Ha   sigma_max {smax:12.3f}   <S2> deviation {s2:+.3f}")


if __name__ == "__main__":
    main()
