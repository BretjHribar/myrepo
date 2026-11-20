"""Basis-size inventory of every required GMTKN55 structure (integer bookkeeping, no numerics)."""
import json
import sys
from .common import BULK, GMTKN_DATA, build_mol, gmtkn_structures, pin_one_core, read_xyz


def main():
    pin_one_core()
    rows, _ = gmtkn_structures()
    out = []
    for row in rows:
        atoms = read_xyz(GMTKN_DATA / row["coordinate_path"])
        mol = build_mol(atoms, row["charge"], row["unpaired_electrons"])
        lmax = max(mol.bas_angular(i) for i in range(mol.nbas))
        out.append({"id": row["id"], "subset": row["subset"], "natm": mol.natm, "nao": int(mol.nao),
                    "nelec": int(mol.nelectron), "charge": row["charge"], "unpaired": row["unpaired_electrons"],
                    "has_ecp": bool(mol.has_ecp()), "lmax": int(lmax),
                    "elements": sorted({a[0] for a in atoms})})
        if mol.nelectron + sum(mol.atom_nelec_core(i) for i in range(mol.natm)) != row["physical_electrons"]:
            raise ValueError(f"Electron count mismatch for {row['id']}")
    dest = BULK / "gmtkn55"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "inventory.json").write_text(json.dumps(out, indent=1))
    naos = sorted(r["nao"] for r in out)
    print("structures", len(out), "nao min/median/max", naos[0], naos[len(naos)//2], naos[-1])
    print("sum nao", sum(naos), "open-shell", sum(r["unpaired"] > 0 for r in out), "ecp", sum(r["has_ecp"] for r in out),
          "lmax", max(r["lmax"] for r in out))
    for lo, hi in ((0, 300), (300, 600), (600, 1000), (1000, 1500), (1500, 2000), (2000, 2500), (2500, 4000)):
        sel = [r for r in out if lo <= r["nao"] < hi]
        print(f"nao [{lo},{hi}): {len(sel)}")
    print("largest:", [(r["id"], r["nao"]) for r in sorted(out, key=lambda r: -r["nao"])[:12]])
    sys.stdout.flush()


if __name__ == "__main__":
    main()
