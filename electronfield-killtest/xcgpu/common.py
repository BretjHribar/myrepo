"""Shared paths, structure loading and molecule construction (bookkeeping only)."""
import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RESEARCH = REPO / ".research"
# Bulk storage lives on the D drive (user instruction, 20 November 2026).
BULK = Path(os.environ.get("XC_BULK", "/mnt/d/xcresearch"))
GMTKN_DATA = RESEARCH / "data/gmtkn55_official_v1/data"
PRED_INPUTS = RESEARCH / "data/gmtkn55_pipeline/prediction_inputs.json"
BASIS = "def2-qzvp"
# def2 effective core potentials exist only beyond Kr; lighter atoms are all-electron.
ECP_ELEMENTS = {"Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
                "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt",
                "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn"}


def pin_one_core():
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "MAX_JOBS"):
        os.environ[key] = "1"
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})


def gmtkn_structures():
    doc = json.loads(PRED_INPUTS.read_text())
    return doc["structures"], doc


def read_xyz(path):
    lines = Path(path).read_text().splitlines()
    n = int(lines[0].split()[0])
    atoms = []
    for line in lines[2:2 + n]:
        sym, x, y, z = line.split()[:4]
        sym = sym[0].upper() + sym[1:].lower()
        atoms.append((sym, (float(x), float(y), float(z))))
    if len(atoms) != n:
        raise ValueError(f"Truncated xyz: {path}")
    return atoms


def build_mol(atoms, charge, unpaired, basis=BASIS, verbose=0, unit="Angstrom"):
    from pyscf import gto
    mol = gto.Mole()
    mol.atom = atoms
    mol.unit = unit
    mol.charge = int(charge)
    mol.spin = int(unpaired)
    mol.basis = basis
    heavy = sorted({sym for sym, _ in atoms if sym in ECP_ELEMENTS})
    if heavy:
        mol.ecp = {sym: basis for sym in heavy}
    mol.verbose = verbose
    mol.cart = False
    mol.build()
    return mol


def dataset_dir(dataset):
    return BULK / dataset


def dataset_structures(dataset):
    """Rows {id, subset, charge, unpaired_electrons, nao} for a dataset, smallest basis first.

    gmtkn55: the pinned public benchmark (geometries read from its xyz files).
    anything else: BULK/<dataset>/structures.json written by a preparation script, atoms inline.
    """
    if dataset == "gmtkn55":
        rows, _ = gmtkn_structures()
        inv = {r["id"]: r for r in json.loads((BULK / "gmtkn55/inventory.json").read_text())}
        for r in rows:
            r["nao"] = inv[r["id"]]["nao"]
        rows.sort(key=lambda r: (r["nao"], r["id"]))
    else:
        # file order is kept: it is a fixed pseudo-random order, so any prefix is an unbiased sample
        rows = json.loads((BULK / dataset / "structures.json").read_text())
    return rows


def row_mol(dataset, row, verbose=0):
    if dataset == "gmtkn55":
        atoms, unit = read_xyz(GMTKN_DATA / row["coordinate_path"]), "Angstrom"
    else:
        atoms, unit = [(a[0], tuple(a[1:4])) for a in row["atoms"]], row.get("unit", "Bohr")
    return build_mol(atoms, row["charge"], row["unpaired_electrons"], verbose=verbose, unit=unit)


def file_stem(structure_id):
    return structure_id.replace("/", "__")
