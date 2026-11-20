"""Train on public reference data outside GMTKN55, test on GMTKN55.

Training pool (never the benchmark's own reactions):
  * ACCDB reactions not overlapping GMTKN55 (see prep_accdb for the overlap rules);
  * MSR-ACC/TAE25 atomization energies.
Reported for each model:
  A. outside-only fit -> error on every available GMTKN55 reaction (a true out-of-benchmark number);
  B. GMTKN55 cross-validation with the outside pool always in the training set;
  C. the benchmark-only cross-validation, for reference.
Benchmark numbers are weighted mean absolute errors over the available reactions (WTMAD-2 weights);
they equal WTMAD-2 only at 1505/1505.
"""
import argparse
import json
from pathlib import Path

from .common import BULK, pin_one_core

pin_one_core()
import torch
from . import fitlib as fl
from .fit import LADDER

NORMALISATION = 57.8167
PBE_SANITY_KCAL = 100.0   # a plain-PBE reaction error beyond this means the SCF found a wrong state
DROPPED = {}


def same_species_spin_change(reaction, structures):
    """True for a spin-state or excitation energy: one species, same charge, different multiplicity.

    Such entries (atomic and molecular excitation sets) are not ground-state reaction energies and are
    left out of training by rule, independent of how any model performs on them.
    """
    if len(reaction["structures"]) != 2 or any(a["dataset"] != "accdb" for a in reaction["structures"]):
        return False
    a, b = (structures[x["id"]] for x in reaction["structures"])
    if a["charge"] != b["charge"] or a["unpaired_electrons"] == b["unpaired_electrons"]:
        return False

    def key(st):
        return sorted((x[0], round(x[1], 3), round(x[2], 3), round(x[3], 3)) for x in st["atoms"])
    return key(a) == key(b)


def outside_pool(keys, base, tables, strict, use_tae, use_accdb):
    """Design matrix, targets, weights and tags for every usable outside reaction."""
    A, y, w, tag = [], [], [], []
    if use_accdb and (BULK / "accdb/reactions.json").exists():
        rx = json.loads((BULK / "accdb/reactions.json").read_text())
        structs = {s_["id"]: s_ for s_ in json.loads((BULK / "accdb/structures.json").read_text())}
        groups = {}
        DROPPED.clear()
        DROPPED.update({"spin_state_or_excitation": 0, "pbe_sanity": []})
        for r in rx:
            if not r["complete"] or r["overlaps_benchmark"] or (strict and r["shares_benchmark_structure"]):
                continue
            ids = [(a["dataset"], a["id"]) for a in r["structures"]]
            if not all(i in tables[d] for d, i in ids):
                continue
            if same_species_spin_change(r, structs):
                DROPPED["spin_state_or_excitation"] += 1
                continue
            pbe = sum(c * tables[d][i]["e_pbe"] for c, (d, i) in zip(r["coefficients"], ids)) * fl.KCAL
            if abs(pbe - r["reference_kcal"]) > PBE_SANITY_KCAL:
                DROPPED["pbe_sanity"].append((r["id"], round(pbe - r["reference_kcal"], 1)))
                continue
            groups.setdefault(r["database"] + "/" + r["subset"], []).append((r, ids))
        for name, items in groups.items():
            mean_abs = sum(abs(r["reference_kcal"]) for r, _ in items) / len(items)
            weight = min(max(NORMALISATION / max(mean_abs, 1e-6), 0.05), 100.0)   # the benchmark's own weighting rule
            for r, ids in items:
                coef = torch.tensor(r["coefficients"], dtype=torch.float64, device=fl.DEV)
                e0 = torch.stack([torch.tensor(fl.base_energy(tables[d][i], base), dtype=torch.float64, device=fl.DEV) for d, i in ids])
                f = torch.tensor([[fl.feature_value(tables[d][i], k) for k in keys] for d, i in ids],
                                 dtype=torch.float64, device=fl.DEV).reshape(len(ids), len(keys))
                a_row = (coef @ f) * fl.KCAL
                y_row = r["reference_kcal"] - float(coef @ e0) * fl.KCAL
                if torch.isfinite(a_row).all() and y_row == y_row:
                    A.append(a_row); y.append(y_row); w.append(weight); tag.append(name)
    if use_tae and (BULK / "tae25/labels.json").exists():
        labels = json.loads((BULK / "tae25/labels.json").read_text())
        t = tables["tae25"]
        mols = sorted(m for m in labels if m in t and all(f"atom/{a}" in t for a in labels[m]["atoms"]))
        if mols:
            mean_abs = sum(labels[m]["tae_hartree"] for m in mols) / len(mols) * fl.KCAL
            weight = NORMALISATION / mean_abs
            for m in mols:
                ids = [m] + [f"atom/{a}" for a in labels[m]["atoms"]]
                coef = torch.tensor([-1.0] + [float(c) for c in labels[m]["atoms"].values()], dtype=torch.float64, device=fl.DEV)
                e0, f = fl.structure_energy_terms(t, ids, keys, base)
                a_row = (coef @ f) * fl.KCAL
                y_row = labels[m]["tae_hartree"] * fl.KCAL - float(coef @ e0) * fl.KCAL
                if torch.isfinite(a_row).all() and y_row == y_row:
                    A.append(a_row); y.append(y_row); w.append(weight); tag.append("TAE25")
    if not A:
        return None
    return (torch.stack(A).reshape(len(A), len(keys)), torch.tensor(y, dtype=torch.float64, device=fl.DEV),
            torch.tensor(w, dtype=torch.float64, device=fl.DEV), tag)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", default="K:,N:,P:")
    ap.add_argument("--lams", default="1e-6,1e-5,1e-4,1e-3")
    ap.add_argument("--strict", action="store_true", help="drop outside reactions that share any larger structure with the benchmark")
    ap.add_argument("--no-tae", action="store_true")
    ap.add_argument("--no-accdb", action="store_true")
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    tables = {"gmtkn55": fl.structure_table()[0]}
    for name in ("accdb", "tae25"):
        tables[name] = fl.structure_table(dataset=name)[0] if (BULK / name / "pbe_qzvp_g3" / "results").exists() else {}
    print("structures with every ingredient:", {k: len(v) for k, v in tables.items()})
    report = {"strict": args.strict, "models": []}
    lams = [float(x) for x in args.lams.split(",")]
    for label, base, blocks in LADDER:
        keys = [k for b in blocks for k in fl.FEATURE_SETS[b]]
        if not keys or not any(label.startswith(x) for x in args.models.split(",")):
            continue
        prob = fl.Problem(tables["gmtkn55"], keys, base)
        pool = outside_pool(keys, base, tables, args.strict, not args.no_tae, not args.no_accdb)
        if pool is None:
            print("no outside reactions available yet")
            return
        Ao, yo, wo, tag = pool
        counts = {}
        for t in tag:
            counts[t] = counts.get(t, 0) + 1
        print(f"\n{label}  (p={len(keys)})   benchmark reactions available {len(prob.y)}/{prob.n_total}; "
              f"outside reactions {len(yo)} in {len(counts)} sets" + ("  [strict]" if args.strict else ""))
        # regularisation chosen without looking at the benchmark: leave-one-set-out (or 5-fold when there
        # are few sets) cross-validation inside the outside pool
        if DROPPED and "dropped" not in report:
            report["dropped"] = dict(DROPPED)
            print(f"   outside reactions left out by rule: {DROPPED['spin_state_or_excitation']} spin-state/excitation entries; "
                  f"{len(DROPPED['pbe_sanity'])} with a plain-PBE error above {PBE_SANITY_KCAL:.0f} kcal/mol (wrong SCF state suspected): "
                  + ", ".join(f"{i} ({e:+.0f})" for i, e in DROPPED["pbe_sanity"][:10]))
        names = sorted(counts)
        idx = torch.tensor([names.index(t) for t in tag], device=fl.DEV)
        if len(names) >= 5:
            folds = [idx != i for i in range(len(names))]
        else:
            g = torch.Generator(device="cpu").manual_seed(0)
            f5 = (torch.randperm(len(yo), generator=g) % 5).to(fl.DEV)
            folds = [f5 != k for k in range(5)]
        fold_masks_out = torch.stack(folds)
        best = None
        chosen = None
        for lam in lams:
            tho = fl.solve(Ao, yo, wo, lam, fold_masks_out)
            eo = tho @ Ao.T - yo[None, :]
            held = torch.zeros_like(yo)
            for j in range(fold_masks_out.shape[0]):
                held[~fold_masks_out[j]] = eo[j][~fold_masks_out[j]]
            outside_cv = float((wo * held.abs()).mean())
            th_out = fl.solve(Ao, yo, wo, lam)[0]
            err_out = prob.A @ th_out - prob.y
            oob = prob.wtmad2(err_out)
            outside_fit = float((wo * (Ao @ th_out - yo).abs()).mean())
            # benchmark cross-validation with the outside pool always in training
            masks, kinds = fl.fold_masks(prob)
            A = torch.cat([prob.A, Ao]); y = torch.cat([prob.y, yo]); w = torch.cat([prob.w, wo])
            full = torch.cat([masks, torch.ones((masks.shape[0], len(yo)), dtype=torch.bool, device=fl.DEV)], dim=1)
            th = fl.solve(A, y, w, lam, full)
            err = th @ prob.A.T - prob.y[None, :]
            rnd = []
            for seed in (0, 1, 2):
                e = torch.zeros_like(prob.y)
                for j, (kind, s_, _) in enumerate(kinds):
                    if kind == "random" and s_ == seed:
                        e[~masks[j]] = err[j][~masks[j]]
                rnd.append(prob.wtmad2(e))
            lso = torch.zeros_like(prob.y)
            for j, (kind, _, _) in enumerate(kinds):
                if kind == "subset":
                    lso[~masks[j]] = err[j][~masks[j]]
            ref = fl.fit_and_validate(prob, lam)
            row = {"lambda": lam, "outside_only_on_benchmark": oob, "outside_fit_weighted_error": outside_fit,
                   "outside_cv_weighted_error": outside_cv,
                   "joint_cv_random": sum(rnd) / 3, "joint_cv_subset_out": prob.wtmad2(lso), "joint_in_sample": prob.wtmad2(err[0]),
                   "benchmark_only_cv_random": ref["cv_random"], "benchmark_only_cv_subset_out": ref["cv_leave_subset_out"],
                   "oob_by_subset": prob.by_subset(err_out)}
            print(f"   lam {lam:7.0e}: outside-only fit -> benchmark {oob:6.3f} (outside pool: fit {outside_fit:6.3f}, held-out {outside_cv:6.3f}) | "
                  f"joint: in-sample {row['joint_in_sample']:6.3f}  CV random {row['joint_cv_random']:6.3f}  CV subset out {row['joint_cv_subset_out']:6.3f} | "
                  f"benchmark only: CV random {ref['cv_random']:6.3f}  CV subset out {ref['cv_leave_subset_out']:6.3f}")
            if best is None or oob < best["outside_only_on_benchmark"]:
                best = row
            if chosen is None or outside_cv < chosen["outside_cv_weighted_error"]:
                chosen = row
        print(f"   -> lambda chosen from outside data alone: {chosen['lambda']:.0e}; benchmark error at that setting: "
              f"{chosen['outside_only_on_benchmark']:.3f} (out-of-benchmark), joint CV random {chosen['joint_cv_random']:.3f}, "
              f"joint CV subset out {chosen['joint_cv_subset_out']:.3f}")
        report["models"].append({"label": label, "p": len(keys), "n_benchmark": len(prob.y), "n_outside": len(yo),
                                 "outside_sets": counts, "chosen_by_outside_cv": chosen,
                                 "lowest_benchmark_error_for_reference_only": best})
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
