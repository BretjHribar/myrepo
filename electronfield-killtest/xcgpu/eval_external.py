"""Out-of-benchmark test and joint training with the public MSR-ACC/TAE25 atomization energies.

Part 1 (transfer): fit on GMTKN55 reactions only, predict W1-F12 atomization energies of molecules
the fit has never seen. No TAE25 label enters the fit.
Part 2 (joint): add a random half of the external molecules to the training set, score the other
half, and report what that does to the GMTKN55 cross-validation numbers.
External errors are plain (unweighted) kcal/mol on total atomization energies, a different quantity
from WTMAD-2.
"""
import argparse
import json
from pathlib import Path

from .common import BULK, pin_one_core

pin_one_core()
import torch
from . import fitlib as fl
from .fit import LADDER


def stats(e):
    return {"mae": float(e.abs().mean()), "mean_signed": float(e.mean()), "rmse": float((e * e).mean().sqrt()),
            "p95_abs": float(torch.quantile(e.abs(), 0.95)), "max_abs": float(e.abs().max()), "n": int(e.numel())}


def line(tag, s):
    return (f"{tag}: MAE {s['mae']:6.2f}  mean {s['mean_signed']:+6.2f}  RMSE {s['rmse']:6.2f}  "
            f"95th pct {s['p95_abs']:6.2f}  max {s['max_abs']:6.2f}  (n={s['n']})")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="tae25")
    ap.add_argument("--scf-tag", default="pbe_qzvp_g3")
    ap.add_argument("--models", default="K:,N:")
    ap.add_argument("--lam", type=float, default=1e-6)
    ap.add_argument("--external-weight", type=float, default=0.19, help="training weight per external reaction (W4-11 has 0.19)")
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    train_rows, _ = fl.structure_table(args.scf_tag)
    ext_rows, ext_problems = fl.structure_table(args.scf_tag, dataset=args.dataset)
    labels = json.loads((BULK / args.dataset / "labels.json").read_text())
    mols = sorted(m for m in labels if m in ext_rows and all(f"atom/{a}" in ext_rows for a in labels[m]["atoms"]))
    print(f"external molecules with every ingredient: {len(mols)} (problems: {len(ext_problems)})")
    if len(mols) < 20:
        return
    ref = torch.tensor([labels[m]["tae_hartree"] for m in mols], dtype=torch.float64, device=fl.DEV) * fl.KCAL
    elements = sorted({a for m in mols for a in labels[m]["atoms"]})
    counts = torch.tensor([[labels[m]["atoms"].get(a, 0) for a in elements] for m in mols], dtype=torch.float64, device=fl.DEV)
    g = torch.Generator(device="cpu").manual_seed(0)
    perm = torch.randperm(len(mols), generator=g).to(fl.DEV)
    half_a, half_b = perm[:len(mols) // 2], perm[len(mols) // 2:]
    report = {"n_external": len(mols), "elements": elements, "models": []}

    for label, base, blocks in LADDER:
        keys = [k for b in blocks for k in fl.FEATURE_SETS[b]]
        if keys and not any(label.startswith(x) for x in args.models.split(",")):
            continue
        e_mol, f_mol = fl.structure_energy_terms(ext_rows, mols, keys, base)
        e_atom, f_atom = fl.structure_energy_terms(ext_rows, [f"atom/{a}" for a in elements], keys, base)
        a_ext = (counts @ f_atom - f_mol) * fl.KCAL                 # external design matrix (reactions x features)
        y_ext = ref - (counts @ e_atom - e_mol) * fl.KCAL           # what the fitted terms must supply
        ok = torch.isfinite(a_ext).all(1) & torch.isfinite(y_ext)
        row = {"label": label, "p": len(keys)}
        print(f"\n{label}  (p={len(keys)})")
        if not keys:
            row["external_unfitted"] = stats(-y_ext[ok])
            print("   " + line("external, nothing fitted      ", row["external_unfitted"]))
            report["models"].append(row)
            continue
        prob = fl.Problem(train_rows, keys, base)
        # part 1: benchmark-only fit
        theta = fl.solve(prob.A, prob.y, prob.w, args.lam)[0]
        row["n_gmtkn_train"] = len(prob.y)
        row["external_transfer"] = stats((a_ext @ theta - y_ext)[ok])
        base_cv = fl.fit_and_validate(prob, args.lam)
        row["gmtkn_cv_random_benchmark_only"] = base_cv["cv_random"]
        row["gmtkn_cv_lso_benchmark_only"] = base_cv["cv_leave_subset_out"]
        print(f"   trained on {len(prob.y)} GMTKN55 reactions only")
        print("   " + line("external, never seen           ", row["external_transfer"]))
        print(f"   GMTKN55 cross-validation: random folds {base_cv['cv_random']:.3f}, whole subset held out {base_cv['cv_leave_subset_out']:.3f}")
        # part 2: joint fit with half of the external molecules
        tr = half_a[ok[half_a]]
        te = half_b[ok[half_b]]
        w_ext = torch.full((len(tr),), args.external_weight, dtype=torch.float64, device=fl.DEV)
        A = torch.cat([prob.A, a_ext[tr]])
        y = torch.cat([prob.y, y_ext[tr]])
        w = torch.cat([prob.w, w_ext])
        theta_j = fl.solve(A, y, w, args.lam)[0]
        row["external_joint_heldout_half"] = stats(a_ext[te] @ theta_j - y_ext[te])
        row["gmtkn_in_sample_joint"] = prob.wtmad2(prob.A @ theta_j - prob.y)
        # GMTKN55 cross-validation with the external half always in the training set
        masks, kinds = fl.fold_masks(prob)
        full_masks = torch.cat([masks, torch.ones((masks.shape[0], len(tr)), dtype=torch.bool, device=fl.DEV)], dim=1)
        th = fl.solve(A, y, w, args.lam, full_masks)
        err = th @ prob.A.T - prob.y[None, :]
        cv_rand, cv_lso = [], torch.zeros_like(prob.y)
        for seed in (0, 1, 2):
            e = torch.zeros_like(prob.y)
            for j, (kind, s_, _) in enumerate(kinds):
                if kind == "random" and s_ == seed:
                    held = ~masks[j]
                    e[held] = err[j][held]
            cv_rand.append(prob.wtmad2(e))
        for j, (kind, _, _) in enumerate(kinds):
            if kind == "subset":
                held = ~masks[j]
                cv_lso[held] = err[j][held]
        row["gmtkn_cv_random_joint"] = sum(cv_rand) / len(cv_rand)
        row["gmtkn_cv_lso_joint"] = prob.wtmad2(cv_lso)
        print(f"   joint fit: + {len(tr)} external molecules in training (weight {args.external_weight} each)")
        print("   " + line("external, held-out half        ", row["external_joint_heldout_half"]))
        print(f"   GMTKN55 cross-validation: random folds {row['gmtkn_cv_random_joint']:.3f}, whole subset held out {row['gmtkn_cv_lso_joint']:.3f}"
              f"   (in-sample {row['gmtkn_in_sample_joint']:.3f})")
        report["models"].append(row)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
