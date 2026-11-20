"""Per-subset breakdown of a fit report (file I/O and a few sums)."""
import json
import sys


def main():
    report = json.load(open(sys.argv[1]))
    key = sys.argv[2]                      # leading text of the model label, e.g. "K:"
    protocol_avg = json.load(open(sys.argv[3])) if len(sys.argv) > 3 else None
    model = next(m for m in report["models"] if m["label"].startswith(key))
    print(model["label"], "| n =", model["n"], "| p =", model["p"], "| lambda =", model.get("lambda"))
    base = next(m for m in report["models"] if m["label"].startswith("PBE"))
    rpa = next(m for m in report["models"] if m["label"].startswith("EXX + RPA"))
    rows = []
    for name, v in model["by_subset_cv_random"].items():
        rows.append((name, v["n"], v["of"], base["by_subset"][name]["mad"], rpa["by_subset"][name]["mad"],
                     model["by_subset_in_sample"][name]["mad"], v["mad"], model["by_subset_cv_lso"][name]["mad"]))
    print(f"{'subset':11s} {'n/of':>8s} {'PBE':>7s} {'EXX+RPA':>8s} {'fit-in':>7s} {'CV-rand':>8s} {'CV-LSO':>7s}   (MAD, kcal/mol)")
    for r in sorted(rows, key=lambda r: -r[6] * r[1]):
        print(f"{r[0]:11s} {r[1]:4d}/{r[2]:<3d} {r[3]:7.2f} {r[4]:8.2f} {r[5]:7.2f} {r[6]:8.2f} {r[7]:7.2f}")
    rx = sorted(model.get("reactions", []), key=lambda r: -abs(r["cv"]) * r["w"])
    tot = sum(abs(r["cv"]) * r["w"] for r in rx) or 1.0
    print("largest weighted cross-validation errors (id, reference, CV error, weight, share of total):")
    for r in rx[:28]:
        print(f"   {r['id']:16s} ref {r['ref']:9.2f}  cv_err {r['cv']:+7.2f}  in_err {r['in']:+7.2f}  w {r['w']:6.2f}  share {abs(r['cv'])*r['w']/tot*100:5.1f}%")
    th = model.get("theta", {})
    print("parameters:")
    for k, v in th.items():
        print(f"   {k:32s} {v:+.4f}")


if __name__ == "__main__":
    main()
