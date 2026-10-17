"""Paired model comparisons on per-patient results (T2 from run_benchmark.py or T3 from run_cross_cancer.py).
   python scripts/compare.py results/per_patient.csv --ref X_hvg_pca --out results/compare.csv
Each patient's metric is first averaged over seeds; then every embedding is compared with --ref by a paired
Wilcoxon test across held-out patients, within each dataset (or held-out cancer) and pooled over all patients.
Holm correction runs within each family: same probe and label-noise level, and either the pooled (primary) tests
or the per-dataset tests.
"""
import argparse
import pandas as pd
from cfm_eval.metrics import paired_compare, holm

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("per_patient")
    ap.add_argument("--ref", default="X_hvg_pca")
    ap.add_argument("--metric", default="auroc")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    pp = pd.read_csv(args.per_patient)
    group = "dataset" if "dataset" in pp else "held_out_cancer"
    family = [c for c in ["probe", "label_noise"] if c in pp]
    pp["patient"] = pp[group].astype(str) + ":" + pp["patient"].astype(str)
    pp = pp.groupby(family + [group, "embedding", "patient"], as_index=False)[args.metric].mean()
    rows = []
    for fam, f in pp.groupby(family) if family else [((), pp)]:
        fam = dict(zip(family, fam if isinstance(fam, tuple) else (fam,)))
        res = []
        groups = list(f.groupby(group))
        for g, d in groups + ([("ALL", f)] if len(groups) > 1 else []):
            ref = d[d.embedding == args.ref]
            for emb in sorted(set(d.embedding) - {args.ref}):
                r = paired_compare(d[d.embedding == emb], ref, col=args.metric)
                res.append({**fam, group: g, "embedding": emb, "ref": args.ref, **r})
        res = pd.DataFrame(res)
        res["p_holm"] = res.groupby(res[group] == "ALL")["p_wilcoxon"].transform(holm)
        rows.append(res)
    out = pd.concat(rows)
    out.to_csv(args.out, index=False)
    print(out.round(4).to_string(index=False))

if __name__ == "__main__":
    main()
