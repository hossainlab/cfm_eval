"""T3 cross-cancer transfer: pool datasets, train the probe on all cancer types but one, test on the held-out type.
   python scripts/run_cross_cancer.py --h5ad "data/ready/*.h5ad" --emb-dir cache/emb --out results_t3/ [--seeds 0 1 2]
Baselines are fitted on the pooled cells (shared genes only); FM embeddings are loaded per dataset and stacked,
and an FM is used only if every dataset has a cache for it. Each cancer type is one dataset here, so held-out
cancer is confounded with study and technology: report T3 as leave-one-dataset-out.
Output: per_patient.csv (held-out cancer x patient x embedding x seed) and summary.csv.
"""
import argparse, glob
from pathlib import Path
import numpy as np, pandas as pd, anndata as ad
from cfm_eval.data import read_std, eligible_patients, subsample_per_patient
from cfm_eval import baselines, metrics
from cfm_eval.embed_fm import load_cached

def pool(files, args, seed):
    parts, fm = [], {m: [] for m in args.fm}
    for f in files:
        name = Path(f).stem
        a = read_std(f)
        a = a[a.obs["patient"].isin(eligible_patients(a, args.min_malignant, args.min_normal))].copy()
        a = subsample_per_patient(a, args.max_cells_per_patient, seed)
        for m in args.fm:
            E = load_cached(a, args.emb_dir, name, m)
            if fm[m] is not None:
                fm[m] = None if E is None else fm[m] + [E]
        a.obs["dataset"] = name
        a.obs["patient"] = name + ":" + a.obs["patient"].astype(str)  # patient IDs can collide across studies
        a.obs_names = name + ":" + a.obs_names
        a.var_names_make_unique()
        parts.append(a)
    genes = sorted(set.intersection(*(set(a.var_names) for a in parts)))
    a = ad.concat([p[:, genes] for p in parts], join="inner", merge=None)
    a.obs = a.obs[["dataset", "patient", "cancer_type", "malignant", "cell_type"]]
    for c in ["dataset", "patient", "cancer_type", "cell_type"]:
        a.obs[c] = a.obs[c].astype(str).astype("category")
    a.obs["malignant"] = a.obs["malignant"].astype(bool)
    for m, Es in fm.items():
        if Es is not None:
            a.obsm[f"X_{m}"] = np.vstack(Es)
    return a

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5ad", nargs="+", required=True)
    ap.add_argument("--emb-dir", default="cache/emb")
    ap.add_argument("--out", default="results_t3")
    ap.add_argument("--fm", nargs="*", default=["scgpt", "geneformer", "uce"])
    ap.add_argument("--max-cells-per-patient", type=int, default=2000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--min-malignant", type=int, default=50)
    ap.add_argument("--min-normal", type=int, default=50)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    files = sorted(sum([glob.glob(x) for x in args.h5ad], []))
    per_patient = []
    for seed in args.seeds:
        a = pool(files, args, seed)
        print(f"[pooled] seed={seed}: {a.n_obs} cells, {a.n_vars} shared genes, "
              f"{a.obs.cancer_type.nunique()} cancer types, {a.obs.patient.nunique()} patients", flush=True)
        keys = [baselines.hvg_pca(a), baselines.random_projection(a, seed=seed)]
        try:
            keys.append(baselines.harmony(a))
        except ImportError:
            pass
        keys += [k for k in a.obsm if k.startswith("X_") and k not in keys and k != "X_pca"]
        for k in keys:
            pp = metrics.cross_cancer_probe(a, k, seed=seed).assign(embedding=k, seed=seed)
            per_patient.append(pp)
            print(f"  {k}: " + ", ".join(f"{c}={v:.3f}" for c, v in
                                         pp.groupby("held_out_cancer")["auroc_pooled"].first().items()), flush=True)
    pp = pd.concat(per_patient)
    pp.to_csv(out / "per_patient.csv", index=False)
    rows = []
    for (k, seed, ct), g in pp.groupby(["embedding", "seed", "held_out_cancer"]):
        mean, lo, hi = metrics.patient_bootstrap(g, seed=seed)
        rows.append({"embedding": k, "seed": seed, "held_out_cancer": ct, "n_patients": len(g),
                     "auroc_pooled": g["auroc_pooled"].iloc[0], "auroc_patient_mean": mean,
                     "auroc_lo": lo, "auroc_hi": hi, "bal_acc": g["bal_acc"].mean()})
    pd.DataFrame(rows).to_csv(out / "summary.csv", index=False)

if __name__ == "__main__":
    main()
