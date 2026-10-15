"""End-to-end driver. Usage:
   python run_benchmark.py --h5ad data/std/*.h5ad --emb-dir cache/emb --out results/
Each input h5ad must already be standardised (cfm_eval.data.standardise) and contain raw counts.
Foundation-model embeddings are read from <emb-dir>/<dataset>__<model>.npy if present.
"""
import argparse, glob, json
from pathlib import Path
import numpy as np, pandas as pd, anndata as ad
from cfm_eval import baselines, metrics
from cfm_eval.data import eligible_patients, subsample_per_patient

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5ad", nargs="+", required=True)
    ap.add_argument("--emb-dir", default="cache/emb")
    ap.add_argument("--out", default="results")
    ap.add_argument("--fm", nargs="*", default=["scgpt", "geneformer", "uce"])
    ap.add_argument("--max-cells-per-patient", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    summary, per_patient = [], []
    for f in sorted(sum([glob.glob(x) for x in args.h5ad], [])):
        name = Path(f).stem
        a = ad.read_h5ad(f)
        a = a[a.obs["patient"].isin(eligible_patients(a))].copy()
        a = subsample_per_patient(a, args.max_cells_per_patient, args.seed)
        keys = [baselines.hvg_pca(a), baselines.random_projection(a, seed=args.seed)]
        try:
            keys.append(baselines.harmony(a))
        except ImportError:
            pass
        for m in args.fm:
            p = Path(args.emb_dir) / f"{name}__{m}.npy"
            if p.exists():
                E = np.load(p)
                # embeddings must be computed on the SAME subsampled cells; store obs_names alongside
                names_p = p.with_suffix(".obs_names.txt")
                if names_p.exists():
                    order = pd.Index(open(names_p).read().split())
                    E = pd.DataFrame(E, index=order).loc[a.obs_names].to_numpy()
                a.obsm[f"X_{m}"] = E; keys.append(f"X_{m}")
        for k in keys:
            u = metrics.unsupervised(a, k, seed=args.seed)
            ps = metrics.patient_structure(a, k)
            pp = metrics.lopo_probe(a, k, seed=args.seed); pp["dataset"] = name; pp["embedding"] = k
            mean, lo, hi = metrics.patient_bootstrap(pp)
            summary.append({"dataset": name, "embedding": k, "n_cells": a.n_obs,
                            "n_patients": a.obs.patient.nunique(), **u, **ps,
                            "lopo_auroc": mean, "lopo_auroc_lo": lo, "lopo_auroc_hi": hi})
            per_patient.append(pp)
        print(f"[done] {name}: {a.n_obs} cells, {a.obs.patient.nunique()} patients, {len(keys)} embeddings")
    pd.DataFrame(summary).to_csv(out / "summary.csv", index=False)
    pd.concat(per_patient).to_csv(out / "per_patient.csv", index=False)

if __name__ == "__main__":
    main()
