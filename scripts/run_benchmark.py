"""End-to-end driver (T1, T2, T4). Usage:
   python scripts/run_benchmark.py --h5ad data/std/*.h5ad --emb-dir cache/emb --out results/
       [--seeds 0 1 2] [--label-noise 0 0.05 0.1 0.2] [--n-train 10 50]
Each input h5ad must already be standardised (cfm_eval.data.standardise) and contain raw counts.
Foundation-model embeddings are read from <emb-dir>/<dataset>__<model>.npy if present.
Outputs: summary.csv (dataset x embedding x seed x label_noise) and per_patient.csv (adds probe = full | fewshot<n>).
"""
import argparse, glob
from pathlib import Path
import pandas as pd
from cfm_eval.data import read_std, eligible_patients, subsample_per_patient, pretrain_overlap
from cfm_eval import baselines, metrics
from cfm_eval.embed_fm import load_cached

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5ad", nargs="+", required=True)
    ap.add_argument("--emb-dir", default="cache/emb")
    ap.add_argument("--out", default="results")
    ap.add_argument("--fm", nargs="*", default=["scgpt", "geneformer", "uce"])
    ap.add_argument("--max-cells-per-patient", type=int, default=2000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--label-noise", type=float, nargs="+", default=[0.0],
                    help="fractions of labels flipped per patient (sensitivity analysis)")
    ap.add_argument("--n-train", type=int, nargs="*", default=[],
                    help="few-shot probes: max training cells per (patient, class)")
    ap.add_argument("--min-malignant", type=int, default=50)
    ap.add_argument("--min-normal", type=int, default=50)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    summary, per_patient = [], []
    for f in sorted(sum([glob.glob(x) for x in args.h5ad], [])):
        name = Path(f).stem
        full = read_std(f)
        full = full[full.obs["patient"].isin(eligible_patients(full, args.min_malignant, args.min_normal))].copy()
        for seed in args.seeds:
            a = subsample_per_patient(full, args.max_cells_per_patient, seed)
            keys = [baselines.hvg_pca(a), baselines.random_projection(a, seed=seed)]
            try:
                keys.append(baselines.harmony(a))
            except ImportError:
                pass
            for m in args.fm:
                E = load_cached(a, args.emb_dir, name, m)
                if E is not None:
                    a.obsm[f"X_{m}"] = E; keys.append(f"X_{m}")
            for noise in args.label_noise:
                b = metrics.flip_labels(a, noise, seed)
                for k in keys:
                    meta = {"dataset": name, "embedding": k, "seed": seed, "label_noise": noise,
                            "pretrain_overlap": pretrain_overlap(name)}
                    row = {**meta, "n_cells": b.n_obs, "n_patients": b.obs.patient.nunique(),
                           **metrics.unsupervised(b, k, seed=seed), **metrics.patient_structure(b, k)}
                    for n in [None] + args.n_train:
                        probe = f"fewshot{n}" if n else "full"
                        pp = metrics.lopo_probe(b, k, seed=seed, n_train_per_class=n).assign(**meta, probe=probe)
                        mean, lo, hi = metrics.patient_bootstrap(pp, seed=seed)
                        sfx = "" if n is None else f"_fs{n}"
                        row.update({f"lopo_auroc{sfx}": mean, f"lopo_auroc{sfx}_lo": lo, f"lopo_auroc{sfx}_hi": hi})
                        per_patient.append(pp)
                    summary.append(row)
            print(f"[done] {name} seed={seed}: {a.n_obs} cells, {a.obs.patient.nunique()} patients, "
                  f"{len(keys)} embeddings x {len(args.label_noise)} noise levels", flush=True)
    pd.DataFrame(summary).to_csv(out / "summary.csv", index=False)
    pd.concat(per_patient).to_csv(out / "per_patient.csv", index=False)

if __name__ == "__main__":
    main()
