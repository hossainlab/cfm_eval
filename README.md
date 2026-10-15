# cfm_eval — zero-shot evaluation of single-cell foundation models on tumours

Environment: python 3.11, scanpy, scikit-learn, anndata, leidenalg, python-igraph, harmonypy
(optional: scvi-tools, scgpt, geneformer, infercnvpy). On macOS set `export NUMBA_CACHE_DIR=/tmp/nb`.

1. Standardise each dataset -> `cfm_eval.data.standardise(...)`, `qc(...)`, write `data/std/<name>.h5ad` (raw counts in X).
2. Subsample once with `subsample_per_patient`, write the cell list, and compute FM embeddings ON THAT LIST:
   save `cache/emb/<name>__scgpt.npy` and `cache/emb/<name>__scgpt.obs_names.txt`.
3. `PYTHONPATH=. python run_benchmark.py --h5ad data/std/*.h5ad --emb-dir cache/emb --out results/`
4. Outputs: `results/summary.csv` (one row per dataset x embedding), `results/per_patient.csv` (LOPO probe per patient).

Smoke test: `PYTHONPATH=. python test/make_synth.py && PYTHONPATH=. python run_benchmark.py --h5ad test/synth.h5ad --out test/results`
Tested here: data.py, baselines.py (hvg_pca, harmony, random_projection), metrics.py, run_benchmark.py.
NOT tested here (no weights downloaded): embed_fm.py (scGPT, Geneformer), baselines.scvi_latent.
