# cfm_eval — zero-shot evaluation of single-cell foundation models on tumours

## Install
Managed with [uv](https://docs.astral.sh/uv/): `uv sync` (Python 3.11, versions locked in `uv.lock`; extras `--extra scvi`, `--extra cnv`).
scGPT, Geneformer and UCE pin old torch stacks and are not in this environment: run `scripts/embed_fm.py` in a separate
FM environment and copy the cached `.npy` files into `cache/emb/`.
macOS: `export NUMBA_CACHE_DIR=/tmp/nb` before running (numba cache fix).

## Data (7 datasets, 6 cancer types; benchmark-ready files ship separately as cfm_eval_data_ready.tar)
| file | source | cancer | assay | value type | eligible patients* | label |
|---|---|---|---|---|---|---|
| geo_GSE115978_melanoma | GEO GSE115978 (Jerby-Arnon 2018) | melanoma | Smart-seq2 | counts as released | 12 | author `Mal` |
| geo_GSE103322_hnscc | GEO GSE103322 (Puram 2017) | head & neck SCC | Smart-seq2 | TPM (from log2(TPM/10+1)) | 9 | author "classified as cancer cell" |
| geo_GSE131907_luad | GEO GSE131907 (Kim 2020) | lung adenocarcinoma (metastatic/advanced samples) | 10x | raw UMI | 21 | author `Malignant cells` |
| cxg_breast_ctm2 | CELLxGENE 6ff7acae… (doi 10.1002/ctm2.1356) | breast (pleural effusion) | 10x 3' v3 | raw counts | 4 | CELLxGENE `malignant cell` |
| cxg_crlm_bdrhapsody | CELLxGENE a3bcb68b… (doi 10.1186/s12943-025-02430-7) | colorectal liver metastasis | BD Rhapsody | raw counts | 6 | CELLxGENE `malignant cell` |
| cxg_gbm_darmanis | CELLxGENE 2e357b12… (doi 10.1016/j.celrep.2017.10.030) | glioblastoma | Smart-seq2 | integer counts | 4 | CELLxGENE `neoplastic cell` |
| cxg_follicular_lymphoma | CELLxGENE 61c4a85a… (doi 10.1158/2643-3230.BCD-21-0075) | follicular lymphoma | 10x 5' v1 | raw counts | 20 | CELLxGENE `malignant cell` |
*≥50 malignant and ≥50 non-malignant cells. GSE72056 (Tirosh 2016) has a loader but is EXCLUDED: its cells are re-released inside GSE115978 ("Tirosh" cohort; 1,724 shared cell names, 99% label agreement).

Standardised obs columns: patient, sample, malignant, cell_type, cancer_type, source (+ source-specific). var index = gene symbol;
var['ensembl_id'] mapped from CELLxGENE tables for GEO data (86% of genes for GSE115978/GSE103322, 65% for GSE131907 — check before Geneformer).
`uns['provenance']` records accession, value type, label source and dropped cells.
Pretraining-leakage grouping: CELLxGENE datasets = "possibly seen" by scGPT/Geneformer; GEO-only = "unknown (check)".

## Rebuild from scratch
```
uv run python scripts/prepare_data.py --download --raw data/raw --std data/std --ready data/ready     # ~3.2 GB download, ~10 min processing
uv run python scripts/prepare_matched.py --std data/std --out data/ready_matched
```
`data/std` = full standardised objects; `data/ready` = eligible patients, ≤1,000 cells per patient × class;
`data/ready_matched` = HARD task, malignant vs same-lineage normal only (FL: B cells; GBM: astrocyte/OPC/oligodendrocyte;
breast: mesothelial; CRLM: cholangiocyte/hepatocyte). Read with `cfm_eval.data.read_std()` (restores layers['counts']).

## Reproduce the results
Commands that produced `results/` (seeds 0–2; label noise 0/5/10/20 %; full, 50-shot and 10-shot probes):
```
# 1. FM embeddings, in the FM environment (one run per model; union of ready + matched cells per dataset)
scripts/embed_all.sh <fm-python> scgpt      <scGPT_human dir>
scripts/embed_all.sh <fm-python> scgpt      <random-weight dir> scgpt_random   # dir from cfm_eval.embed_fm.make_random_scgpt
scripts/embed_all.sh <fm-python> geneformer <Geneformer dir>
scripts/embed_all.sh <fm-python> uce        <UCE repo clone with model_files/>

# 2. Benchmark (T1, T2, T4) and cross-cancer transfer (T3)
FM="--fm scgpt scgpt_random geneformer uce"
uv run python scripts/run_benchmark.py --h5ad data/ready/*.h5ad --out results/ready \
    --seeds 0 1 2 --label-noise 0 0.05 0.1 0.2 --n-train 10 50 $FM
uv run python scripts/run_benchmark.py --h5ad data/ready_matched/*.h5ad --out results/matched \
    --seeds 0 1 2 --label-noise 0 0.05 0.1 0.2 --n-train 10 50 --min-malignant 20 --min-normal 20 $FM
uv run python scripts/run_cross_cancer.py --h5ad "data/ready/*.h5ad" --out results/t3 --seeds 0 1 2 $FM

# 3. Paired comparisons (Wilcoxon across held-out patients, Holm) and figures
for t in ready matched t3; do for r in hvg_pca randproj scgpt_random; do
  uv run python scripts/compare.py results/$t/per_patient.csv --ref X_$r --out results/$t/compare_vs_$r.csv
done; done
uv run python scripts/make_figures.py --results results --out results/figures
```
Tracked in git: `results/{ready,matched,t3}/` (`summary.csv`, `per_patient.csv`, `compare_vs_*.csv`) and `results/figures/`
(fig1/fig2/figS1 as PDF + PNG, `captions.txt`). Not tracked: `data/`, `cache/emb/` (FM embeddings, ~3.8 GB), logs.

Smoke test on synthetic data (expected values in `docs/implementation_guide.md` §7, e.g. HVG+PCA LOPO AUROC ≈ 0.959):
```
uv run python test/make_synth.py && uv run python scripts/run_benchmark.py --h5ad test/synth.h5ad --out test/results
```

## Findings so far
See `docs/knowledge_base/07_project_map.md` §7.5. The full malignant-vs-all-normal probe is at ceiling (≈ 0.99 for every
embedding, including the random-projection control), so models are ranked on unsupervised metrics, few-shot probes, the
lineage-matched task and cross-cancer transfer.

## Tested vs untested
Run on real data: loaders.py (all 4 GEO + CELLxGENE), data.py, baselines.py (hvg_pca, harmony, random_projection), metrics.py,
embed_fm.py (scGPT, random-weight scGPT, Geneformer, UCE), and every script in `scripts/`. Not tested: `baselines.scvi_latent`.
