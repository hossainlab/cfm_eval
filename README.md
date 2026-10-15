# cfm_eval — zero-shot evaluation of single-cell foundation models on tumours

## Install
Python 3.11: scanpy, scikit-learn, anndata, leidenalg, python-igraph, harmonypy, openpyxl (optional: scvi-tools, scgpt, geneformer).
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
PYTHONPATH=. python scripts/prepare_data.py --download --raw data/raw --std data/std --ready data/ready     # ~3.2 GB download, ~10 min processing
PYTHONPATH=. python scripts/prepare_matched.py --std data/std --out data/ready_matched
```
`data/std` = full standardised objects; `data/ready` = eligible patients, ≤1,000 cells per patient × class;
`data/ready_matched` = HARD task, malignant vs same-lineage normal only (FL: B cells; GBM: astrocyte/OPC/oligodendrocyte;
breast: mesothelial; CRLM: cholangiocyte/hepatocyte). Read with `cfm_eval.data.read_std()` (restores layers['counts']).

## Run the benchmark
```
PYTHONPATH=. python scripts/run_benchmark.py --h5ad "data/ready/*.h5ad" --emb-dir cache/emb --out results/
PYTHONPATH=. python scripts/run_benchmark.py --h5ad "data/ready_matched/*.h5ad" --out results_matched/ --min-malignant 20 --min-normal 20
```
FM embeddings: compute on the cells of the ready file, save `cache/emb/<dataset>__<model>.npy` + `.obs_names.txt`.

## Baseline results already computed (results_baseline/)
HVG+PCA, random projection and Harmony on all 7 ready sets and the 4 matched sets. Key observation:
leave-one-patient-out AUROC for malignant vs ALL normal cells is 0.96–1.00 for every embedding, including the random-projection
negative control — this task is at ceiling and cannot rank foundation models. Lineage-matched sets are harder for unsupervised
separation (e.g. follicular lymphoma: Leiden ARI 0.94 HVG+PCA vs 0.00 random projection) but linear probes remain ≥0.90.
Rank models on unsupervised metrics, the matched task, and harder settings (few-shot probes, cross-dataset transfer) rather than
on all-normal probe AUROC.

## Tested vs untested
Tested here on real data: loaders.py (all 4 GEO + CELLxGENE), data.py, baselines.py (hvg_pca, harmony, random_projection),
metrics.py, run_benchmark.py, prepare_data.py, prepare_matched.py. NOT tested: embed_fm.py (scGPT, Geneformer), scvi_latent.
