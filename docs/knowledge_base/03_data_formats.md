# 3. Data formats, identifiers and sources

## 3.1 AnnData and `.h5ad`

The Python single-cell ecosystem (the **scverse**: scanpy, anndata, scvi-tools) stores data in an **AnnData** object, saved to disk as an **`.h5ad`** file (HDF5 under the hood). Think of it as a matrix with aligned pandas tables around it:

```
                     var  (one row per gene: DataFrame)
                     ┌───────────────────────────┐
                     │ feature_name, ensembl_id  │
                     └───────────────────────────┘
  obs (one row       ┌───────────────────────────┐   obsm (per-cell matrices,
  per cell:          │                           │   e.g. embeddings)
  DataFrame)         │   X  (cells × genes)      │   ┌──────────────────┐
  ┌──────────────┐   │   usually sparse          │   │ X_hvg_pca  (n×50) │
  │ patient      │   │                           │   │ X_scgpt   (n×512) │
  │ malignant    │   │   layers["counts"]        │   │ X_harmony (n×50)  │
  │ cell_type    │   │   (same shape as X)       │   └──────────────────┘
  │ cancer_type  │   └───────────────────────────┘
  └──────────────┘   uns: free-form metadata (provenance, parameters)
```

| Slot | Meaning | In this project |
|---|---|---|
| `adata.X` | The main matrix, usually `scipy.sparse` | Raw counts (contract: never log-transformed on disk) |
| `adata.layers["counts"]` | Alternative matrices of the same shape | A copy of raw counts, restored by `read_std()` |
| `adata.obs` | Per-cell table; index = `obs_names` (cell IDs) | `patient`, `sample`, `malignant` (bool), `cell_type`, `cancer_type`, `source` |
| `adata.var` | Per-gene table; index = `var_names` | Index = gene symbol; columns `feature_name`, `ensembl_id` |
| `adata.obsm` | Per-cell multi-column arrays | Embeddings: `X_hvg_pca`, `X_randproj`, `X_harmony`, `X_<fm>` |
| `adata.uns` | Unstructured metadata | `provenance`: accession, value type, label source, subsampling rule |

Useful idioms:

```python
from cfm_eval.data import read_std
a = read_std("data/ready/cxg_gbm_darmanis.h5ad")   # restores layers["counts"]
a.obs["malignant"].mean()                          # fraction malignant
a.obs.groupby("patient")["malignant"].agg(["sum", "count"])
a[a.obs["malignant"]].copy()                       # subset cells (copy() avoids "view" surprises)
```

Pitfall: slicing an AnnData returns a **view**. Modifying a view triggers warnings or silent copies; call `.copy()` when you subset and then modify.

## 3.2 Gene identifiers

The same gene has several names:

| Identifier | Example | Notes |
|---|---|---|
| **Gene symbol** (HGNC) | `TP53`, `PTPRC` | Human-readable. Symbols get renamed over time, and some are ambiguous or duplicated. |
| **Ensembl gene ID** | `ENSG00000141510` | Stable, unambiguous. Version suffixes (`.17`) are usually stripped. |

Different models expect different identifiers:

- **scGPT** matches *gene symbols* against its vocabulary (`var["feature_name"]`). Its human vocabulary has about 60,700 entries.
- **Geneformer** needs *Ensembl IDs* (`var["ensembl_id"]`) and the per-cell total count `obs["n_counts"]`. Its V2 vocabulary has about 20,000 protein-coding genes.
- **UCE** maps genes to protein embeddings by symbol.

Mapping coverage is a real issue: for the GEO datasets only 86 % (melanoma, HNSCC) and 65 % (LUAD) of symbols could be mapped to Ensembl IDs, and unmapped genes are invisible to Geneformer. **Always report the fraction of genes matched per model and dataset** (guide §10). Low overlap silently degrades a model. `scripts/embed_fm.py` prints these numbers.

## 3.3 Where the data come from

| Source | What it is | How we use it |
|---|---|---|
| **NCBI GEO** (Gene Expression Omnibus) | Archive where authors deposit raw and processed data; accessions look like `GSE131907` | Melanoma (GSE115978), HNSCC (GSE103322), LUAD (GSE131907) |
| **CZ CELLxGENE Discover / Census** | Curated, standardised `.h5ad` collection (tens of millions of cells) with harmonised cell-type labels; Census is its programmatic API | Breast, CRLM, GBM, follicular lymphoma; also the **pretraining-overlap audit** |
| **Curated Cancer Cell Atlas (3CA)** | 124 curated tumour scRNA-seq datasets with malignant annotations and meta-programs | Planned primary source in the guide; reference for labels and meta-programs |

**Pretraining overlap.** scGPT's whole-human model was pretrained on CELLxGENE Census data (May 2023 snapshot, per its `args.json`), and Geneformer V2 on a large public corpus. A CELLxGENE dataset may therefore have been *seen during pretraining*, a form of leakage. We tag every dataset (`cfm_eval.data.pretrain_overlap`): `cxg_*` = `possibly_seen`, `geo_*` = `unknown`. Results should always be reported split by this flag.

**Duplicate data.** GSE72056 (Tirosh 2016 melanoma) is excluded because its cells are re-released inside GSE115978. Counting them twice would inflate the patient count.

## 3.4 The files in this repo

```
data/
  std/            full standardised objects (rebuilt by scripts/prepare_data.py; large, not shipped)
  ready/          benchmark-ready: eligible patients only, ≤1,000 cells per patient × class
  ready_matched/  HARD task: malignant vs same-lineage normal cells only
  */MANIFEST.csv  per-dataset counts, label source, value type, accession
cache/emb/        cached foundation-model embeddings: <dataset>__<model>.npy + .obs_names.txt
results/          benchmark outputs (summary.csv, per_patient.csv, compare.csv)
```

**Eligibility rule**: a patient is kept only if they have ≥ 50 malignant *and* ≥ 50 normal cells (`eligible_patients`, relaxed to 20/20 for the matched task). Without both classes, the patient cannot be a held-out test case in leave-one-patient-out evaluation.

**Subsampling**: at most 1,000 malignant and 1,000 normal cells per patient (`subsample_per_patient`), so that one large patient does not dominate, and to fit a 24 GB laptop. The subsample is fixed once (seed 0) and saved. **Every embedding must be computed on exactly these cells.**

**Embedding cache contract**: `cache/emb/<dataset>__<model>.npy` (cells × dims) plus `.obs_names.txt` (one cell ID per line, same order). The benchmark re-orders rows by `obs_names` (`cfm_eval.embed_fm.load_cached`), so a cache may cover a superset of cells. One cache per dataset serves both `ready` and `ready_matched`.

Next: [04_standard_analysis.md](04_standard_analysis.md)
