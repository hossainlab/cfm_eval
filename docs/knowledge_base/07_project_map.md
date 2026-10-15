# 7. Project map: from concepts to code

## 7.1 Hypotheses

| ID | Hypothesis | Tested by |
|---|---|---|
| H1 | FM embeddings separate malignant from normal cells without labels at least as well as HVG+PCA | T1 |
| H2 | A linear probe on FM embeddings generalises to held-out patients better than on HVG+PCA | T2 (LOPO; full and few-shot) |
| H3 | FM embeddings transfer across cancer types better than baselines | T3 |
| H4 | FM embeddings remove patient batch effects in normal cells *without* erasing patient-specific malignant structure | T4 |
| H5 | Gains come from pretraining, not architecture or dimensionality | random projection, random-weight scGPT |

The full research design, including timeline and expected figures, is in `docs/implementation_guide.md`.

## 7.2 Datasets

| File stem | Cancer | Technology | Values | Eligible patients | Pretraining overlap |
|---|---|---|---|---|---|
| `geo_GSE115978_melanoma` | Melanoma | Smart-seq2 | counts | 12 | unknown |
| `geo_GSE103322_hnscc` | Head & neck SCC | Smart-seq2 | **TPM** (caveat) | 9 | unknown |
| `geo_GSE131907_luad` | Lung adenocarcinoma | 10x 3′ | UMI | 21 | unknown |
| `cxg_breast_ctm2` | Breast (pleural effusion) | 10x 3′ v3 | UMI | 4 | possibly seen |
| `cxg_crlm_bdrhapsody` | Colorectal liver metastasis | BD Rhapsody | counts | 6 | possibly seen |
| `cxg_gbm_darmanis` | Glioblastoma | Smart-seq2 | counts | 4 | possibly seen |
| `cxg_follicular_lymphoma` | Follicular lymphoma | 10x 5′ v1 | UMI | 20 | possibly seen |

Total in `data/ready`: 93,614 cells from 76 patients. The matched task (`data/ready_matched`) covers the four CELLxGENE datasets: malignant cells versus same-lineage normal cells (chapter 1.3).

## 7.3 Pipeline

```
scripts/prepare_data.py      download + standardise + QC  -> data/std, data/ready
scripts/prepare_matched.py   lineage-matched subsets       -> data/ready_matched
        │
        ├── (FM env) scripts/embed_all.sh / embed_fm.py   -> cache/emb/<dataset>__<model>.npy + .obs_names.txt
        │
scripts/run_benchmark.py     T1, T2 (full + few-shot), T4, label noise, seeds -> results/<set>/summary.csv, per_patient.csv
scripts/run_cross_cancer.py  T3 on pooled datasets (shared genes)            -> results/t3/
scripts/compare.py           paired Wilcoxon vs HVG+PCA, Holm, bootstrap CIs  -> compare.csv
```

| Module | Role |
|---|---|
| `cfm_eval/loaders.py` | Source-specific loaders (GEO, CELLxGENE) into a common format |
| `cfm_eval/data.py` | `standardise`, `qc`, `eligible_patients`, `subsample_per_patient`, `read_std`, `make_ready(_matched)`, `pretrain_overlap` |
| `cfm_eval/baselines.py` | `hvg_pca`, `harmony`, `random_projection`, `scvi_latent` |
| `cfm_eval/embed_fm.py` | Cache I/O (`load_cached`, `save_cached`), scGPT and Geneformer wrappers, random-weight scGPT, coverage checks |
| `cfm_eval/metrics.py` | All metrics and statistics (chapter 6) |

## 7.4 Commands

```bash
uv sync                                    # main environment
export NUMBA_CACHE_DIR=/tmp/nb             # macOS numba cache fix

# smoke test; numbers must match docs/implementation_guide.md §7
uv run python test/make_synth.py
uv run python scripts/run_benchmark.py --h5ad test/synth.h5ad --out test/results

# full benchmark (baselines + any cached FM embeddings)
uv run python scripts/run_benchmark.py --h5ad "data/ready/*.h5ad" --out results/ready \
    --seeds 0 1 2 --label-noise 0 0.05 0.1 0.2 --n-train 10 50
uv run python scripts/run_benchmark.py --h5ad "data/ready_matched/*.h5ad" --out results/matched \
    --seeds 0 1 2 --label-noise 0 0.05 0.1 0.2 --n-train 10 50 --min-malignant 20 --min-normal 20
uv run python scripts/run_cross_cancer.py --h5ad "data/ready/*.h5ad" --out results/t3 --seeds 0 1 2
uv run python scripts/compare.py results/ready/per_patient.csv --out results/ready/compare.csv
```

## 7.5 Current status (baselines only, 2026-10-15)

Mean over datasets, seeds 0–2, no label noise.

**Ready set (malignant vs all normal)**

| Embedding | kNN purity | Leiden ARI | LOPO AUROC (full) | Few-shot 50 | Few-shot 10 | Normal excess | Malignant excess |
|---|---|---|---|---|---|---|---|
| HVG+PCA | 0.984 | 0.949 | 0.990 | 0.955 | 0.886 | 0.402 | 0.768 |
| Random projection | 0.853 | 0.677 | 0.990 | 0.990 | 0.987 | 0.130 | 0.557 |
| Harmony | 0.981 | 0.925 | 0.994 | 0.991 | 0.958 | 0.106 | 0.342 |

**Matched set (malignant vs lineage-matched normal)**

| Embedding | kNN purity | Leiden ARI | LOPO AUROC (full) | Few-shot 10 |
|---|---|---|---|---|
| HVG+PCA | 0.971 | 0.904 | 0.966 | 0.863 |
| Random projection | 0.889 | 0.564 | 0.955 | 0.931 |
| Harmony | 0.958 | 0.848 | 0.973 | 0.905 |

**T3, held-out cancer AUROC (pooled cells)**: most cancers are at 0.97–0.99 for every baseline. The informative ones are follicular lymphoma (HVG+PCA 0.65, random projection 0.69, Harmony 0.87), glioblastoma (0.72 / 0.88 / 0.87) and melanoma (0.97 / 0.93 / 0.90).

**What we have learned so far**

1. The full malignant-vs-all-normal probe is at ceiling (≈ 0.99 for everything, including the random control). It cannot rank models.
2. Discriminating settings are: unsupervised metrics (T1), few-shot probes, the matched task, and T3 for lymphoma and GBM.
3. Open question: in few-shot probes, the 512-d random projection *beats* 50-d HVG+PCA. This may be a dimensionality or regularisation effect of the fixed-C probe, not a property of the information content. Investigate before interpreting FM few-shot results, because the FMs are 512-d (scGPT) and 768-d (Geneformer).
4. Label noise lowers all scores roughly equally. The baseline ranking was stable at 5–20 % noise.
5. Harmony removes normal-cell patient effects (0.40 → 0.11) but also roughly halves malignant patient structure (0.77 → 0.34). This is the H4 trade-off.

## 7.6 What is in progress or not done

| Item | Status |
|---|---|
| scGPT embeddings | Done for all 7 datasets (MPS, about 42 s per 1,000 cells, 5,000-cell chunks in separate processes) |
| Random-weight scGPT embeddings | Running |
| Geneformer V2-104M | Environment and wrapper ready; timing and padding check pending |
| UCE | Not started (stretch goal) |
| scVI baseline | Code exists, not run |
| CNV tier-2 labels and high-confidence set | Not started (infercnvpy extra) |
| T5 meta-programs | Not started |
| Figures (guide §9) | Not started |
| Low scGPT vocabulary match on CRLM (0.58) | To investigate |

## 7.7 House rules (from CLAUDE.md, do not break)

- The patient is the unit of replication.
- No leakage: probe tuning uses training patients only.
- Report T4 separately for normal and malignant cells.
- Report Harmony separately (it uses patient labels).
- Give each FM the input it expects: raw counts, the right gene identifier.
- Compute every FM embedding on exactly the benchmark cell set, and cache it with `obs_names`.
