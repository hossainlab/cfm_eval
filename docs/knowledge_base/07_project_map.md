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

## 7.5 Current status (baselines + scGPT, 2026-10-15)

Mean over datasets, seeds 0–2, no label noise. Few-shot n = training cells per (patient, class). Results in `results/{ready,matched,t3}/`; paired comparisons in `compare_vs_{hvg_pca,randproj,scgpt_random}.csv`.

**Ready set (malignant vs all normal; 7 datasets, 76 patients)**

| Embedding | kNN purity | Leiden ARI | LOPO AUROC (full) | Few-shot 50 | Few-shot 10 | Normal excess | Malignant excess |
|---|---|---|---|---|---|---|---|
| HVG+PCA | 0.984 | 0.949 | 0.990 | 0.955 | 0.886 | 0.402 | 0.768 |
| Random projection | 0.853 | 0.677 | 0.990 | 0.990 | 0.987 | 0.130 | 0.557 |
| Harmony (uses patient labels) | 0.981 | 0.925 | 0.994 | 0.991 | 0.958 | 0.106 | 0.342 |
| scGPT | 0.980 | 0.921 | 0.995 | 0.994 | 0.992 | 0.391 | 0.547 |
| scGPT random weights | 0.637 | 0.015 | 0.955 | 0.906 | 0.827 | 0.022 | 0.038 |

**Matched set (malignant vs lineage-matched normal; 4 datasets, 28 patients)**

| Embedding | kNN purity | Leiden ARI | LOPO AUROC (full) | Few-shot 10 | Normal excess | Malignant excess |
|---|---|---|---|---|---|---|
| HVG+PCA | 0.971 | 0.904 | 0.966 | 0.863 | 0.393 | 0.692 |
| Random projection | 0.889 | 0.564 | 0.955 | 0.931 | 0.166 | 0.430 |
| Harmony | 0.958 | 0.848 | 0.973 | 0.905 | 0.096 | 0.374 |
| scGPT | 0.949 | 0.709 | 0.973 | 0.950 | 0.415 | 0.550 |
| scGPT random weights | 0.652 | 0.000 | 0.868 | 0.658 | 0.018 | 0.060 |

**T3, held-out cancer AUROC (pooled cells)**

| Held-out cancer | HVG+PCA | Random proj. | Harmony | scGPT | scGPT random |
|---|---|---|---|---|---|
| Breast | 0.995 | 0.972 | 0.992 | 0.987 | 0.921 |
| CRLM | 0.992 | 0.992 | 0.987 | 0.991 | 0.971 |
| Follicular lymphoma | 0.653 | 0.691 | 0.870 | 0.553 | 0.641 |
| Glioblastoma | 0.718 | 0.877 | 0.870 | 0.927 | 0.831 |
| HNSCC | 0.992 | 0.984 | 0.995 | 0.959 | 0.861 |
| LUAD | 0.979 | 0.984 | 0.980 | 0.994 | 0.906 |
| Melanoma | 0.974 | 0.928 | 0.899 | 0.877 | 0.784 |

**scGPT paired comparisons** (mean AUROC difference across held-out patients, patient-bootstrap 95 % CI, Holm-adjusted p)

| Task | vs HVG+PCA | vs random projection | vs scGPT random weights |
|---|---|---|---|
| Ready, few-shot 10 | +0.064 [0.036, 0.097], p = 5e-7 | +0.005 [0.002, 0.008], p = 6e-4 | +0.146 [0.131, 0.161], p < 1e-12 |
| Ready, full probe | +0.004 [−0.004, 0.015], p = 0.79 | +0.004 [0.000, 0.006] | +0.032 [0.025, 0.040] |
| Matched, few-shot 10 | +0.044 [0.002, 0.088], p = 0.37 | +0.023 [0.010, 0.037], p = 0.03 | +0.272 [0.245, 0.302] |
| Matched, full probe | −0.002 [−0.017, 0.015], p = 0.35 | +0.019 [0.010, 0.029], p = 0.001 | +0.090 [0.070, 0.114] |
| T3 (cross-cancer) | **−0.042** [−0.069, −0.015], p = 0.004 | −0.040 [−0.063, −0.020], p = 0.06 | +0.044 [0.018, 0.069], p = 3e-4 |

**What we have learned so far**

1. The full malignant-vs-all-normal probe is at ceiling (≈ 0.99 for everything, including random projection). It cannot rank models.
2. **H5, pretraining matters for the architecture**: pretrained scGPT beats its random-weight twin everywhere, by large margins. Random-weight scGPT collapses (all cells nearly identical, Leiden ARI ≈ 0) and is worse than random projection.
3. **But pretrained scGPT barely beats random projection.** In few-shot probes scGPT's big win over HVG+PCA (+0.064) is mostly shared by the 512-d random projection; scGPT's edge over random projection is +0.005 on the ready set and +0.02 on the matched set. Dimensionality of the embedding, not learned biology, explains most of the few-shot gain. The open question about the fixed-C probe and dimensionality (50-d vs 512-d) is now central.
4. **H1 (unsupervised)**: scGPT does not beat HVG+PCA (ready Leiden ARI 0.92 vs 0.95; matched 0.71 vs 0.90).
5. **H3 (cross-cancer)**: scGPT is significantly *worse* than HVG+PCA overall. It is best for GBM and LUAD but near chance for follicular lymphoma (0.55).
6. **H4 (patient structure)**: scGPT keeps normal-cell patient effects (0.39, same as HVG+PCA) while losing some malignant patient structure (0.77 → 0.55): the unfavourable combination.
7. **Label noise changes the ranking**: in few-shot-10 probes scGPT leads random projection without noise (0.992 vs 0.987) but trails it at 10 % noise (0.812 vs 0.824) and 20 % noise (0.662 vs 0.684). The scGPT-vs-random-projection conclusion depends on label quality.
8. **Pretraining overlap**: scGPT's few-shot-10 AUROC is similar on possibly-seen CELLxGENE datasets (0.990) and GEO datasets (0.995); no sign that leakage inflates it.
9. Harmony removes normal-cell patient effects (0.40 → 0.11) but also roughly halves malignant patient structure (0.77 → 0.34). This is the H4 trade-off.

## 7.6 What is in progress or not done

| Item | Status |
|---|---|
| scGPT embeddings | Done for all 7 datasets (MPS, about 42 s per 1,000 cells, 5,000-cell chunks in separate processes) |
| Random-weight scGPT embeddings | Done; benchmarked |
| Geneformer V2-104M | Environment and wrapper ready; timing and padding check pending |
| UCE | Not started (stretch goal) |
| scVI baseline | Code exists, not run |
| CNV tier-2 labels and high-confidence set | Not started (infercnvpy extra) |
| T5 meta-programs | Not started |
| Figures (guide §9) | Not started |
| Probe dimensionality control (e.g. PCA of each embedding to 50-d, or tuned C via inner LOPO) | Needed to interpret few-shot results |
| Low scGPT vocabulary match on CRLM (0.58) | To investigate |

## 7.7 House rules (from CLAUDE.md, do not break)

- The patient is the unit of replication.
- No leakage: probe tuning uses training patients only.
- Report T4 separately for normal and malignant cells.
- Report Harmony separately (it uses patient labels).
- Give each FM the input it expects: raw counts, the right gene identifier.
- Compute every FM embedding on exactly the benchmark cell set, and cache it with `obs_names`.
