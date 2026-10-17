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
FM="--fm scgpt scgpt_random geneformer uce"
uv run python scripts/run_benchmark.py --h5ad "data/ready/*.h5ad" --out results/ready \
    --seeds 0 1 2 --label-noise 0 0.05 0.1 0.2 --n-train 10 50 $FM
uv run python scripts/run_benchmark.py --h5ad "data/ready_matched/*.h5ad" --out results/matched \
    --seeds 0 1 2 --label-noise 0 0.05 0.1 0.2 --n-train 10 50 --min-malignant 20 --min-normal 20 $FM
uv run python scripts/run_cross_cancer.py --h5ad "data/ready/*.h5ad" --out results/t3 --seeds 0 1 2 $FM
for t in ready matched t3; do for r in hvg_pca randproj scgpt_random; do
  uv run python scripts/compare.py results/$t/per_patient.csv --ref X_$r --out results/$t/compare_vs_$r.csv
done; done
uv run python scripts/make_figures.py --results results --out results/figures
```

## 7.5 Current status (baselines + scGPT, Geneformer, UCE, 2026-10-17)

Mean over datasets, seeds 0–2, no label noise. Few-shot n = training cells per (patient, class). Embedding size: HVG+PCA 50-d, random projection 512-d, scGPT 512-d, Geneformer V2-104M 768-d, UCE (4-layer) 1280-d. Results in `results/{ready,matched,t3}/`; paired comparisons in `compare_vs_{hvg_pca,randproj,scgpt_random}.csv`; figures in `results/figures/`.

**Ready set (malignant vs all normal; 7 datasets, 76 patients)**

| Embedding | kNN purity | Leiden ARI | LOPO AUROC (full) | Few-shot 50 | Few-shot 10 | Normal excess | Malignant excess |
|---|---|---|---|---|---|---|---|
| HVG+PCA | 0.984 | 0.949 | 0.990 | 0.955 | 0.886 | 0.402 | 0.768 |
| Random projection | 0.853 | 0.677 | 0.990 | 0.990 | 0.987 | 0.130 | 0.557 |
| Harmony (uses patient labels) | 0.981 | 0.925 | 0.994 | 0.991 | 0.958 | 0.106 | 0.342 |
| scGPT | 0.980 | 0.921 | 0.995 | 0.994 | 0.992 | 0.391 | 0.547 |
| Geneformer | 0.979 | 0.907 | 0.995 | 0.995 | 0.994 | 0.427 | 0.631 |
| UCE | 0.987 | 0.951 | 0.993 | 0.995 | 0.995 | 0.357 | 0.577 |
| scGPT random weights | 0.637 | 0.015 | 0.955 | 0.906 | 0.827 | 0.022 | 0.038 |

**Matched set (malignant vs lineage-matched normal; 4 datasets, 28 patients)**

| Embedding | kNN purity | Leiden ARI | LOPO AUROC (full) | Few-shot 50 | Few-shot 10 | Normal excess | Malignant excess |
|---|---|---|---|---|---|---|---|
| HVG+PCA | 0.971 | 0.904 | 0.966 | 0.928 | 0.863 | 0.393 | 0.692 |
| Random projection | 0.889 | 0.564 | 0.955 | 0.944 | 0.931 | 0.166 | 0.430 |
| Harmony | 0.958 | 0.848 | 0.973 | 0.952 | 0.905 | 0.096 | 0.374 |
| scGPT | 0.949 | 0.709 | 0.973 | 0.961 | 0.950 | 0.415 | 0.550 |
| Geneformer | 0.965 | 0.875 | 0.969 | 0.967 | 0.954 | 0.414 | 0.602 |
| UCE | 0.974 | 0.908 | 0.970 | 0.959 | 0.950 | 0.343 | 0.583 |
| scGPT random weights | 0.652 | 0.000 | 0.868 | 0.785 | 0.658 | 0.018 | 0.060 |

**Few-shot-10 AUROC under label noise (ready set)**

| Embedding | 0 % | 5 % | 10 % | 20 % |
|---|---|---|---|---|
| HVG+PCA | 0.886 | 0.797 | 0.734 | 0.626 |
| Random projection | 0.987 | 0.902 | 0.824 | 0.684 |
| Harmony | 0.958 | 0.866 | 0.793 | 0.672 |
| scGPT | 0.992 | 0.896 | 0.812 | 0.662 |
| Geneformer | 0.994 | 0.909 | 0.828 | 0.689 |
| UCE | 0.995 | 0.904 | 0.809 | 0.671 |
| scGPT random weights | 0.827 | 0.738 | 0.680 | 0.589 |

**T3, held-out cancer AUROC (pooled cells)**

| Held-out cancer | HVG+PCA | Random proj. | Harmony | scGPT | Geneformer | UCE | scGPT random |
|---|---|---|---|---|---|---|---|
| Breast | 0.995 | 0.972 | 0.992 | 0.987 | 0.985 | 0.978 | 0.921 |
| CRLM | 0.992 | 0.992 | 0.987 | 0.991 | 0.989 | 0.995 | 0.971 |
| Follicular lymphoma | 0.653 | 0.691 | 0.870 | 0.553 | 0.444 | 0.729 | 0.641 |
| Glioblastoma | 0.718 | 0.877 | 0.870 | 0.927 | 0.919 | 0.925 | 0.831 |
| HNSCC | 0.992 | 0.984 | 0.995 | 0.959 | 0.869 | 1.000 | 0.861 |
| LUAD | 0.979 | 0.984 | 0.980 | 0.994 | 0.994 | 0.992 | 0.906 |
| Melanoma | 0.974 | 0.928 | 0.899 | 0.877 | 0.849 | 0.982 | 0.784 |
| **Mean** | 0.900 | 0.918 | 0.942 | 0.898 | 0.864 | 0.943 | 0.845 |

**FM paired comparisons** (mean AUROC difference across held-out patients, pooled over datasets, patient-bootstrap 95 % CI, Holm-adjusted p; no label noise)

| Task | Model | vs HVG+PCA | vs random projection |
|---|---|---|---|
| Ready, few-shot 10 (n = 76) | scGPT | +0.064 [0.036, 0.097], p < 1e-4 | +0.005 [0.002, 0.008], p = 5e-4 |
| | Geneformer | +0.066 [0.038, 0.099], p < 1e-4 | +0.007 [0.005, 0.010], p < 1e-4 |
| | UCE | +0.067 [0.039, 0.100], p < 1e-4 | +0.008 [0.005, 0.011], p < 1e-4 |
| Ready, full probe | scGPT | +0.004 [−0.004, 0.015], p = 0.79 | +0.004 [0.000, 0.006], p < 1e-4 |
| | Geneformer | +0.004 [−0.003, 0.015], p = 0.71 | +0.004 [0.003, 0.006], p < 1e-4 |
| | UCE | +0.003 [−0.004, 0.014], p = 0.16 | +0.003 [0.001, 0.005], p < 1e-4 |
| Matched, few-shot 10 (n = 28) | scGPT | +0.044 [0.002, 0.088], p = 0.37 | +0.023 [0.010, 0.037], p = 0.03 |
| | Geneformer | +0.057 [0.022, 0.095], p = 0.04 | +0.036 [0.020, 0.053], p = 0.005 |
| | UCE | +0.054 [0.018, 0.093], p = 0.02 | +0.033 [0.012, 0.053], p = 0.03 |
| Matched, full probe | scGPT | −0.002 [−0.017, 0.015], p = 0.39 | +0.019 [0.010, 0.030], p = 0.001 |
| | Geneformer | −0.004 [−0.019, 0.012], p = 0.39 | +0.017 [0.009, 0.025], p = 5e-4 |
| | UCE | 0.000 [−0.012, 0.015], p = 0.39 | +0.021 [0.010, 0.033], p = 0.001 |
| T3, cross-cancer (n = 76) | scGPT | **−0.042** [−0.069, −0.015], p = 0.006 | −0.040 [−0.063, −0.020], p = 0.06 |
| | Geneformer | **−0.091** [−0.130, −0.054], p < 1e-4 | **−0.089** [−0.124, −0.057], p = 3e-4 |
| | UCE | +0.020 [−0.003, 0.044], p = 0.38 | **+0.023** [0.011, 0.033], p < 1e-4 |

scGPT vs its random-weight twin: ready few-shot 10 +0.146 [0.131, 0.161]; matched few-shot 10 +0.272 [0.245, 0.302]; T3 +0.044 [0.018, 0.069] (all p < 1e-3).

**What we have learned so far**

1. The full malignant-vs-all-normal probe is at ceiling (≈ 0.99 for everything, including random projection). It cannot rank models.
2. **H5, pretraining matters for the architecture**: pretrained scGPT beats its random-weight twin everywhere, by large margins. Random-weight scGPT collapses (all cells nearly identical, Leiden ARI ≈ 0) and is worse than random projection.
3. **But all three FMs barely beat random projection.** In 10-shot probes every FM beats HVG+PCA by ≈ +0.065, yet the 512-d random projection shares most of that gain: the FM edge over it is +0.005–0.008 on the ready set and +0.02–0.04 on the matched set. Embedding dimensionality (50-d vs 512–1280-d under a fixed-C probe), not learned biology, explains most of the few-shot gain. A dimension-matched control is now the central open item (7.6).
4. **H1 (unsupervised)**: no FM beats HVG+PCA. UCE ties it (ready Leiden ARI 0.95 vs 0.95; matched 0.91 vs 0.90); Geneformer is slightly lower (0.91 / 0.88); scGPT is clearly lower on the matched set (0.71).
5. **H3 (cross-cancer)**: the FMs split. UCE is the best embedding (mean 0.943, on a par with Harmony, which uses patient labels) and significantly beats random projection. scGPT and Geneformer are significantly *worse* than HVG+PCA; Geneformer is below chance on follicular lymphoma (0.44) and weak on HNSCC (0.87). Each held-out cancer is one study, so T3 is leave-one-study-out.
6. **H4 (patient structure)**: all FMs keep normal-cell patient effects (normal excess 0.36–0.43, like HVG+PCA's 0.40) while losing some malignant patient structure (0.77 → 0.55–0.63): the unfavourable combination. UCE is the least bad (0.36 / 0.58).
7. **Label noise erases the FM lead**: at 10 % noise random projection (0.824) beats scGPT (0.812) and UCE (0.809); only Geneformer stays marginally ahead (0.828). The FM-vs-random-projection conclusion depends on label quality.
8. **Pretraining overlap**: FM few-shot-10 AUROC is similar on possibly-seen CELLxGENE datasets (0.990–0.994) and GEO datasets (0.995–0.997); no sign that leakage inflates it.
9. Harmony removes normal-cell patient effects (0.40 → 0.11) but also roughly halves malignant patient structure (0.77 → 0.34). This is the H4 trade-off.

## 7.6 What is in progress or not done

| Item | Status |
|---|---|
| scGPT embeddings | Done for all 7 datasets (MPS, about 42 s per 1,000 cells, 5,000-cell chunks in separate processes) |
| Random-weight scGPT embeddings | Done; benchmarked |
| Geneformer V2-104M embeddings | Done for all 7 datasets; benchmarked |
| UCE (4-layer) embeddings | Done for all 7 datasets; benchmarked |
| scVI baseline | Code exists, not run |
| CNV tier-2 labels and high-confidence set | Not started (infercnvpy extra) |
| T5 meta-programs | Not started |
| Figures (guide §9) | Done: `results/figures/` from `scripts/make_figures.py` |
| Probe dimensionality control (e.g. PCA of each embedding to 50-d, or tuned C via inner LOPO) | Needed to interpret few-shot results |
| Low scGPT vocabulary match on CRLM (0.58) | To investigate |

## 7.7 House rules (from CLAUDE.md, do not break)

- The patient is the unit of replication.
- No leakage: probe tuning uses training patients only.
- Report T4 separately for normal and malignant cells.
- Report Harmony separately (it uses patient labels).
- Give each FM the input it expects: raw counts, the right gene identifier.
- Compute every FM embedding on exactly the benchmark cell set, and cache it with `obs_names`.
