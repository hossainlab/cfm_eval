# 6. Evaluation and statistics

Everything in this chapter is implemented in `cfm_eval/metrics.py` and driven by the scripts in `scripts/`.

## 6.1 How do you evaluate an embedding?

An embedding is just a matrix (cells × dimensions). We ask two kinds of question about it:

- **Unsupervised**: without using labels, is the malignant/normal distinction already visible in the geometry (who is near whom)?
- **Supervised probe**: if we train the *simplest possible* classifier on top of the embedding, how well does it predict malignancy on cells it has never seen?

**Linear probe.** A logistic regression on standardised embedding features, with a fixed regularisation strength (`C = 1`). We use a linear model on purpose: a deep classifier could learn the task itself and would measure the probe, not the embedding (guide §5).

## 6.2 The tasks

| Task | Question | Function | Script |
|---|---|---|---|
| **T1** unsupervised separation | Are malignant and normal cells separated without labels? | `unsupervised` | `run_benchmark.py` |
| **T2** cross-patient generalisation | Does a probe trained on other patients work on a new patient? | `lopo_probe` | `run_benchmark.py` |
| **T3** cross-cancer transfer | Does a probe trained on other cancer types work on a new one? | `cross_cancer_probe` | `run_cross_cancer.py` |
| **T4** patient structure | Do normal cells mix across patients while malignant cells stay patient-specific? | `patient_structure` | `run_benchmark.py` |
| T5 (optional) | Does the embedding capture malignant meta-programs? | not implemented | — |

## 6.3 Leave-one-patient-out (LOPO), the core design

For each patient *p* in a dataset:

1. **Train** the probe on all cells from every *other* patient.
2. **Test** on all cells of patient *p*.
3. Record AUROC, AUPRC and balanced accuracy for *p*.

```
patients:  A  B  C  D  E
fold 1:   [test][  train  train  train  train ]
fold 2:   [train][test][ train  train  train  ]
...
```

Why not a random 80/20 split of cells? Because cells from the same patient are highly correlated: same genome, same batch. A random split puts sibling cells in both train and test, the probe memorises patient-specific patterns, and the score is inflated. This is **leakage**. LOPO mimics the real use case: a new patient walks in.

**No-leakage rule**: any hyperparameter tuning must use only the training patients. Never use cells of the held-out patient.

**Few-shot variant** (`--n-train 10 50`). The full probe is at ceiling (AUROC ≈ 0.99 for everything), so it cannot rank models. With only 10 or 50 training cells per (patient, class), better embeddings should need fewer examples. This is where differences show up.

**T3** is the same idea at a higher level: hold out a whole cancer type. In our data each cancer type comes from a single study, so "held-out cancer" also means "held-out lab and technology". Report it as leave-one-dataset-out.

## 6.4 Metrics, one by one

### Classification metrics (T2, T3)

- **AUROC** (area under the ROC curve). The probability that a randomly chosen malignant cell gets a higher score than a randomly chosen normal cell. 0.5 = chance, 1.0 = perfect. Insensitive to class balance and to the decision threshold. **Our primary metric.**
- **AUPRC** (area under the precision–recall curve). More sensitive when the positive class is rare; its chance level equals the fraction of positives.
- **Balanced accuracy**. Mean of sensitivity and specificity at threshold 0.5. Unlike AUROC, it does depend on the threshold, so it reveals calibration shifts between patients.

### Unsupervised metrics (T1)

- **kNN purity**: for each cell, the fraction of its 15 nearest neighbours with the same label, averaged over cells. 1.0 = neighbourhoods are pure.
- **Silhouette width (ASW)**: compares each cell's mean distance to its own class with its mean distance to the other class. Ranges −1 to 1. Typically small (0.05–0.3) in high-dimensional data, so compare relatively, not absolutely.
- **Leiden majority-vote ARI**: cluster with Leiden, label each cluster by its majority class, then compare with the truth using the **adjusted Rand index** (0 = chance agreement, 1 = identical partitions).

### Patient-structure metric (T4)

**Same-patient excess**: for each cell, take the fraction of its 15 neighbours from the same patient and subtract the fraction expected by chance (that patient's share of cells). Average over cells. Computed **separately** for normal and malignant cells.

| | Normal cells | Malignant cells |
|---|---|---|
| Value near 0 | Good: patients mix, batch effect removed | Suspicious: patient-specific tumour genetics may be erased |
| Large value | Batch effect remains | Expected: each tumour is genetically distinct |

The ideal embedding has **low normal excess and high malignant excess**. Never collapse these into a single "batch-mixing" score (guide §1, §10).

## 6.5 Statistics: the patient is the unit

We have tens of thousands of cells but only 4–21 patients per dataset. Cells within a patient are not independent observations. Treating 30,000 cells as 30,000 samples gives absurdly narrow confidence intervals. Therefore:

- **Confidence intervals**: `patient_bootstrap` resamples *patients* with replacement (2,000 times), recomputes the mean per-patient AUROC each time, and takes the 2.5th and 97.5th percentiles.
- **Model comparisons**: `paired_compare` uses a **paired Wilcoxon signed-rank test** across held-out patients. For each patient, compute AUROC(model) − AUROC(HVG+PCA); test whether the median difference is 0. *Paired*, because the same patient is evaluated by both models, which cancels out patient difficulty. Non-parametric, because AUROC differences are not normally distributed.
- **Multiple comparisons**: testing many models × datasets inflates false positives. **Holm correction** (`metrics.holm`) adjusts p-values within a pre-specified family. In `scripts/compare.py` the families are: same probe and label-noise level, and either the pooled tests (primary) or the per-dataset tests.
- **Effect sizes**: always report the median and mean difference with a bootstrap CI, not just p-values. If the CI of the difference includes 0, say the models are statistically equivalent at this sample size.
- **Power caveat**: with 4 patients the smallest possible two-sided Wilcoxon p-value is 0.125, so a significant per-dataset result is impossible. That is why pooled tests across datasets are the primary analysis. Small datasets (breast and GBM with 4 patients each) contribute to pooled tests and CIs, not to standalone claims.
- **Seeds**: repeat with at least 3 seeds (`--seeds 0 1 2`); `compare.py` averages each patient's metric over seeds before testing.

## 6.6 Robustness analyses

- **Label noise** (`--label-noise 0 0.05 0.1 0.2`): flip a fraction of malignant labels per patient (`flip_labels`). Absolute scores drop for every model. The question is whether the *ranking* of models changes. If it does, conclusions depend on label quality.
- **Seen vs unseen**: split results by `pretrain_overlap`.
- **Negative controls**: random projection and random-weight scGPT (chapter 4.3).

## 6.7 Worked example of reading the output

`results/ready/summary.csv` has one row per dataset × embedding × seed × label-noise level. `results/ready/per_patient.csv` has one row per held-out patient × probe. To compare models:

```bash
uv run python scripts/compare.py results/ready/per_patient.csv --ref X_hvg_pca --out results/ready/compare.csv
```

A row such as `probe=fewshot10, dataset=ALL, embedding=X_scgpt, mean_diff=+0.04 [0.01, 0.07], p_holm=0.003` reads: "with 10 training cells per patient and class, scGPT's AUROC is on average 0.04 higher than HVG+PCA across all held-out patients; the 95 % patient-bootstrap CI excludes zero, and the difference survives Holm correction."

Next: [07_project_map.md](07_project_map.md)
