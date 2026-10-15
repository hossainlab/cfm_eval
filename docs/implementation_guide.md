* Implementation guide: cancer-focused zero-shot evaluation of single-cell foundation models

## 1. Research question and hypotheses

**Main question.** Do zero-shot cell embeddings from single-cell foundation models capture tumour biology better than cheap baselines? The test is whether the embeddings separate malignant from non-malignant cells and generalise to patients and cancer types the probe has never seen.

| ID | Hypothesis                                                                                                          | Test                                         |
| -- | ------------------------------------------------------------------------------------------------------------------- | -------------------------------------------- |
| H1 | FM embeddings separate malignant from normal cells without labels at least as well as HVG+PCA                       | T1 unsupervised                              |
| H2 | A linear probe on FM embeddings generalises to held-out patients better than on HVG+PCA                             | T2 leave-one-patient-out (LOPO)              |
| H3 | FM embeddings transfer across cancer types better than baselines                                                    | T3 leave-one-cancer-out                      |
| H4 | FM embeddings remove patient batch effects in normal cells *without* erasing patient-specific malignant structure | T4 patient structure                         |
| H5 | Gains come from pretraining, not architecture or dimensionality                                                     | random-weight and random-projection controls |

H4 is the cancer-specific contribution. Malignant cells from different patients carry different genomes, so they *should* stay separated. Standard integration metrics reward mixing them, which is the wrong target here. Everything else in the design serves an honest test of H1–H5.

## 2. Data

**Primary source: Curated Cancer Cell Atlas (3CA).** 3CA hosts 124 scRNA-seq datasets covering more than 40 cancer types. Expression matrices and metadata can be downloaded per study, and each dataset has cell-type annotations with a malignant label, usually supported by inferred copy-number profiles. Download needs no registration.

**Selection rules** (write them down before looking at results):

- 6–10 datasets spanning 4–5 cancer types (e.g., glioma, head-and-neck SCC, lung adenocarcinoma, breast, melanoma, colorectal).
- At least 5 patients per dataset, each with ≥50 malignant and ≥50 non-malignant cells (`eligible_patients()`).
- Raw counts are available. Several FMs need raw counts, and Geneformer also needs Ensembl IDs.
- Mix 10x and plate-based (Smart-seq2) datasets so technology can be analysed as a factor.

**Pretraining-overlap audit.** scGPT and Geneformer were pretrained on large public corpora, largely CELLxGENE. For every dataset, record whether it, or the study it comes from, is in CELLxGENE Census. Report every result split into "possibly seen in pretraining" and "unseen". This is cheap, and it is exactly the leakage control reviewers will ask about.

**Size on a 24 GB laptop.** Cap cells at 2,000 per patient (1,000 malignant + 1,000 normal; `subsample_per_patient`). That gives roughly 10–40k cells per dataset. Fix the subsample once and save the cell list. Every embedding must be computed on exactly those cells.

## 3. Labels and ground truth

Author labels are the starting point, but they are partly produced by the same expression data being evaluated, so the analysis uses three label tiers:

1. **Author malignant label** (3CA). This is the primary label.
2. **Own CNV call.** Run inferCNV-style inference yourself (infercnvpy in Python, or inferCNV/CopyKAT in R) with immune and stromal cells as the reference, and threshold the CNV score.
3. **High-confidence set.** Keep cells where tiers 1 and 2 agree; report the main results on this set and on all cells.

Optional stronger ground truth: a dataset with matched scDNA-seq, or the allele-aware calls of Numbat. Numbat needs BAM files, so treat it as a stretch goal.

**Label-noise sensitivity.** Randomly flip 5%, 10% and 20% of labels and check whether model rankings change. If they do, your conclusions depend on label quality, and you should say so.

## 4. Embeddings

| Embedding                                            | Type             | What it controls for             | Laptop path                                                                               |
| ---------------------------------------------------- | ---------------- | -------------------------------- | ----------------------------------------------------------------------------------------- |
| scGPT (whole-human)                                  | FM, zero-shot    | —                               | CPU: `embed_data(..., device="cpu", use_fast_transformer=False)`; MPS may need patching |
| Geneformer (smallest checkpoint or quantised)        | FM, zero-shot    | —                               | CPU/MPS; needs Ensembl IDs and `n_counts`                                               |
| UCE (4-layer)                                        | FM, zero-shot    | —                               | stretch; or a single Colab/Kaggle pass                                                    |
| HVG + PCA (50)                                       | baseline         | "Is pretraining needed at all?"  | seconds                                                                                   |
| scVI (unsupervised, no batch key)                    | baseline         | learned non-pretrained model     | minutes on CPU                                                                            |
| Harmony on PCA                                       | baseline         | explicit batch removal           | seconds; uses patient labels, so report separately                                        |
| Random projection of log-expression                  | negative control | dimensionality alone             | seconds                                                                                   |
| Random-weight FM (same architecture, re-initialised) | negative control | architecture vs pretraining (H5) | same cost as the FM                                                                       |

**Compute strategy.**

1. Time each FM on 1,000 cells first and extrapolate before launching a full dataset.
2. Run each FM once per dataset, save `.npy` plus the `obs_names` order, and never recompute.
3. If a CPU run would take more than about 8 h per dataset, run that single embedding pass on a free Kaggle or Colab GPU, then download the `.npy`.
4. Fine-tuning is out of scope. That is the point of a zero-shot study, and it keeps compute small.

## 5. Tasks and metrics (implemented in `metrics.py`)

| Task                                  | Function               | Metrics                                                                            |
| ------------------------------------- | ---------------------- | ---------------------------------------------------------------------------------- |
| T1 unsupervised separation            | `unsupervised`       | kNN purity of malignant label, silhouette (ASW), Leiden majority-vote ARI          |
| T2 cross-patient generalisation       | `lopo_probe`         | per-held-out-patient AUROC, AUPRC, balanced accuracy                               |
| T3 cross-cancer transfer              | `cross_cancer_probe` | AUROC on held-out cancer type (pooled across datasets)                             |
| T4 patient structure                  | `patient_structure`  | same-patient neighbour excess, **separately** for normal and malignant cells |
| T5 (optional) malignant heterogeneity | add                    | correlation of embedding neighbourhoods with 3CA meta-program scores               |

**Probe choice.** The probe is logistic regression with standardised features. A linear probe measures what the embedding already contains; a deep probe would measure the probe. Fix C in advance, or tune it with inner LOPO on the training patients only.

## 6. Statistics

- **The patient is the unit of replication, not the cell.** Confidence intervals come from resampling patients (`patient_bootstrap`).
- **Model comparisons** use a paired Wilcoxon test across held-out patients (`paired_compare`), with Holm correction across the comparisons you pre-specify.
- **Repeat over seeds** for subsampling and Leiden clustering (≥3 seeds), and report the spread.
- **Report effect sizes** (median AUROC difference), not just p-values. If the CI of the difference includes 0, say the models are equivalent.

## 7. Smoke test (run here, synthetic data)

A synthetic tumour dataset was used to check the pipeline end to end: 6 patients, 4 shared normal cell types with mild patient batch effects, and malignant cells carrying patient-specific copy-number blocks and programs (4,680 cells after QC).

| Embedding         | kNN purity | LOPO AUROC (95% CI)  | Normal same-patient excess | Malignant same-patient excess |
| ----------------- | ---------- | -------------------- | -------------------------- | ----------------------------- |
| HVG + PCA         | 0.998      | 0.959 (0.907–0.993) | 0.464                      | 0.833                         |
| Random projection | 0.693      | 0.812 (0.734–0.882) | 0.095                      | 0.482                         |
| Harmony           | 0.977      | 0.992 (0.978–0.999) | −0.016                    | 0.553                         |

These numbers only show that the code runs and that the T4 metric behaves as intended. Harmony brought normal-cell patient mixing to about zero, but it also reduced the patient-specific structure of malignant cells from 0.833 to 0.553. That is the trade-off H4 is designed to expose in real data, where FMs should be judged the same way. They say nothing about real foundation models.

## 8. Timeline (about 16 weeks, part-time compatible)

| Weeks  | Milestone                                                                           |
| ------ | ----------------------------------------------------------------------------------- |
| 1–2   | Pre-register datasets, hypotheses, metrics; set up repo; run smoke test             |
| 3–4   | Download and standardise 3CA datasets; QC; pretraining-overlap audit                |
| 5–6   | CNV labels (tier 2), high-confidence set, label-agreement report                    |
| 7–9   | Baselines plus scGPT and Geneformer embeddings (time first, cache everything)       |
| 10–11 | T1–T4 on all datasets; seeds; random-weight control                                |
| 12     | Label-noise sensitivity; seen-vs-unseen split                                       |
| 13–14 | Figures: per-patient AUROC distributions, rank heatmap, T4 normal-vs-malignant plot |
| 15–16 | Write-up (bioRxiv / workshop paper); public repo with exact commands                |

## 9. Expected figures

1. A study design schematic (datasets × labels × embeddings × tasks).
2. LOPO AUROC per patient, one dot per held-out patient, for each embedding, with paired lines.
3. A T4 scatter plot: normal-cell mixing (x) vs malignant patient-specificity (y). The ideal embedding sits in the lower right of the normal axis while keeping malignant structure.
4. A cross-cancer transfer matrix.
5. Robustness panels for label noise, seen vs unseen datasets, and the random-weight control.

## 10. Pitfalls to avoid

- **Cell order mismatch** between the AnnData and the cached embeddings. Always save and reload `obs_names` (the driver checks for this).
- **Gene-ID mismatch.** scGPT matches gene symbols to its vocabulary and Geneformer expects Ensembl IDs. Report the fraction of genes matched per dataset, because low overlap silently degrades FMs.
- **Normalisation.** Give each FM the input it expects (raw counts in `layers['counts']`). Don't pass log-scaled matrices.
- **Rewarding the wrong thing.** Do not use batch-mixing scores on malignant cells as a success metric (H4).
- **Leakage in the probe.** Tune hyperparameters only on training patients, and never use cells from the held-out patient.
- **Over-claiming.** "No significant difference across N patients" is a result, not a failure. Frame it as an equivalence finding with CIs.
