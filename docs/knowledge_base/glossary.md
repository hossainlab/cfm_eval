# Glossary

**AnnData / .h5ad**: Python object and file format for single-cell data: matrix `X` plus per-cell `obs`, per-gene `var`, embeddings `obsm`. See chapter 3.

**Aneuploidy**: abnormal number of chromosomes; a hallmark of many cancers.

**ARI (adjusted Rand index)**: agreement between two partitions, corrected for chance; 0 = random, 1 = identical.

**ASW (average silhouette width)**: how much closer cells are to their own class than to the other class; −1 to 1.

**AUROC / AUPRC**: area under the ROC / precision–recall curve. AUROC 0.5 = chance, 1 = perfect.

**Balanced accuracy**: mean of sensitivity and specificity.

**Batch effect**: systematic technical difference between groups of cells processed separately.

**Bulk RNA-seq**: expression averaged over all cells of a sample (contrast with single-cell).

**Cell barcode**: sequence tag identifying which cell a molecule came from.

**Cell embedding**: a vector representing one cell, produced by PCA or a model.

**Cell type / lineage**: functional class of a cell / its developmental family tree.

**CELLxGENE (Discover, Census)**: Chan Zuckerberg Initiative's curated single-cell data collection and its API; also the pretraining source for scGPT.

**CLS token**: special token whose output vector summarises the whole input; used as the cell embedding.

**CNA / CNV (copy-number alteration / variation)**: gain or loss of chromosome segments; visible in expression as broad shifts along a chromosome.

**Count matrix**: cells × genes table of mRNA counts.

**Doublet**: two cells captured as one.

**Dropout**: a gene expressed in a cell but recorded as 0 because its molecules were not captured.

**Eligible patient**: has ≥ 50 malignant and ≥ 50 normal cells (20/20 in the matched task).

**Ensembl ID**: stable gene identifier, e.g. `ENSG00000141510`.

**Few-shot probe**: probe trained on only *n* cells per (patient, class).

**Fine-tuning**: further training of a pretrained model on task data (out of scope here).

**Foundation model (scFM)**: large model pretrained on millions of cells, reused for many tasks.

**GEO**: NCBI Gene Expression Omnibus data archive; accessions `GSE…`.

**Harmony**: batch-correction method that adjusts a PCA embedding using batch labels.

**Holm correction**: step-down adjustment of p-values for multiple comparisons.

**Housekeeping gene**: gene expressed in nearly every cell.

**HVG (highly variable genes)**: genes with the highest variance relative to their mean; usually the top 2,000.

**kNN graph**: each cell linked to its *k* nearest neighbours in an embedding.

**kNN purity**: fraction of a cell's neighbours sharing its label.

**Leakage**: information from test data influencing training (e.g. same-patient cells in both).

**Leiden**: graph community-detection (clustering) algorithm.

**Library size (n_counts)**: total counts in a cell.

**Linear probe**: logistic regression trained on frozen embeddings to measure what they encode.

**log1p**: `log(1 + x)` transform.

**LOPO (leave-one-patient-out)**: train on all patients but one, test on the held-out patient, repeat.

**Malignant / neoplastic**: cancer cell, descended from the transformed lineage.

**Marker gene**: gene specific to a cell type.

**Matched task**: malignant vs lineage-matched normal cells (hard version).

**Meta-program**: recurring expression programme across many tumours (catalogued in 3CA).

**Metastasis**: cancer spread to a distant organ.

**Mitochondrial fraction (pct_counts_mt)**: share of counts from `MT-` genes; high values flag dying cells.

**MPS**: Apple-silicon GPU backend in PyTorch.

**Negative control**: method expected to fail (random projection, random-weight model).

**Normal (in this project)**: any non-malignant cell in the sample.

**obs / var / obs_names**: AnnData per-cell table / per-gene table / cell IDs.

**PCA**: linear projection onto directions of maximal variance.

**Pretraining overlap / leakage**: evaluation data possibly included in the model's pretraining corpus.

**Random projection**: multiplication by a random matrix; a no-learning control.

**Rank-value encoding**: Geneformer's tokenisation: genes sorted by corpus-median-scaled expression.

**scGPT / Geneformer / UCE**: the three foundation models evaluated (chapter 5).

**scVI**: variational autoencoder for scRNA-seq counts.

**Smart-seq2**: plate-based, full-length, deep-per-cell scRNA-seq protocol.

**Same-patient excess (T4)**: neighbours from the same patient beyond chance; reported separately for normal and malignant cells.

**Sparsity**: proportion of zeros in the count matrix.

**Symbol (gene)**: human-readable gene name, e.g. `TP53`.

**TME (tumour microenvironment)**: non-malignant cells in and around a tumour.

**TPM**: transcripts per million; normalised, non-integer expression unit.

**Transcriptome**: all mRNA in a cell.

**UMAP**: 2-D visualisation of a kNN graph; not for computing metrics.

**UMI**: unique molecular identifier; lets each original mRNA molecule be counted once.

**Wilcoxon signed-rank test**: non-parametric paired test, used across held-out patients.

**Zero-shot**: using a pretrained model's outputs without any training on the task.

**10x Chromium / BD Rhapsody**: high-throughput droplet / microwell scRNA-seq platforms.
