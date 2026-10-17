# 1. Biology primer

This chapter covers only the biology needed to understand the project. It is deliberately simplified; where the simplification matters for the analysis, that is pointed out.

## 1.1 Cells and the genome

The human body is made of roughly 30 trillion **cells**. Almost every cell carries the same instruction manual, the **genome**: about 3 billion letters of **DNA** split across 23 pairs of **chromosomes**. A **gene** is a stretch of DNA that encodes a product the cell can make. Humans have about 20,000 protein-coding genes plus many thousands of non-coding genes.

Every cell has the same genes, yet a neuron, a T cell and a liver cell look and behave completely differently. The difference is *which genes are switched on, and how strongly*. That pattern is called **gene expression**, and it is what our data measure.

## 1.2 From DNA to RNA to protein

The "central dogma" of molecular biology:

```
DNA (gene, stored permanently)
  │  transcription: the cell copies an active gene into a messenger molecule
  ▼
mRNA (messenger RNA, short-lived, many copies when the gene is highly active)
  │  translation: the ribosome reads the mRNA and builds a protein
  ▼
Protein (does the actual work: enzymes, receptors, structure, signals)
```

A gene that is strongly "on" produces many mRNA copies; a gene that is "off" produces none. **Counting the mRNA molecules of each gene inside a cell is therefore a proxy for what that cell is doing.** The set of all mRNA in a cell is its **transcriptome**, and measuring it is **transcriptomics**.

Two practical consequences:

- mRNA is a *snapshot*: it reflects the cell's state at the moment it was captured.
- Some genes are on in nearly every cell (**housekeeping genes**, e.g. ribosomal proteins). Others are highly specific (**marker genes**): `CD3E` marks T cells, `PTPRC` (also called CD45) marks all immune cells, `EPCAM` marks epithelial cells, `PECAM1` marks endothelial cells.

## 1.3 Cell types and lineages

Cells are grouped into **cell types** (T cell, B cell, macrophage, fibroblast, hepatocyte, astrocyte …). Cell types sit in a family tree called a **lineage**: for example, B cells and T cells are both lymphocytes, which are both immune cells. Cells of the same lineage share large parts of their expression programme.

This matters for our project. A cancer usually arises from one normal cell type, its **cell of origin**, and still resembles it:

| Cancer in our data | Arises from | "Lineage-matched" normal counterpart in the matched task |
|---|---|---|
| Follicular lymphoma | B cells | B cells |
| Glioblastoma | Glial cells of the brain | Astrocytes, oligodendrocytes, oligodendrocyte precursor cells |
| Colorectal liver metastasis | Colon epithelium, now growing in the liver | Hepatocytes, cholangiocytes (liver epithelium) |
| Breast cancer (pleural effusion) | Breast epithelium, here floating in fluid around the lung | Mesothelial cells |

Telling a lymphoma cell from a T cell is easy because they are different lineages. Telling a lymphoma cell from a *healthy B cell* is hard because they share a lineage. That is why the project has a separate, harder "lineage-matched" task (chapter 7).

## 1.4 What cancer is at the cellular level

Cancer is a disease of the genome. Over many divisions, a cell accumulates **mutations**, changes in its DNA. Most are harmless. A few hit genes that control growth, and the cell starts dividing without restraint. Its descendants form the tumour. Cells derived from this lineage are called **malignant** (or neoplastic, or tumour) cells.

Kinds of genomic change you will hear about:

- **Point mutations**: single-letter changes (e.g. in `TP53` or `KRAS`). These are mostly *invisible* in our data, because expression data rarely reveal which letter is in an mRNA.
- **Copy-number alterations (CNAs or CNVs)**: whole chunks of chromosomes are duplicated or lost. A cell with three copies of a chromosome arm makes roughly 1.5× more mRNA from all genes on that arm. This *is* visible in expression data, as a broad shift across neighbouring genes, and it is the basis of the CNV-inference methods described in chapter 2.
- **Aneuploidy**: an abnormal number of chromosomes, common in cancer and almost absent in normal cells.

Key point: **malignant cells are defined by their genome, not by their expression.** Expression is an indirect readout. Two cells can look similar in expression while one is malignant and the other is not, especially within the same lineage.

## 1.5 The tumour microenvironment

A tumour is not a ball of identical cancer cells. A biopsy typically contains:

- **Malignant cells**, sometimes a minority.
- **Immune cells**: T cells, B cells, NK cells, macrophages, monocytes, dendritic cells, mast cells, plasma cells.
- **Stromal cells**: fibroblasts (often "cancer-associated fibroblasts"), pericytes, smooth muscle.
- **Endothelial cells**: blood vessel lining.
- **Normal tissue cells** of the organ: hepatocytes in liver, alveolar cells in lung, glial cells in brain.

All the non-malignant cells together form the **tumour microenvironment (TME)**. In this project they make up the "normal" class.

## 1.6 Heterogeneity: why every patient is different

This is the single most important biological idea for understanding the evaluation design.

**Normal cells are similar across patients.** A T cell from patient A looks like a T cell from patient B. Differences between them are mostly technical (sample handling, sequencing run) plus modest biology (age, inflammation).

**Malignant cells are different across patients.** Each patient's tumour has its own unique set of mutations and copy-number changes, acquired by chance during that patient's disease. Patient A's tumour might have gained chromosome 7 and lost 10; patient B's might have gained 1q and lost 17p. Their cancer cells therefore express genes at different levels. Studies of tumour single-cell data routinely find that **malignant cells cluster by patient, while normal cells cluster by cell type** (Tirosh et al. 2016; Puram et al. 2017; see the reading list).

There is also **intra-tumour heterogeneity**: within one patient, subgroups of malignant cells (subclones) differ, and cells cycle between states (proliferating, stressed, partially differentiated). Recurring expression patterns shared across many patients' tumours are called **meta-programs**; the Curated Cancer Cell Atlas catalogues them.

Why this matters for us:

1. **A model must generalise across patients.** If an embedding only learns "patient A's cancer looks like this", it will fail on patient B. That is why we test on held-out patients (leave-one-patient-out, chapter 6).
2. **Batch-mixing scores must not be applied to malignant cells.** Standard single-cell benchmarks reward embeddings where cells from different patients mix together. For normal cells that is right: patient differences there are mostly noise. For malignant cells it is wrong: the patient differences are real genetics. An embedding that perfectly mixes malignant cells across patients may be erasing true biology. The project therefore measures patient structure separately for normal and malignant cells (task T4).

## 1.7 Cancer types in this project

| Cancer | Where | Notes for non-biologists |
|---|---|---|
| Melanoma | Skin pigment cells (melanocytes) | Heavily mutated, often studied for immunotherapy |
| Head and neck squamous cell carcinoma (HNSCC) | Lining of mouth, throat | Epithelial cancer |
| Lung adenocarcinoma (LUAD) | Glandular cells of the lung | Most common lung cancer type |
| Breast cancer | Breast epithelium; here sampled from pleural effusion (fluid around the lungs), i.e. metastatic | |
| Colorectal liver metastasis (CRLM) | Colon cancer that has spread to the liver | Tissue is liver, tumour is colonic |
| Glioblastoma (GBM) | Brain glial cells | Aggressive brain tumour |
| Follicular lymphoma (FL) | B cells in lymph nodes | Blood cancer; malignant cells are B cells, so the normal B cells are close relatives |

**Metastasis** means cancer cells that have spread from the original organ (the primary tumour) to another site.

## 1.8 Check your understanding

1. Why can't we tell a malignant cell from a normal one just by reading its DNA sequence in this data? *(We measure mRNA counts, not the DNA sequence; most mutations are invisible. Copy-number changes are partly visible.)*
2. Why should normal cells from two patients look alike while malignant cells do not? *(Normal cells share the same genome programme; each tumour has its own genomic alterations.)*
3. Why is "malignant vs lineage-matched normal" harder than "malignant vs all normal"? *(Same lineage means largely shared expression; the remaining difference is subtler.)*

Next: [02_single_cell_rnaseq.md](02_single_cell_rnaseq.md)
