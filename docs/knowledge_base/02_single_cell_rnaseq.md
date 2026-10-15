# 2. Single-cell RNA sequencing

## 2.1 What is measured

**Single-cell RNA sequencing (scRNA-seq)** measures, for each individual cell, how many mRNA molecules of each gene were captured. The output is a **count matrix**:

```
            gene_1  gene_2  gene_3  ...  gene_30000
cell_1        0       3       0              1
cell_2        5       0       0              0
...
cell_50000    0       1      12              0
```

- Rows are **cells** (also called *observations*, `obs`). A dataset has thousands to hundreds of thousands.
- Columns are **genes** (also called *variables* or *features*, `var`). Usually 20,000–35,000.
- Values are non-negative integers (for count-based protocols).

Before single-cell methods, **bulk RNA-seq** ground up a whole tissue sample and measured the *average* expression of millions of cells. A tumour biopsy average blends cancer, immune and stromal cells into one number per gene. Single-cell data un-mix that average, which is what makes this project possible.

## 2.2 How it works (enough to understand the artefacts)

1. **Dissociate** the tissue into a suspension of single cells (enzymes, mechanical force). This stresses cells and can kill fragile types.
2. **Isolate** each cell: in a tiny oil droplet (droplet methods) or in a well of a plate (plate methods).
3. **Capture and tag** the mRNA. Each molecule gets a **cell barcode** (which cell it came from) and, in most droplet methods, a **UMI** (unique molecular identifier, a random tag that lets us count each original molecule once even after it is copied many times).
4. **Amplify** (copy) and **sequence** the tagged molecules.
5. **Align** reads to the reference genome and **count** molecules per gene per cell.

## 2.3 Technologies in our datasets

| Technology | Datasets here | Cells per run | Genes detected per cell | Values |
|---|---|---|---|---|
| **10x Genomics Chromium** (droplet, 3′ or 5′ end) | LUAD, breast, follicular lymphoma | Many (thousands to tens of thousands) | Few (~1,000–3,000) | UMI counts |
| **BD Rhapsody** (microwell) | CRLM | Many | Few | UMI/molecule counts |
| **Smart-seq2** (plate, full-length) | Melanoma, HNSCC, GBM | Few (hundreds to a few thousand) | Many (~4,000–8,000) | Read counts, often released as TPM |

**10x-style data**: lots of cells, shallow per cell, integer UMI counts.
**Smart-seq2 data**: fewer cells, deep per cell, often published as **TPM** (transcripts per million), a length- and depth-normalised value that is *not* an integer count.

Why this matters: models and baselines assume a particular input. Foundation models were pretrained mostly on 10x-style UMI counts. Our HNSCC dataset is available only as TPM, back-transformed from the authors' `log2(TPM/10 + 1)` values, so it violates the "raw counts" contract and its results should be read with that caveat. Mixing technologies is intentional (guide §2): it lets us check whether conclusions hold across technologies.

## 2.4 Properties of the data you must know

**Sparsity.** Most entries are zero, often 90–95 % for 10x data. A zero can mean "gene off" or "gene on but its molecules were not captured" (a *dropout*). You cannot tell which from one cell.

**Library size varies.** Cells differ in total counts (the **library size**, or `n_counts`) for technical reasons: capture efficiency, cell size. A cell with 2× the total counts is not 2× as "active". This is why we normalise (chapter 4).

**Over-dispersion.** Counts are noisier than a Poisson process; models such as scVI use a negative-binomial distribution.

**Quality control (QC).** Bad "cells" are removed before analysis:

- *Too few genes detected*: empty droplets or broken cells. We require ≥ 200 genes (`cfm_eval/data.py: qc`).
- *High mitochondrial fraction*: genes starting with `MT-` come from mitochondria. A dying cell leaks its mRNA from the main compartment but keeps mitochondrial transcripts, so a high `MT-` fraction (here > 20 %) flags dying cells.
- *Doublets*: two cells in one droplet, which look like an impossible hybrid. Not filtered in this project's pipeline; most source datasets were already curated.

## 2.5 Batch effects and patient effects

A **batch effect** is a systematic, non-biological difference between groups of cells processed separately (different day, kit lot, sequencer, lab). Batch effects can be large enough that cells cluster by batch rather than by type.

In tumour data, **each patient is usually its own batch**. So patient-to-patient differences mix two things:

- **Technical batch effects**: affect every cell from that patient, normal and malignant alike.
- **Real patient biology**: mostly in the malignant cells, which carry patient-specific genomes (chapter 1.6).

Batch-correction methods (Harmony, scVI with a batch key) try to remove the first. They cannot tell it apart from the second, so they tend to remove some real tumour signal too. The project quantifies this trade-off in task T4. Correction methods are covered in chapter 4.

## 2.6 Where the "malignant" label comes from

We need a ground-truth label, malignant or not, for every cell. Nobody can look at a cell under the sequencer and see that it is cancer, so labels are *inferred*. Typical approaches, often combined by the original authors:

1. **Marker genes and clustering.** Cluster cells, then identify immune (`PTPRC`), stromal, endothelial and epithelial clusters by markers. In carcinomas, epithelial cells inside the tumour are candidate malignant cells.
2. **CNV inference** (inferCNV, CopyKAT, infercnvpy, Numbat). Order genes by chromosome position, average expression in sliding windows, and compare with a reference of clearly normal cells (e.g. immune cells). Cells showing broad gains or losses along chromosome arms are called malignant. This is the closest thing to a genomic ground truth in expression data.
3. **Patient-specific clustering.** Because malignant cells cluster by patient and normal cells by type, clusters that are unique to one patient are suspicious for malignancy.

**Caveat, central to this project:** these labels were produced *from the same expression data we evaluate*, often using clustering on PCA, the very baseline we compare against. A baseline might score well partly because the labels were made with something like it. The implementation guide therefore plans three label tiers: (1) author labels, (2) our own CNV calls, (3) a high-confidence set where both agree. It also plans a **label-noise sensitivity** analysis that randomly flips 5–20 % of labels and checks whether model rankings change. The flipping is already implemented (`metrics.flip_labels`); tiers 2 and 3 are not yet done.

Our labels per dataset are listed in `data/ready/MANIFEST.csv` (`label_source` column): for example, author column `cell.types == 'Mal'` for melanoma, and CELLxGENE `cell_type in ['malignant cell', 'neoplastic cell']` for CELLxGENE datasets.

## 2.7 Check your understanding

1. A cell has 0 counts for `CD3E`. Is it definitely not a T cell? *(No: it could be a dropout. Look at several marker genes together.)*
2. Why can't we just compare raw counts of a gene between two cells? *(Library sizes differ; normalise first.)*
3. Why might Harmony lower the "malignant same-patient excess" in T4, and why is that not automatically good? *(It removes patient differences, including real tumour genetics.)*

Next: [03_data_formats.md](03_data_formats.md)
