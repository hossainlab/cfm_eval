# Reading list

Ordered from "read first" to "reference". DOIs come from `data/data_sources.csv` and the dataset manifests; the URLs were checked there on 2026-10-15.

## Start here (concepts)

1. **scanpy** — Wolf, Angerer & Theis (2018). *SCANPY: large-scale single-cell gene expression data analysis.* Genome Biology. doi:10.1186/s13059-017-1382-0. The "Preprocessing and clustering 3k PBMCs" tutorial at https://scanpy.readthedocs.io/ walks through chapter 4 hands-on in about an hour. **Do this first.**
2. **Curated Cancer Cell Atlas (3CA)** — Nature Cancer (2025). doi:10.1038/s43018-025-00957-8. https://www.weizmann.ac.il/sites/3CA/. Overview of tumour scRNA-seq, malignant annotation and meta-programs.

## The foundation models

3. **scGPT** — Cui et al. (2024). Nature Methods. doi:10.1038/s41592-024-02201-0. https://github.com/bowang-lab/scGPT
4. **Geneformer** — Theodoris et al. (2023). Nature. doi:10.1038/s41586-023-06139-9. https://huggingface.co/ctheodoris/Geneformer. Read the model card for the V2 changes and the exclusion of malignant cells from pretraining.
5. **UCE** — Rosen et al. (2023). bioRxiv. doi:10.1101/2023.11.28.568918. https://github.com/snap-stanford/UCE

## Critical evaluations (closest related work)

6. **Kedzierska et al. (2025)**. Zero-shot evaluation of single-cell foundation models. Genome Biology. doi:10.1186/s13059-025-03574-x. Shows zero-shot embeddings often fail to beat simple baselines.
7. **Wu et al. (2025)**. Benchmark of single-cell foundation models, including cancer cell identification. Genome Biology. doi:10.1186/s13059-025-03781-6. The closest prior benchmark; we position our work against it.

## Baselines and integration

8. **scVI** — Lopez et al. (2018). Nature Methods. doi:10.1038/s41592-018-0229-2
9. **Harmony** (Python port): https://github.com/slowkow/harmonypy

## Malignant labels and copy-number inference

10. **CopyKAT** — Gao et al. (2021). Nature Biotechnology. doi:10.1038/s41587-020-00795-2
11. **Numbat** — Gao et al. (2023). Nature Biotechnology. doi:10.1038/s41587-022-01468-y. Allele-aware; needs BAM files (stretch goal).
12. **infercnvpy**: https://github.com/icbi-lab/infercnvpy · **inferCNV**: https://github.com/broadinstitute/infercnv
13. **Chang et al. (2026, preprint)**. Benchmark of 12 CNV inference methods on 28 datasets. doi:10.64898/2026.04.12.718050

## Data sources

14. **CELLxGENE** — CZI Cell Science Program et al. (2025). Nucleic Acids Research. doi:10.1093/nar/gkae1142. https://cellxgene.cziscience.com/
15. Datasets used here (see `data/ready/MANIFEST.csv`):
    - Breast pleural effusion — doi:10.1002/ctm2.1356
    - Colorectal liver metastasis (BD Rhapsody) — doi:10.1186/s12943-025-02430-7
    - Glioblastoma (Darmanis et al. 2017) — doi:10.1016/j.celrep.2017.10.030
    - Follicular lymphoma — doi:10.1158/2643-3230.BCD-21-0075
    - Melanoma — GEO GSE115978 (Jerby-Arnon et al. 2018)
    - HNSCC — GEO GSE103322 (Puram et al. 2017)
    - Lung adenocarcinoma — GEO GSE131907 (Kim et al. 2020)

## Background classics (optional)

- Tirosh et al. (2016), single-cell melanoma (GEO GSE72056), and Puram et al. (2017), HNSCC. Both are early demonstrations that malignant cells cluster by patient while normal cells cluster by type, the observation behind task T4. Find them via their GEO accessions.
