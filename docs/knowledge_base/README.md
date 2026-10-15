# cfm_eval knowledge base

This folder is for teammates who are joining the project without a background in biology or single-cell genomics. It explains the domain from the ground up and then connects each idea to the code in this repository. You do not need to read the research design (`docs/implementation_guide.md`) first; this knowledge base is the on-ramp to it.

## The project in one paragraph

Tumours are mixtures of cancer cells and many kinds of normal cells. A technology called single-cell RNA sequencing measures which genes are switched on in each individual cell. Large "foundation models" (scGPT, Geneformer, UCE) have been pretrained on tens of millions of such cells, much as language models are pretrained on text, and they turn each cell into a vector (an *embedding*). This project asks a sceptical question: **without any fine-tuning, do those embeddings capture tumour biology better than cheap, decades-old methods such as PCA?** In particular, do they separate cancer cells from normal cells, and does that hold for patients and cancer types the model has never seen?

## Reading order

| # | File | What you will learn | Time |
|---|------|---------------------|------|
| 1 | [01_biology_primer.md](01_biology_primer.md) | Cells, genes, gene expression, what cancer is at the cellular level, why every patient's tumour is different | 25 min |
| 2 | [02_single_cell_rnaseq.md](02_single_cell_rnaseq.md) | How the data are produced, what a count matrix is, technology differences, noise, batch effects, where "malignant" labels come from | 25 min |
| 3 | [03_data_formats.md](03_data_formats.md) | AnnData / `.h5ad`, gene identifiers, data sources, the standardised format used in this repo | 15 min |
| 4 | [04_standard_analysis.md](04_standard_analysis.md) | The classical pipeline (normalise → HVG → PCA → neighbours → clustering) and batch correction; these are our baselines | 20 min |
| 5 | [05_foundation_models.md](05_foundation_models.md) | What single-cell foundation models are, how scGPT, Geneformer and UCE turn a cell into tokens, and their known weaknesses | 25 min |
| 6 | [06_evaluation_and_statistics.md](06_evaluation_and_statistics.md) | Linear probes, leave-one-patient-out, every metric we report, and why the *patient* is the unit of statistics | 30 min |
| 7 | [07_project_map.md](07_project_map.md) | How everything above maps to hypotheses, datasets, scripts, environments and current results | 20 min |
| — | [glossary.md](glossary.md) | One-line definitions of every term used here | reference |
| — | [reading_list.md](reading_list.md) | Papers and resources, with DOIs, in a suggested order | reference |

If you only have an hour, read files 1, 2 and 7 and keep the glossary open.

## Conventions used in these notes

- **"Cell"** always means a biological cell (one row of the data matrix), never a notebook cell.
- **"Normal"** means *non-malignant*: immune, stromal, endothelial and healthy tissue cells that live inside or near the tumour. It does not mean "healthy patient".
- Code references look like `cfm_eval/metrics.py`; commands are run from the repository root.
- Numbers quoted as "current results" come from the baseline runs in `results/` and will change as foundation-model embeddings are added. Treat them as orientation, not conclusions.

## Keeping this up to date

If you change a metric, add a dataset or add a model, update the matching file here and the status section of [07_project_map.md](07_project_map.md). If a concept confused you when you joined, add it to the glossary.
