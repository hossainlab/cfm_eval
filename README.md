# cfm_eval — zero-shot evaluation of single-cell foundation models on tumours

Single-cell foundation models such as scGPT, Geneformer and UCE are pretrained on millions of cells and are often claimed to produce cell embeddings that capture biology out of the box. This project asks whether that holds in cancer. Without any fine-tuning, do these embeddings distinguish malignant from non-malignant cells better than cheap baselines, and do they generalise to patients and cancer types they have never seen?

Each foundation-model embedding is compared against HVG + PCA, scVI and Harmony, and against a random-projection negative control. The comparison runs on curated tumour scRNA-seq datasets from the Curated Cancer Cell Atlas (3CA). Four tasks are evaluated:

- **Unsupervised separation:** whether malignant and normal cells separate without labels.
- **Cross-patient generalisation:** a linear probe that is tested on a held-out patient.
- **Cross-cancer transfer:** a linear probe that is tested on a held-out cancer type.
- **Patient structure:** checks that patient batch effects are removed from normal cells without erasing the patient-specific signal of malignant cells, which carry distinct genomes.

The patient, not the cell, is the unit of replication throughout. Confidence intervals come from resampling patients, and models are compared with paired tests across held-out patients.

The full research design, including hypotheses, data selection, labels, statistics and pitfalls, is in [`docs/implementation_guide.md`](docs/implementation_guide.md). External data, models and tools are listed in [`data/data_sources.csv`](data/data_sources.csv).
