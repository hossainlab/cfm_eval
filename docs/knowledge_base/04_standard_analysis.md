# 4. The standard analysis pipeline (our baselines)

Foundation models are only interesting if they beat what people already do. This chapter explains the classical scRNA-seq pipeline, because its pieces are exactly our baselines (`cfm_eval/baselines.py`).

## 4.1 The classical pipeline

```
raw counts
  │ 1. normalise library size     sc.pp.normalize_total(target_sum=1e4)
  │ 2. log-transform              sc.pp.log1p            -> log(1 + x)
  │ 3. highly variable genes      sc.pp.highly_variable_genes(n_top_genes=2000)
  │ 4. scale                      sc.pp.scale(max_value=10)  -> z-score per gene, clipped
  │ 5. PCA                        sc.tl.pca(n_comps=50)      -> X_hvg_pca  (our main baseline)
  │ 6. kNN graph                  sc.pp.neighbors(k=15)
  │ 7. cluster                    sc.tl.leiden
  ▼ 8. visualise                  sc.tl.umap
```

**1. Library-size normalisation.** Divide each cell's counts by its total and multiply by 10,000 ("counts per 10k"). This removes the technical difference in how many molecules were captured per cell.

**2. Log transform.** `log(1 + x)` compresses the huge dynamic range (a few genes have thousands of counts, most have 0–5) and makes variance more comparable across genes. The `+1` keeps zeros at zero.

**3. Highly variable genes (HVGs).** Most genes are either off everywhere or constant. Keeping the ~2,000 genes whose variance across cells is highest relative to their mean concentrates the biological signal and cuts noise. In this project HVGs are selected **without** a patient batch key on purpose: the baseline must be label-free, like a zero-shot model.

**4. Scaling.** Each gene is standardised to mean 0 and variance 1 (clipped at 10), so highly expressed genes do not dominate PCA.

**5. PCA.** Principal component analysis projects 2,000 genes onto the 50 orthogonal directions of greatest variance. The result is a 50-dimensional **embedding** per cell: `X_hvg_pca`. This is the main baseline every foundation model must beat (hypotheses H1 and H2).

**6. k-nearest-neighbour graph.** Connect each cell to its *k* most similar cells in the embedding (Euclidean distance). Nearly every downstream method (clustering, UMAP, our kNN-purity and T4 metrics) operates on this graph.

**7. Leiden clustering.** A community-detection algorithm that partitions the kNN graph into densely connected groups. The `resolution` parameter controls how many clusters you get. It is stochastic, so results vary with the seed; we repeat over seeds.

**8. UMAP.** A 2-D layout for *visualisation only*. Distances in UMAP plots are distorted; never compute metrics on UMAP coordinates.

## 4.2 Batch correction and integration

When cells cluster by patient or batch instead of biology, **integration** methods try to remove the batch signal.

**Harmony** (`baselines.harmony`). Takes the PCA embedding and the batch labels (here: patient), and iteratively shifts cells so that every cluster contains a fair mix of batches. Fast, popular. Two important consequences for us:

- It **uses patient labels**, so it is not label-free. We report it as a separate category from the zero-shot and label-free embeddings.
- It cannot distinguish technical patient effects from real tumour genetics. It pulls malignant cells from different patients together too. In our current results, Harmony lowers *normal* same-patient excess from 0.40 to 0.11 (good) but also lowers *malignant* excess from 0.77 to 0.34 (possible loss of real biology). Exposing this trade-off is the purpose of task T4.

**scVI** (`baselines.scvi_latent`, optional extra, not yet run). A variational autoencoder trained on raw counts with a negative-binomial likelihood. It learns a low-dimensional latent space per dataset. It is a *learned but not pretrained* baseline: it answers "is a neural model trained on this dataset alone as good as a foundation model pretrained on millions of cells?" We run it without a batch key so that it stays label-free.

## 4.3 Negative controls

A **negative control** is a method that should *not* work well if our hypothesis is right. If it does work well, the task is too easy or the metric is broken.

**Random projection** (`baselines.random_projection`). Multiply log-normalised expression by a random Gaussian matrix to get 512 dimensions, the same size as scGPT's embedding. There is no learning at all. It tests "how much does dimensionality alone buy?" (H5). Surprising but true: random projections approximately preserve distances (Johnson–Lindenstrauss lemma), so they can retain a lot of signal.

**Random-weight foundation model** (`scgpt_random`). The exact scGPT architecture and tokenisation, but with freshly initialised, *untrained* weights (`cfm_eval.embed_fm.make_random_scgpt`). If pretrained scGPT is not better than its random-weight twin, the gains come from the architecture or tokenisation, not from pretraining (H5).

## 4.4 Why cheap baselines are hard to beat

Several published evaluations (e.g. Kedzierska et al. 2025, see reading list) found that zero-shot foundation-model embeddings often **match or underperform** HVG+PCA, scVI or Harmony on clustering and integration. Reasons include: the tasks are easy (cell types differ massively in expression), the models were not trained with these tasks as an objective, and pretraining data differ from the evaluation data. Our current baseline numbers show the same ceiling effect: **malignant-vs-all-normal AUROC is 0.99 even for random projection.** That is why the project emphasises harder settings: lineage-matched normals, few-shot probes, cross-cancer transfer, and unsupervised metrics.

## 4.5 Check your understanding

1. Why select HVGs without a batch key here, when many tutorials use one? *(The baseline must be label-free, like the zero-shot FM.)*
2. Why is Harmony reported in a separate category? *(It uses patient labels.)*
3. If random projection gets AUROC 0.99, what does that tell you about the task? *(The task is at ceiling and cannot rank models.)*

Next: [05_foundation_models.md](05_foundation_models.md)
