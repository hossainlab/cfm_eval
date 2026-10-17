# 5. Single-cell foundation models

## 5.1 The idea, by analogy with language models

| Language model | Single-cell foundation model (scFM) |
|---|---|
| Token = word piece | Token = gene (plus, depending on the model, its expression level) |
| Sentence/document = sequence of tokens | One **cell** = a set or sequence of its expressed genes |
| Pretraining corpus = web text | Tens of millions of cells from public atlases |
| Objective = predict masked or next words | Predict masked genes or their expression from the rest of the cell |
| Sentence embedding (e.g. `[CLS]` vector) | **Cell embedding**: one vector per cell |

The hope: by learning which genes co-occur across millions of cells and tissues, the model learns general "rules" of cell biology, and its cell embeddings capture cell state better than PCA on one dataset.

**Zero-shot** means we use the pretrained model as-is: feed cells in, take the embedding out, no training on our data. **Fine-tuning** (updating model weights on labelled data) is deliberately out of scope (guide §4). The question is what the pretrained representation already contains, and fine-tuning would also need a GPU budget we do not have.

The tricky part compared with text: a cell is not naturally a sequence. Genes have no inherent order, and each gene carries a continuous value. Each model solves this differently, and that choice matters.

## 5.2 The three models

### scGPT (Cui et al. 2024, *Nature Methods*)

- **Pretraining data**: about 33 million human cells from CELLxGENE Census. The whole-human checkpoint's `args.json` points to a May 2023 Census snapshot.
- **Tokenisation**: each token = gene identity embedding + **binned expression value**. Per cell, non-zero expression values are split into 51 bins (`n_bins: 51`), so the value is relative within the cell. At most 1,200 genes are used per cell (`max_seq_len: 1200`); cells with more expressed genes are **randomly subsampled**, which is why we fix a seed.
- **Architecture**: transformer, 12 layers, 8 heads, 512-dimensional embedding.
- **Pretraining objective**: generative masked expression prediction, including predicting gene expression from the cell embedding.
- **Cell embedding**: the 512-d vector at the `<cls>` position (`embed_data(..., cell_embedding_mode="cls")`).
- **Gene IDs**: symbols, about 60,700-token vocabulary. Coverage in our data ranges from 0.58 (CRLM) to 0.89 (LUAD).

### Geneformer (Theodoris et al. 2023, *Nature*; V2 2024)

- **Pretraining data**: V2 uses about 104 million human cells (Genecorpus-104M). **The authors excluded malignant cells and cell lines from pretraining** because of their rewired genomes. So Geneformer has, by design, never seen a cancer cell. (A separate "CLcancer" variant was continually pretrained on about 14 million cancer cells; we do not use it in the main comparison.)
- **Tokenisation: rank-value encoding.** Each gene's count is divided by that gene's median across the whole corpus, then genes are *sorted* by this scaled value. The cell becomes a sequence of gene IDs, highest-ranked first. The absolute values are discarded. This down-weights ubiquitous housekeeping genes and up-weights genes that are unusually high in this cell. It is also more robust to technical scaling.
- **Architecture**: BERT encoder, 12 layers, 768-dimensional, input up to 4,096 genes (V2-104M).
- **Pretraining objective**: masked gene prediction (15 % masked).
- **Cell embedding**: `<cls>` token of the penultimate layer (our implementation, `cfm_eval.embed_fm.geneformer_embed`, reproduces Geneformer's `EmbExtractor`, whose code only runs on CUDA).
- **Gene IDs**: Ensembl, about 20,000 protein-coding genes. GEO datasets lose genes without an Ensembl mapping.

### UCE, Universal Cell Embeddings (Rosen et al. 2023, bioRxiv)

- **Tokenisation**: genes are represented by **protein language-model embeddings** (ESM2) of the protein they encode, rather than a learned ID. This makes the model species-agnostic. Genes are sampled in proportion to expression and ordered by chromosome position.
- **Sizes**: 4-layer and 33-layer variants. Treated as a stretch goal (heavier compute).

## 5.3 Known issues to keep in mind

1. **Zero-shot often loses to baselines.** Independent evaluations (Kedzierska et al. 2025; Wu et al. 2025) found scGPT and Geneformer zero-shot embeddings frequently do no better than HVG+PCA, scVI or Harmony on clustering and batch integration. A null result is plausible and still publishable (guide §10: "no significant difference across N patients is a result, not a failure").
2. **Input conventions are fragile.** Pass raw counts, never log-transformed data. Use the right gene identifier. A model fed the wrong input fails silently, with plausible-looking but degraded embeddings. Our wrappers log the vocabulary match and, for scGPT, how many parameters the checkpoint actually loads (`scgpt_checkpoint_coverage`). scGPT loads weights non-strictly and would silently leave mismatched layers random.
3. **Pretraining leakage.** CELLxGENE datasets may be in the pretraining corpus. Report results split by `pretrain_overlap`.
4. **Non-determinism.** scGPT samples genes randomly for cells with more than 1,200 expressed genes; Leiden clustering is seeded. Fix seeds and repeat over at least 3 seeds.
5. **Compute.** On an Apple M4 Pro, scGPT runs at about 46 s per 1,000 cells on the GPU (MPS) and about 160 s on CPU. Always time 1,000 cells before launching a full dataset (guide §4).

## 5.4 Practical setup in this repo

scGPT and Geneformer pin old or conflicting library versions, so they live **outside** the main uv environment:

| Env | Path | Key versions | Used for |
|---|---|---|---|
| main | `.venv` (uv) | Python 3.11, scanpy | baselines, metrics, all analysis |
| scGPT | `../cfm_eval_fm/.venv` | torch 2.3.1, torchtext 0.18, numpy < 2, scgpt 0.2.4 (`--no-deps`) | `scgpt`, `scgpt_random` |
| Geneformer | `../cfm_eval_fm/.venv_gf` | transformers < 5, datasets < 4, geneformer (editable from HF repo) | `geneformer` |

Weights live in `../cfm_eval_fm/weights/` (scGPT) and `../cfm_eval_fm/geneformer_repo/Geneformer-V2-104M/`. Only the cached `.npy` embeddings come back into this repo. To embed everything with one model:

```bash
scripts/embed_all.sh ../cfm_eval_fm/.venv/bin/python scgpt ../cfm_eval_fm/weights/scGPT_human scgpt --device mps
```

The script skips datasets that are already cached, so it is safe to interrupt and restart. On macOS, `PYTORCH_ENABLE_MPS_FALLBACK=1` is set automatically because one PyTorch operation scGPT uses is not implemented on MPS.

## 5.5 Check your understanding

1. Why might Geneformer be at a disadvantage on malignant cells specifically? *(It never saw malignant cells during pretraining.)*
2. Why does scGPT need a fixed seed even at inference time? *(Random gene subsampling above 1,200 genes, plus random tie-breaking in binning.)*
3. What would it mean if `scgpt_random` scored as well as `scgpt`? *(The benefit comes from architecture or tokenisation, not pretraining: H5.)*

Next: [06_evaluation_and_statistics.md](06_evaluation_and_statistics.md)
