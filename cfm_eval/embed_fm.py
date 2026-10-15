"""Foundation-model embedding wrappers. UNTESTED in this sandbox (weights not downloaded).
Run each once, cache to .npy, then never recompute.

scGPT : pip install scgpt ; download 'whole-human' checkpoint dir (args.json, best_model.pt, vocab.json)
        CPU path documented upstream: device="cpu", use_fast_transformer=False.
Geneformer : pip install from huggingface.co/ctheodoris/Geneformer ; needs Ensembl IDs in var['ensembl_id']
        and obs['n_counts']; tokenise -> EmbExtractor. Prefer the smallest / quantised checkpoint on a laptop.
"""
import numpy as np
from pathlib import Path

def cached(fn):
    def wrap(adata, cache_path, *a, **k):
        p = Path(cache_path)
        if p.exists():
            E = np.load(p)
        else:
            E = fn(adata, *a, **k)
            p.parent.mkdir(parents=True, exist_ok=True)
            np.save(p, E)
        assert E.shape[0] == adata.n_obs, "embedding/cell count mismatch: check cell order"
        return E
    return wrap

@cached
def scgpt_embed(adata, model_dir, gene_col="feature_name", batch_size=32, device="cpu"):
    import scgpt as scg
    a = adata.copy()
    a.X = a.layers["counts"]
    if gene_col not in a.var:
        a.var[gene_col] = a.var_names
    out = scg.tasks.embed_data(a, model_dir, gene_col=gene_col, batch_size=batch_size,
                               device=device, use_fast_transformer=False, return_new_adata=True)
    return np.asarray(out.X)

@cached
def geneformer_embed(adata, model_dir, workdir, model_version="V2", emb_layer=-1, forward_batch_size=16):
    """Sketch of the Geneformer path (check argument names against your installed version)."""
    from geneformer import TranscriptomeTokenizer, EmbExtractor
    w = Path(workdir); (w / "h5ad").mkdir(parents=True, exist_ok=True)
    a = adata.copy(); a.X = a.layers["counts"]
    a.obs["n_counts"] = np.asarray(a.X.sum(1)).ravel()
    a.obs["cell_idx"] = np.arange(a.n_obs)
    a.write_h5ad(w / "h5ad" / "data.h5ad")
    TranscriptomeTokenizer(custom_attr_name_dict={"cell_idx": "cell_idx"},
                           model_version=model_version).tokenize_data(
        str(w / "h5ad"), str(w), "tok", file_format="h5ad")
    ex = EmbExtractor(model_type="Pretrained", emb_mode="cell", max_ncells=None,
                      emb_layer=emb_layer, emb_label=["cell_idx"],
                      forward_batch_size=forward_batch_size, model_version=model_version)
    df = ex.extract_embs(model_dir, str(w / "tok.dataset"), str(w), "embs")
    df = df.sort_values("cell_idx")
    return df.drop(columns=["cell_idx"]).to_numpy()
