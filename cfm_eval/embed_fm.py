"""Foundation-model embedding wrappers. UNTESTED in this sandbox (weights not downloaded).
Run each once via scripts/embed_fm.py (writes .npy + .obs_names.txt), then never recompute.

scGPT : pip install scgpt ; download 'whole-human' checkpoint dir (args.json, best_model.pt, vocab.json)
        CPU path documented upstream: device="cpu", use_fast_transformer=False.
Geneformer : pip install from huggingface.co/ctheodoris/Geneformer ; needs Ensembl IDs in var['ensembl_id']
        and obs['n_counts']; tokenise -> EmbExtractor. Prefer the smallest / quantised checkpoint on a laptop.
"""
import numpy as np
import pandas as pd
from pathlib import Path

def load_cached(adata, emb_dir, dataset, model):
    """Cached embedding for `adata`'s cells, or None. Rows are matched by obs_names when
    <dataset>__<model>.obs_names.txt exists, so the cache may cover a superset of the cells
    (one file per dataset can serve both data/ready and data/ready_matched)."""
    p = Path(emb_dir) / f"{dataset}__{model}.npy"
    if not p.exists():
        return None
    E = np.load(p)
    names_p = p.with_suffix(".obs_names.txt")
    if names_p.exists():
        order = pd.Index(names_p.read_text().split())
        assert len(order) == len(E), f"{p}: {len(E)} rows but {len(order)} obs_names"
        missing = adata.obs_names.difference(order)
        assert missing.empty, f"{p}: {len(missing)} cells have no embedding"
        return E[order.get_indexer(adata.obs_names)]
    assert len(E) == adata.n_obs, f"{p}: no obs_names file and row count != n_obs"
    return E

def save_cached(E, obs_names, emb_dir, dataset, model):
    p = Path(emb_dir) / f"{dataset}__{model}.npy"
    p.parent.mkdir(parents=True, exist_ok=True)
    np.save(p, np.asarray(E, dtype=np.float32))
    p.with_suffix(".obs_names.txt").write_text("\n".join(map(str, obs_names)) + "\n")
    return p

def gene_match(adata, vocab, gene_col):
    """Fraction of the dataset's genes found in a model vocabulary (low overlap silently degrades FMs)."""
    g = adata.var[gene_col] if gene_col in adata.var else adata.var_names.to_series()
    return float(g.dropna().astype(str).isin(set(vocab)).mean())

def scgpt_embed(adata, model_dir, gene_col="feature_name", batch_size=32, device="cpu", seed=0, chunk_size=5000):
    """Cells with >1200 expressed genes are randomly subsampled and binning breaks ties randomly: seed both."""
    import os, torch
    import scgpt as scg
    np.random.seed(seed); torch.manual_seed(seed)
    if not hasattr(os, "sched_getaffinity"):  # macOS: scGPT sizes its DataLoader with this; empty set -> num_workers=0
        os.sched_getaffinity = lambda pid: set()
    a = adata.copy()
    a.X = a.layers["counts"]
    if gene_col not in a.var:
        a.var[gene_col] = a.var_names
    # Chunked: embed_data densifies its whole input and MPS caches memory across batches, so 46k cells
    # ran out of unified memory on a 24 GB Mac. Each cell's embedding depends only on its own counts.
    E = []
    for i in range(0, a.n_obs, chunk_size):
        out = scg.tasks.embed_data(a[i:i + chunk_size].copy(), model_dir, gene_col=gene_col, batch_size=batch_size,
                                   device=device, use_fast_transformer=False, return_new_adata=True)
        E.append(np.asarray(out.X))
        if device == "mps":
            torch.mps.empty_cache()
    return np.vstack(E)

def geneformer_embed(adata, model_dir, workdir, emb_layer=-1, batch_size=8, device="cpu", nproc=4):
    """Geneformer V2 CLS embedding. Tokenisation is Geneformer's own TranscriptomeTokenizer (median-scaled rank
    encoding); the forward pass reproduces EmbExtractor(emb_mode="cls", emb_layer=-1), whose code hard-codes CUDA:
    CLS token of hidden_states[n_layers + emb_layer], i.e. the penultimate layer by default."""
    import shutil, torch
    from datasets import load_from_disk
    from transformers import BertForMaskedLM
    from geneformer import TranscriptomeTokenizer
    w = Path(workdir); shutil.rmtree(w, ignore_errors=True); (w / "h5ad").mkdir(parents=True)
    a = adata[:, adata.var["ensembl_id"].notna().values].copy()  # GEO sets map only 65-86% of symbols to Ensembl
    a.X = a.layers["counts"]; a.layers.clear()
    a.var = a.var[["ensembl_id"]].astype(str)
    a.obs = pd.DataFrame({"n_counts": np.asarray(a.X.sum(1)).ravel(), "cell_idx": np.arange(a.n_obs)},
                         index=a.obs_names)
    a.write_h5ad(w / "h5ad" / "data.h5ad")
    tok = TranscriptomeTokenizer({"cell_idx": "cell_idx"}, nproc=nproc, model_version="V2")
    print(f"  Geneformer token match (ensembl_id): {a.var['ensembl_id'].isin(set(tok.gene_token_dict)).mean():.3f} "
          f"of mapped genes; {a.n_vars / adata.n_vars:.3f} of genes have an Ensembl ID", flush=True)
    tok.tokenize_data(w / "h5ad", w, "tok", file_format="h5ad")
    ds = load_from_disk(str(w / "tok.dataset"))
    assert sorted(ds["cell_idx"]) == list(range(adata.n_obs)), "tokeniser dropped cells"
    pad, cls = tok.gene_token_dict["<pad>"], tok.gene_token_dict["<cls>"]
    model = BertForMaskedLM.from_pretrained(model_dir, output_hidden_states=True).to(device).eval()
    layer = model.config.num_hidden_layers + emb_layer
    order = np.argsort(-np.asarray(ds["length"]))  # length-sorted batches: little padding
    E = np.zeros((adata.n_obs, model.config.hidden_size), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(order), batch_size):
            b = ds.select(order[i:i + batch_size].tolist())
            seqs = b["input_ids"]; L = max(map(len, seqs))
            x = torch.full((len(seqs), L), pad, dtype=torch.long); m = torch.zeros_like(x)
            for j, sq in enumerate(seqs):
                x[j, :len(sq)] = torch.as_tensor(sq); m[j, :len(sq)] = 1
            assert (x[:, 0] == cls).all(), "first token is not <cls>"
            h = model(input_ids=x.to(device), attention_mask=m.to(device)).hidden_states[layer]
            E[np.asarray(b["cell_idx"])] = h[:, 0].float().cpu().numpy()
    return E

def _scgpt_model(model_dir):
    """The TransformerModel exactly as scgpt.tasks.embed_data builds it (use_fast_transformer=False)."""
    import json
    from scgpt.model import TransformerModel
    from scgpt.tokenizer import GeneVocab
    vocab = GeneVocab.from_file(Path(model_dir) / "vocab.json")
    for s in ["<pad>", "<cls>", "<eoc>"]:
        if s not in vocab:
            vocab.append_token(s)
    c = json.load(open(Path(model_dir) / "args.json"))
    return TransformerModel(ntoken=len(vocab), d_model=c["embsize"], nhead=c["nheads"], d_hid=c["d_hid"],
                            nlayers=c["nlayers"], nlayers_cls=c["n_layers_cls"], n_cls=1, vocab=vocab,
                            dropout=c["dropout"], pad_token=c["pad_token"], pad_value=c["pad_value"], do_mvc=True,
                            do_dab=False, use_batch_labels=False, domain_spec_batchnorm=False,
                            explicit_zero_prob=False, use_fast_transformer=False, pre_norm=False)

def scgpt_checkpoint_coverage(model_dir):
    """Fraction of model parameters that embed_data's non-strict load actually fills from best_model.pt."""
    import torch
    sd = _scgpt_model(model_dir).state_dict()
    pre = {k.replace("Wqkv.", "in_proj_"): v for k, v in torch.load(Path(model_dir) / "best_model.pt", map_location="cpu").items()}
    hit = [k for k, v in sd.items() if k in pre and pre[k].shape == v.shape]
    return sum(sd[k].numel() for k in hit) / sum(v.numel() for v in sd.values()), sorted(set(sd) - set(hit))

def make_random_scgpt(model_dir, out_dir, seed=0):
    """H5 control: same architecture, vocab and config as the pretrained checkpoint, default (random) init."""
    import shutil, torch
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    for f in ["vocab.json", "args.json"]:
        shutil.copy(Path(model_dir) / f, out / f)
    torch.manual_seed(seed)
    torch.save(_scgpt_model(model_dir).state_dict(), out / "best_model.pt")
    return out
