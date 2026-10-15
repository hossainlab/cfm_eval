"""Baseline embeddings. All write to adata.obsm[<name>] and return the key."""
import numpy as np
import scanpy as sc

def _lognorm(adata):
    a = adata.copy()
    a.X = a.layers["counts"].copy()
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    return a

def hvg_pca(adata, n_hvg=2000, n_pcs=50, key="X_hvg_pca"):
    a = _lognorm(adata)
    # HVGs chosen without patient batch_key on purpose: zero-shot, label-free, no batch info
    sc.pp.highly_variable_genes(a, n_top_genes=n_hvg, flavor="seurat")
    a = a[:, a.var.highly_variable].copy()
    sc.pp.scale(a, max_value=10)
    sc.tl.pca(a, n_comps=n_pcs)
    adata.obsm[key] = a.obsm["X_pca"]
    return key

def harmony(adata, base_key="X_hvg_pca", batch="patient", key="X_harmony"):
    """NOTE: uses patient labels -> also removes real inter-patient tumour genetics. Report as such."""
    import harmonypy
    ho = harmonypy.run_harmony(adata.obsm[base_key], adata.obs, batch, verbose=False)
    Z = np.asarray(ho.Z_corr)
    adata.obsm[key] = Z.T if Z.shape[0] != adata.n_obs else Z
    return key

def random_projection(adata, dim=512, seed=0, key="X_randproj"):
    """Negative control: random linear map of log-normalised expression."""
    a = _lognorm(adata)
    R = np.random.default_rng(seed).normal(size=(a.n_vars, dim)) / np.sqrt(dim)
    adata.obsm[key] = np.asarray(a.X @ R)  # sparse @ dense: never densify the cell x gene matrix
    return key

def scvi_latent(adata, n_latent=30, max_epochs=100, key="X_scvi", batch=None):
    """Optional (pip install scvi-tools). CPU training of ~50k cells takes minutes-tens of minutes."""
    import scvi
    a = adata.copy()
    scvi.model.SCVI.setup_anndata(a, layer="counts", batch_key=batch)
    m = scvi.model.SCVI(a, n_latent=n_latent)
    m.train(max_epochs=max_epochs, accelerator="cpu")
    adata.obsm[key] = m.get_latent_representation()
    return key
