"""Load and standardise tumour scRNA-seq AnnData objects.

Required obs columns after standardise(): patient, cancer_type, malignant (bool), cell_type.
adata.X must hold RAW counts (foundation models tokenise raw/normalised counts themselves).
"""
import numpy as np
import scanpy as sc

REQUIRED = ["patient", "cancer_type", "malignant", "cell_type"]

def standardise(adata, patient_col, malignant_col, cell_type_col, cancer_type,
                malignant_values=("malignant", "Malignant", True, 1, "yes")):
    a = adata.copy()
    a.obs["patient"] = a.obs[patient_col].astype(str).values
    a.obs["malignant"] = a.obs[malignant_col].isin(list(malignant_values)).values
    a.obs["cell_type"] = a.obs[cell_type_col].astype(str).values
    a.obs["cancer_type"] = cancer_type
    a.layers["counts"] = a.X.copy()
    return a

def qc(adata, min_genes=200, max_mt=20.0):
    a = adata.copy()
    a.var["mt"] = a.var_names.str.upper().str.startswith("MT-")
    sc.pp.calculate_qc_metrics(a, qc_vars=["mt"], inplace=True, percent_top=None)
    keep = (a.obs["n_genes_by_counts"] >= min_genes) & (a.obs["pct_counts_mt"] <= max_mt)
    return a[keep].copy()

def eligible_patients(adata, min_malignant=50, min_normal=50):
    """Patients with enough cells of both classes for leave-patient-out evaluation."""
    t = adata.obs.groupby("patient", observed=True)["malignant"].agg(["sum", "count"])
    t["normal"] = t["count"] - t["sum"]
    return t.index[(t["sum"] >= min_malignant) & (t["normal"] >= min_normal)].tolist()

def subsample_per_patient(adata, max_cells=2000, seed=0):
    """Cap cells per patient (stratified by malignant) so no patient dominates."""
    rng = np.random.default_rng(seed)
    idx = []
    for (_, _), g in adata.obs.groupby(["patient", "malignant"], observed=True):
        n = min(len(g), max_cells // 2)
        idx.extend(rng.choice(g.index.values, n, replace=False))
    return adata[adata.obs_names.isin(idx)].copy()

def pretrain_overlap(dataset):
    """Pretraining-leakage group: CELLxGENE datasets may be in scGPT/Geneformer pretraining corpora."""
    return "possibly_seen" if dataset.startswith("cxg_") else "unknown"

def write_std(adata, path):
    """Write a standardised object without duplicating counts (X holds counts; layer restored on read)."""
    a = adata.copy()
    if "counts" in a.layers:
        a.X = a.layers["counts"]; del a.layers["counts"]
    a.write_h5ad(path, compression="gzip")

def read_std(path):
    import anndata as ad
    a = ad.read_h5ad(path)
    a.layers["counts"] = a.X.copy()
    return a

def make_ready(adata, max_cells=2000, min_malignant=50, min_normal=50, seed=0):
    """Benchmark-ready subset: eligible patients only, capped per patient (stratified by malignant)."""
    el = eligible_patients(adata, min_malignant, min_normal)
    a = adata[adata.obs["patient"].isin(el)].copy()
    for c in a.obs.select_dtypes("category"):
        a.obs[c] = a.obs[c].cat.remove_unused_categories()
    a = subsample_per_patient(a, max_cells, seed)
    a.uns.setdefault("provenance", {})
    a.uns["provenance"].update(ready_rule=f"patients with >={min_malignant} malignant and >={min_normal} normal; "
                                          f"<= {max_cells//2} cells per patient x class; seed={seed}")
    return a

# Lineage-matched normal counterparts: the HARD task (malignant vs same-lineage normal).
# Malignant-vs-all-normal is near ceiling (AUROC ~0.96-1.0 even for random projection on 6 datasets).
MATCHED_NORMAL = {
    "cxg_follicular_lymphoma": ["B cell"],
    "cxg_gbm_darmanis": ["astrocyte", "oligodendrocyte precursor cell", "oligodendrocyte"],
    "cxg_breast_ctm2": ["mesothelial cell"],
    "cxg_crlm_bdrhapsody": ["cholangiocyte", "hepatocyte"],
}

def make_ready_matched(adata, normal_labels, max_cells=2000, min_malignant=20, min_normal=20, seed=0):
    """HARD-task subset: malignant cells + same-lineage normal cells only, per eligible patient."""
    keep = adata.obs["malignant"].values | adata.obs["cell_type"].astype(str).isin(normal_labels).values
    a = adata[keep].copy()
    a = make_ready(a, max_cells, min_malignant, min_normal, seed)
    a.uns["provenance"]["task"] = f"lineage-matched: malignant vs {normal_labels}"
    return a
