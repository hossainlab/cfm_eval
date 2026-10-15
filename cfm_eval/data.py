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
    t = adata.obs.groupby("patient")["malignant"].agg(["sum", "count"])
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
