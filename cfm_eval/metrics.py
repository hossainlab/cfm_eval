"""Evaluation metrics. Patient is the unit of replication throughout."""
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             balanced_accuracy_score, silhouette_score, adjusted_rand_score)
from sklearn.neighbors import NearestNeighbors

def knn_purity(E, labels, k=15):
    nn = NearestNeighbors(n_neighbors=k + 1).fit(E)
    idx = nn.kneighbors(E, return_distance=False)[:, 1:]
    lab = np.asarray(labels)
    return float((lab[idx] == lab[:, None]).mean())

def unsupervised(adata, key, k=15, resolution=0.5, seed=0):
    """T1: does the embedding separate malignant from normal without any labels?"""
    E = adata.obsm[key]; y = adata.obs["malignant"].values
    a = sc.AnnData(np.zeros((adata.n_obs, 1)), obs=adata.obs[["malignant"]].copy())
    a.obsm["E"] = E
    sc.pp.neighbors(a, use_rep="E", n_neighbors=k, random_state=seed)
    sc.tl.leiden(a, resolution=resolution, random_state=seed, flavor="igraph",
                 n_iterations=2, directed=False)
    # majority-vote cluster label -> ARI vs malignant
    cl = a.obs["leiden"].astype(str)
    maj = pd.Series(y).groupby(cl.values).mean() > 0.5
    return {"knn_purity": knn_purity(E, y, k),
            "asw_malignant": float(silhouette_score(E, y, sample_size=min(5000, len(y)), random_state=seed)),
            "leiden_ari": float(adjusted_rand_score(y, cl.map(maj).values))}

def _subsample_train(y, p, tr, n_per_class, rng):
    """Few-shot: keep at most n cells per (training patient, class)."""
    idx = np.flatnonzero(tr)
    keep = [rng.choice(g, min(len(g), n_per_class), replace=False)
            for g in pd.Series(idx).groupby([p[idx], y[idx]]).apply(lambda s: s.values)]
    m = np.zeros_like(tr); m[np.concatenate(keep)] = True
    return m

def lopo_probe(adata, key, C=1.0, seed=0, n_train_per_class=None):
    """T2: leave-one-patient-out linear probe. Returns per-patient metrics.
    n_train_per_class: few-shot variant, at most n cells per (training patient, class); the full probe is near ceiling."""
    E = adata.obsm[key]; y = adata.obs["malignant"].values.astype(int); p = adata.obs["patient"].astype(str).values
    rng = np.random.default_rng(seed)
    rows = []
    for pt in np.unique(p):
        te = p == pt; tr = ~te
        if n_train_per_class:
            tr = _subsample_train(y, p, tr, n_train_per_class, rng)
        if len(np.unique(y[te])) < 2 or len(np.unique(y[tr])) < 2:
            continue
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000, random_state=seed))
        clf.fit(E[tr], y[tr])
        s = clf.predict_proba(E[te])[:, 1]
        rows.append({"patient": pt, "auroc": roc_auc_score(y[te], s),
                     "auprc": average_precision_score(y[te], s),
                     "bal_acc": balanced_accuracy_score(y[te], s > 0.5),
                     "frac_malignant": y[te].mean()})
    return pd.DataFrame(rows)

def cross_cancer_probe(adata, key, C=1.0, seed=0):
    """T3: train on all cancer types but one, test on the held-out type.
    One row per held-out patient (the unit of replication); auroc_pooled is over all held-out cells."""
    E = adata.obsm[key]; y = adata.obs["malignant"].values.astype(int); c = adata.obs["cancer_type"].astype(str).values
    p = adata.obs["patient"].astype(str).values
    rows = []
    for ct in np.unique(c):
        te = c == ct
        if te.all() or len(np.unique(y[te])) < 2:
            continue
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000, random_state=seed))
        clf.fit(E[~te], y[~te]); s = clf.predict_proba(E[te])[:, 1]
        pooled = roc_auc_score(y[te], s)
        for pt in np.unique(p[te]):
            m = p[te] == pt
            if len(np.unique(y[te][m])) < 2:
                continue
            rows.append({"held_out_cancer": ct, "patient": pt, "auroc": roc_auc_score(y[te][m], s[m]),
                         "bal_acc": balanced_accuracy_score(y[te][m], s[m] > 0.5), "auroc_pooled": pooled})
    return pd.DataFrame(rows)

def patient_structure(adata, key, k=15):
    """T4: patient mixing, reported SEPARATELY for normal and malignant cells.
    Normal cells should mix across patients (batch effect removed); malignant cells are expected
    to stay patient-specific (distinct genomes). An embedding that mixes malignant cells is NOT
    automatically better - it may be erasing real tumour genetics."""
    out = {}
    for name, mask in [("normal", ~adata.obs["malignant"].values), ("malignant", adata.obs["malignant"].values)]:
        E = adata.obsm[key][mask]; p = adata.obs["patient"].astype(str).values[mask]
        if len(np.unique(p)) < 2:
            continue
        nn = NearestNeighbors(n_neighbors=k + 1).fit(E)
        idx = nn.kneighbors(E, return_distance=False)[:, 1:]
        same = (p[idx] == p[:, None]).mean(1)
        expected = pd.Series(p).map(pd.Series(p).value_counts(normalize=True)).values.astype(float)
        out[f"{name}_same_patient_frac"] = float(same.mean())
        out[f"{name}_same_patient_excess"] = float((same - expected).mean())  # 0 = perfectly mixed
    return out

def patient_bootstrap(per_patient, col="auroc", n=2000, seed=0):
    """CI by resampling patients (not cells)."""
    v = per_patient[col].dropna().values
    rng = np.random.default_rng(seed)
    bs = rng.choice(v, (n, len(v)), replace=True).mean(1)
    return float(v.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))

def paired_compare(pp_a, pp_b, col="auroc", n=2000, seed=0):
    """Paired Wilcoxon across held-out patients: model A vs model B.
    Effect size = median and mean difference (A - B), with a patient-bootstrap CI of the mean."""
    from scipy.stats import wilcoxon
    m = pp_a.merge(pp_b, on="patient", suffixes=("_a", "_b"))
    d = (m[f"{col}_a"] - m[f"{col}_b"]).values
    stat = wilcoxon(d) if (d != 0).any() else None
    bs = np.random.default_rng(seed).choice(d, (n, len(d)), replace=True).mean(1)
    return {"n_patients": len(m), "median_diff": float(np.median(d)), "mean_diff": float(d.mean()),
            "mean_diff_lo": float(np.percentile(bs, 2.5)), "mean_diff_hi": float(np.percentile(bs, 97.5)),
            "p_wilcoxon": float(stat.pvalue) if stat else 1.0}

def holm(pvals):
    """Holm step-down adjusted p-values (family = the comparisons passed in)."""
    p = np.asarray(pvals, float); o = np.argsort(p); m = len(p)
    adj = np.maximum.accumulate((m - np.arange(m)) * p[o]).clip(max=1.0)
    out = np.empty(m); out[o] = adj
    return out

def flip_labels(adata, frac, seed=0):
    """Label-noise sensitivity: flip the malignant label of a random `frac` of cells within each patient."""
    a = adata.copy()
    if frac <= 0:
        return a
    rng = np.random.default_rng(seed)
    y = a.obs["malignant"].values.astype(bool).copy()
    for _, idx in pd.Series(np.arange(a.n_obs)).groupby(a.obs["patient"].astype(str).values):
        f = rng.choice(idx.values, int(round(frac * len(idx))), replace=False)
        y[f] = ~y[f]
    a.obs["malignant"] = y
    return a
