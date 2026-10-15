"""Loaders that turn public tumour scRNA-seq releases into standardised AnnData.

Every loader returns an AnnData with:
  X / layers['counts'] : counts (10x/BD: raw UMI; Smart-seq2 GEO: TPM back-transformed from log2(TPM/10+1))
  obs : patient, malignant (bool), cell_type, cancer_type, sample, source, plus source-specific columns
  var : index = gene symbol; var['feature_name'] = symbol; var['ensembl_id'] where known
  uns['provenance'] : accession, files, value_type, label_source, and dropped-cell counts
GEO raw files: https://ftp.ncbi.nlm.nih.gov/geo/series/<GSExxxnnn>/<GSE>/suppl/
"""
import gzip, io, re
import numpy as np, pandas as pd, scipy.sparse as sp, anndata as ad

GEO = "https://ftp.ncbi.nlm.nih.gov/geo/series/{p}/{a}/{kind}/{f}"
def geo_url(acc, fname, kind="suppl"):
    return GEO.format(p=acc[:-3] + "nnn", a=acc, kind=kind, f=fname)

FILES = {
    "GSE115978": ["GSE115978_cell.annotations.csv.gz", "GSE115978_counts.csv.gz"],
    "GSE72056": ["GSE72056_melanoma_single_cell_revised_v2.txt.gz"],
    "GSE103322": ["GSE103322_HNSCC_all_data.txt.gz"],
    "GSE131907": ["GSE131907_Lung_Cancer_cell_annotation.txt.gz", "GSE131907_Lung_Cancer_raw_UMI_matrix.txt.gz"],
}

def _finish(X, cells, genes, obs, prov):
    genes = pd.Index([g.strip().strip("'\"") for g in genes])
    a = ad.AnnData(sp.csr_matrix(X, dtype=np.float32), obs=obs.set_index(pd.Index(cells, name=None)),
                   var=pd.DataFrame(index=genes))
    a.var_names_make_unique()
    a.var["feature_name"] = a.var_names
    a.layers["counts"] = a.X.copy()
    a.obs["malignant"] = a.obs["malignant"].astype(bool)
    for c in ["patient", "cell_type", "cancer_type", "sample", "source"]:
        a.obs[c] = a.obs[c].astype(str).astype("category")
    a.uns["provenance"] = prov
    return a

def _unlog_tpm(M):
    """GEO Smart-seq2 releases (Tirosh 2016, Puram 2017) store log2(TPM/10 + 1)."""
    return (np.power(2.0, M) - 1.0) * 10.0

# ---------------------------------------------------------------- GSE115978
def load_gse115978(raw_dir):
    """Jerby-Arnon 2018 melanoma (Smart-seq2). Labels: cell.types == 'Mal'. '?' cells dropped."""
    annp = f"{raw_dir}/GSE115978_cell.annotations.csv.gz"
    ann = pd.read_csv(annp)
    M = pd.read_csv(f"{raw_dir}/GSE115978_counts.csv.gz", index_col=0)
    M = M[ann["cells"]]  # align
    keep = (ann["cell.types"] != "?").values
    ann = ann[keep]; M = M.loc[:, keep]
    obs = pd.DataFrame({"patient": ann["samples"].values, "sample": ann["samples"].values,
                        "malignant": (ann["cell.types"] == "Mal").values,
                        "cell_type": ann["cell.types"].values, "cancer_type": "melanoma",
                        "source": "GEO:GSE115978", "treatment_group": ann["treatment.group"].values,
                        "cohort": ann["Cohort"].values})
    prov = dict(accession="GSE115978", study="Jerby-Arnon et al. 2018 Cell", assay="Smart-seq2",
                value_type="counts as released (GSE115978_counts.csv.gz)",
                label_source="author cell.types == 'Mal'", dropped_unresolved=int((~keep).sum()))
    return _finish(M.T.values, M.columns, M.index, obs, prov)

# ---------------------------------------------------------------- GSE72056
def load_gse72056(raw_dir):
    """Tirosh 2016 melanoma (Smart-seq2). Row 3: malignant (1=no, 2=yes, 0=unresolved); 0 dropped."""
    D = pd.read_csv(f"{raw_dir}/GSE72056_melanoma_single_cell_revised_v2.txt.gz", sep="\t", index_col=0)
    meta, expr = D.iloc[:3], D.iloc[3:]
    tumour = meta.iloc[0].astype(int).astype(str); mal = meta.iloc[1].astype(int); nm = meta.iloc[2].astype(int)
    keep = (mal != 0).values
    nm_map = {0: "malignant_or_unresolved", 1: "T", 2: "B", 3: "Macrophage", 4: "Endothelial", 5: "CAF", 6: "NK"}
    ct = np.where(mal == 2, "Malignant", nm.map(nm_map).values)
    obs = pd.DataFrame({"patient": "Mel" + tumour.values, "sample": "Mel" + tumour.values,
                        "malignant": (mal == 2).values, "cell_type": ct, "cancer_type": "melanoma",
                        "source": "GEO:GSE72056"})[keep]
    X = _unlog_tpm(expr.loc[:, keep].T.values.astype(np.float32))
    prov = dict(accession="GSE72056", study="Tirosh et al. 2016 Science", assay="Smart-seq2",
                value_type="TPM back-transformed from log2(TPM/10+1)",
                label_source="author malignant row (2=yes, 1=no)", dropped_unresolved=int((~keep).sum()),
                note="Same lab as GSE115978; check for shared patients before treating as independent")
    return _finish(X, expr.columns[keep], expr.index, obs, prov)

# ---------------------------------------------------------------- GSE103322
def _gse103322_patients(series_matrix_path):
    """Map cell title -> patient using GEO series-matrix sample characteristics, if available."""
    titles, chars = None, []
    with gzip.open(series_matrix_path, "rt", errors="replace") as fh:
        for line in fh:
            if line.startswith("!Sample_title"):
                titles = [x.strip('"') for x in line.rstrip("\n").split("\t")[1:]]
            elif line.startswith("!Sample_characteristics_ch1"):
                chars.append([x.strip('"') for x in line.rstrip("\n").split("\t")[1:]])
            elif line.startswith("!series_matrix_table_begin"):
                break
    return titles, chars

def load_gse103322(raw_dir, series_matrix=None):
    """Puram 2017 HNSCC (Smart-seq2). Rows 3/4: classified as cancer / non-cancer cell; unlabelled dropped.
    Patient = numeric tumour ID parsed from cell name (HN28_..., HNSCC17_..., HNSCC_17 -> 'HN17' style), unless
    a series matrix with a patient characteristic is supplied."""
    D = pd.read_csv(f"{raw_dir}/GSE103322_HNSCC_all_data.txt.gz", sep="\t", index_col=0)
    meta, expr = D.iloc[:5], D.iloc[5:].astype(np.float32)
    cancer = pd.to_numeric(meta.iloc[2]).astype(int); noncancer = pd.to_numeric(meta.iloc[3]).astype(int)
    lymph = pd.to_numeric(meta.iloc[1]).astype(int); nct = meta.iloc[4].astype(str).str.strip().str.lstrip("-")
    keep = ((cancer + noncancer) == 1).values
    cells = pd.Index(D.columns)
    def pid(c):
        m = re.match(r"^(?:HNSCC|HN)_?(\d+)", c.strip('"'))
        return f"HN{m.group(1)}" if m else c.split("_")[0]
    patient = cells.map(pid)
    obs = pd.DataFrame({"patient": patient, "sample": [p + ("_LN" if l else "_primary") for p, l in zip(patient, lymph)],
                        "malignant": (cancer == 1).values,
                        "cell_type": np.where(cancer == 1, "Malignant", nct.values),
                        "cancer_type": "head_and_neck_SCC", "source": "GEO:GSE103322",
                        "lymph_node": lymph.values.astype(bool)})[keep]
    X = _unlog_tpm(expr.loc[:, keep].T.values)
    prov = dict(accession="GSE103322", study="Puram et al. 2017 Cell", assay="Smart-seq2",
                value_type="TPM back-transformed from log2(TPM/10+1)",
                label_source="author 'classified as cancer cell' row", dropped_unlabelled=int((~keep).sum()),
                patient_rule="regex ^(HNSCC|HN)_?(\\d+) -> HN<d>; verify against the paper's patient table")
    return _finish(X, cells[keep], expr.index, obs, prov)

# ---------------------------------------------------------------- GSE131907
def load_gse131907(raw_dir, min_malignant=50, min_normal=50, chunk_genes=2000):
    """Kim 2020 LUAD (10x, raw UMI). 'Malignant cells' subtype exists only in metastatic/advanced samples
    (mBrain, mLN, tL/B); primary tLung epithelium is labelled tS1-3 and is EXCLUDED (label ambiguous).
    Keeps samples with >= min_malignant malignant and >= min_normal non-epithelial cells; streams the
    dense 390 MB text matrix so peak memory stays a few GB."""
    ann = pd.read_csv(f"{raw_dir}/GSE131907_Lung_Cancer_cell_annotation.txt.gz", sep="\t")
    ann["is_mal"] = ann["Cell_subtype"] == "Malignant cells"
    ann["is_norm"] = (ann["Cell_type"] != "Epithelial cells") & (ann["Cell_type"] != "Undetermined")
    g = ann.groupby("Sample").agg(mal=("is_mal", "sum"), norm=("is_norm", "sum"))
    ok = g.index[(g.mal >= min_malignant) & (g.norm >= min_normal)]
    sel = ann[ann.Sample.isin(ok) & (ann.is_mal | ann.is_norm)].set_index("Index")
    path = f"{raw_dir}/GSE131907_Lung_Cancer_raw_UMI_matrix.txt.gz"
    with gzip.open(path, "rt") as fh:
        header = fh.readline().rstrip("\n").split("\t")
        cols = header[1:] if header[0] in ("Index", "", "gene", "Gene") or len(header) == len(ann) + 1 else header
        pos = pd.Index(cols)
        idx = pos.get_indexer(sel.index)
        assert (idx >= 0).all(), "annotation barcodes missing from matrix header"
        order = np.argsort(idx); idx_sorted = idx[order]
        genes, blocks = [], []
        buf = []
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            genes.append(parts[0])
            v = np.asarray(parts[1:], dtype=np.float32)[idx_sorted]
            buf.append(sp.csr_matrix(v))
            if len(buf) >= chunk_genes:
                blocks.append(sp.vstack(buf)); buf = []
        if buf: blocks.append(sp.vstack(buf))
    GxC = sp.vstack(blocks).tocsc()
    inv = np.empty_like(order); inv[order] = np.arange(len(order))
    X = GxC[:, inv].T.tocsr()
    s = sel
    obs = pd.DataFrame({"patient": s["Sample"].values, "sample": s["Sample"].values,
                        "sample_origin": s["Sample_Origin"].values, "malignant": s["is_mal"].values,
                        "cell_type": np.where(s["is_mal"], "Malignant", s["Cell_type"].values),
                        "cell_subtype": s["Cell_subtype"].values, "cancer_type": "lung_adenocarcinoma",
                        "source": "GEO:GSE131907"})
    prov = dict(accession="GSE131907", study="Kim et al. 2020 Nat Commun", assay="10x 3'",
                value_type="raw UMI counts", label_source="author Cell_subtype == 'Malignant cells'",
                samples_kept=list(map(str, ok)),
                note="Each sample is one patient-site; check the paper for patients contributing >1 sample")
    return _finish(X, s.index.values, genes, obs, prov)

# ---------------------------------------------------------------- CELLxGENE
CXG_API = "https://api.cellxgene.cziscience.com/curation/v1/datasets"

def find_cellxgene_tumour_datasets(min_donors=4, max_mb=1000, label=("malignant cell", "neoplastic cell")):
    """Query CELLxGENE Discover for human, non-spatial datasets that contain a malignant/neoplastic label."""
    import requests
    ds = requests.get(CXG_API, timeout=120).json()
    rows = []
    for d in ds:
        if d.get("spatial") or "Homo sapiens" not in [o["label"] for o in d.get("organism", [])]:
            continue
        cts = [c["label"] for c in d.get("cell_type", [])]
        if not any(c in label for c in cts):
            continue
        h = [x for x in d.get("assets", []) if x["filetype"] == "H5AD"]
        if not h:
            continue
        rows.append(dict(dataset_id=d["dataset_id"], title=d["title"], collection_doi=d.get("collection_doi"),
                         donors=len(d.get("donor_id", [])), cells=d["cell_count"],
                         n_cell_types=len(cts), has_normal=len(cts) > 1,
                         disease="; ".join(x["label"] for x in d.get("disease", [])),
                         assay="; ".join(x["label"] for x in d.get("assay", [])),
                         h5ad_url=h[0]["url"], h5ad_mb=round(h[0]["filesize"] / 1e6)))
    t = pd.DataFrame(rows)
    return t[(t.donors >= min_donors) & (t.h5ad_mb <= max_mb) & t.has_normal].sort_values("h5ad_mb")

def load_cellxgene(h5ad_path, cancer_type, dataset_id=None, collection_doi=None,
                   malignant_labels=("malignant cell", "neoplastic cell"), drop_labels=("unknown",)):
    """Standardise a CELLxGENE h5ad: raw counts from .raw.X when present; Ensembl IDs kept in var['ensembl_id']."""
    a = ad.read_h5ad(h5ad_path)
    if a.raw is not None:
        X = a.raw.X; var = a.raw.var.copy()
        if "feature_name" not in var and "feature_name" in a.var:
            var["feature_name"] = a.var.reindex(var.index)["feature_name"]
    else:
        X = a.X; var = a.var.copy()
    X = sp.csr_matrix(X, dtype=np.float32)
    is_int = np.allclose(X.data[: min(100000, X.nnz)] % 1, 0)
    ct = a.obs["cell_type"].astype(str)
    keep = ~ct.isin(drop_labels).values
    obs = pd.DataFrame({"patient": a.obs["donor_id"].astype(str).values,
                        "sample": a.obs.get("sample_id", a.obs["donor_id"]).astype(str).values,
                        "malignant": ct.isin(malignant_labels).values, "cell_type": ct.values,
                        "cancer_type": cancer_type, "source": f"CELLxGENE:{dataset_id}",
                        "assay": a.obs["assay"].astype(str).values,
                        "tissue": a.obs["tissue"].astype(str).values,
                        "disease": a.obs["disease"].astype(str).values}, index=a.obs_names)[keep]
    out = ad.AnnData(X[keep], obs=obs, var=pd.DataFrame(index=var.index))
    out.var["ensembl_id"] = var.index.values
    out.var["feature_name"] = var["feature_name"].astype(str).values if "feature_name" in var else var.index.values
    # symbol index for consistency with GEO loaders
    out.var_names = pd.Index(out.var["feature_name"].values); out.var_names_make_unique()
    out.layers["counts"] = out.X.copy()
    for c in ["patient", "cell_type", "cancer_type", "sample", "source"]:
        out.obs[c] = out.obs[c].astype(str).astype("category")
    out.uns["provenance"] = dict(source="CELLxGENE Discover", dataset_id=dataset_id, collection_doi=collection_doi,
                                 value_type="raw counts (.raw.X)" if a.raw is not None else ("integer X" if is_int else "X (not integer: check)"),
                                 label_source="CELLxGENE cell_type in " + str(list(malignant_labels)),
                                 dropped_labels=str(list(drop_labels)), dropped_cells=int((~keep).sum()),
                                 possibly_in_FM_pretraining=True)
    return out

def symbol_to_ensembl_map(cellxgene_adatas):
    """Build a symbol->Ensembl map from CELLxGENE var tables (for Geneformer on GEO data)."""
    m = {}
    for a in cellxgene_adatas:
        for s, e in zip(a.var["feature_name"], a.var["ensembl_id"]):
            m.setdefault(str(s), str(e))
    return m

def add_ensembl(adata, mapping):
    adata.var["ensembl_id"] = adata.var["feature_name"].map(mapping)
    adata.uns.setdefault("provenance", {})["ensembl_mapped_frac"] = float(adata.var["ensembl_id"].notna().mean())
    return adata
