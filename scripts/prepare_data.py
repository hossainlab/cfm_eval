"""Download (optional) and standardise all P3 datasets, then write benchmark-ready subsets.

  python scripts/prepare_data.py --raw data/raw --std data/std --ready data/ready [--download]

GEO files go in data/raw/, CELLxGENE h5ad files in data/raw/cellxgene/.
GSE72056 is loaded but EXCLUDED from the ready set: its cells are re-released inside GSE115978 ('Tirosh' cohort).
"""
import argparse, json, subprocess
from pathlib import Path
import pandas as pd
from cfm_eval import loaders as L
from cfm_eval.data import write_std, read_std, make_ready, eligible_patients

CXG = {  # name: (cancer_type, dataset_id, collection DOI)
    "cxg_breast_ctm2": ("breast_cancer", "6ff7acae-ac81-45f5-9348-94e85a25fe7c", "10.1002/ctm2.1356"),
    "cxg_crlm_bdrhapsody": ("colorectal_liver_metastasis", "a3bcb68b-a36d-460f-bdde-da4fc0771c8c", "10.1186/s12943-025-02430-7"),
    "cxg_gbm_darmanis": ("glioblastoma", "2e357b12-781b-4c90-9321-30206529cc4a", "10.1016/j.celrep.2017.10.030"),
    "cxg_follicular_lymphoma": ("follicular_lymphoma", "61c4a85a-9897-42c4-9dc4-c3506639854d", "10.1158/2643-3230.BCD-21-0075"),
}
GEO = {
    "geo_GSE115978_melanoma": ("GSE115978", L.load_gse115978, True),
    "geo_GSE103322_hnscc": ("GSE103322", L.load_gse103322, True),
    "geo_GSE131907_luad": ("GSE131907", L.load_gse131907, True),
    "geo_GSE72056_melanoma": ("GSE72056", L.load_gse72056, False),  # redundant with GSE115978
}

def download(raw):
    (raw / "cellxgene").mkdir(parents=True, exist_ok=True)
    for acc, files in L.FILES.items():
        for f in files:
            if not (raw / f).exists():
                subprocess.run(["curl", "-sSL", "-o", str(raw / f), L.geo_url(acc, f)], check=True)
    for name, (_, did, _) in CXG.items():
        p = raw / "cellxgene" / f"{name}.h5ad"
        if not p.exists():
            subprocess.run(["curl", "-sSL", "-o", str(p), f"https://datasets.cellxgene.cziscience.com/{did}.h5ad"], check=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw"); ap.add_argument("--std", default="data/std")
    ap.add_argument("--ready", default="data/ready"); ap.add_argument("--download", action="store_true")
    ap.add_argument("--only", nargs="*"); ap.add_argument("--max-cells-per-patient", type=int, default=2000)
    a = ap.parse_args(); raw, std, ready = map(Path, (a.raw, a.std, a.ready))
    std.mkdir(parents=True, exist_ok=True); ready.mkdir(parents=True, exist_ok=True)
    if a.download: download(raw)
    cxg_objs, manifest = [], []
    for name, (ct, did, doi) in CXG.items():
        if a.only and name not in a.only: continue
        x = L.load_cellxgene(raw / "cellxgene" / f"{name}.h5ad", ct, did, doi); write_std(x, std / f"{name}.h5ad")
        cxg_objs.append(x)
    ens = L.symbol_to_ensembl_map(cxg_objs) if cxg_objs else {}
    for name, (acc, fn, use) in GEO.items():
        if a.only and name not in a.only: continue
        x = fn(str(raw))
        if ens: L.add_ensembl(x, ens)
        x.uns["provenance"]["possibly_in_FM_pretraining"] = "unknown (GEO-only; check CELLxGENE)"
        write_std(x, std / f"{name}.h5ad")
        if not use: continue
    for p in sorted(std.glob("*.h5ad")):
        x = read_std(p); name = p.stem
        use = GEO.get(name, (None, None, True))[2]
        row = dict(dataset=name, cells=x.n_obs, genes=x.n_vars, patients=x.obs.patient.nunique(),
                   malignant=int(x.obs.malignant.sum()), eligible_patients=len(eligible_patients(x)),
                   cancer_type=str(x.obs.cancer_type.iloc[0]), in_ready_set=use,
                   **{k: str(v) for k, v in x.uns.get("provenance", {}).items() if k in
                      ("accession", "dataset_id", "collection_doi", "assay", "value_type", "label_source")})
        if use:
            r = make_ready(x, a.max_cells_per_patient); write_std(r, ready / f"{name}.h5ad")
            row.update(ready_cells=r.n_obs, ready_patients=r.obs.patient.nunique(), ready_malignant=int(r.obs.malignant.sum()))
        manifest.append(row)
    pd.DataFrame(manifest).to_csv(ready / "MANIFEST.csv", index=False)
    print(pd.DataFrame(manifest)[["dataset", "cells", "patients", "eligible_patients", "in_ready_set"]].to_string())

if __name__ == "__main__":
    main()
