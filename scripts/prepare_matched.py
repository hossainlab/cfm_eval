"""Write HARD-task (lineage-matched) subsets: python scripts/prepare_matched.py --std data/std --out data/ready_matched"""
import argparse
from pathlib import Path
import pandas as pd
from cfm_eval.data import read_std, write_std, make_ready_matched, MATCHED_NORMAL
ap = argparse.ArgumentParser(); ap.add_argument("--std", default="data/std"); ap.add_argument("--out", default="data/ready_matched")
a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True); rows = []
for name, labels in MATCHED_NORMAL.items():
    p = Path(a.std) / f"{name}.h5ad"
    if not p.exists(): continue
    r = make_ready_matched(read_std(p), labels)
    if r.obs.patient.nunique() < 2: print("skip", name, "(<2 eligible patients)"); continue
    write_std(r, out / f"{name}.h5ad")
    rows.append(dict(dataset=name, normal_labels="; ".join(labels), cells=r.n_obs, patients=r.obs.patient.nunique(),
                     malignant=int(r.obs.malignant.sum()), matched_normal=int((~r.obs.malignant).sum())))
pd.DataFrame(rows).to_csv(out / "MANIFEST.csv", index=False); print(pd.DataFrame(rows).to_string())
