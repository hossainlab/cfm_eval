"""Publication figures from the benchmark outputs.
   python scripts/make_figures.py --results results --out results/figures
Reads results/{ready,matched,t3}/{summary,per_patient,compare_vs_*}.csv and writes, as PDF (vector) + 300-dpi PNG:
  fig1_main         a/b few-shot-10 LOPO AUROC per held-out patient (ready, matched); c label noise;
                    d cross-cancer transfer (T3); e patient structure (T4)
  fig2_paired       paired AUROC differences (patient-bootstrap 95% CI, Holm) vs HVG+PCA and vs random projection
  figS1_datasets    per-dataset few-shot-10 AUROC and unsupervised Leiden ARI
Embeddings without results are dropped, so the script also runs on partial results.
Colour encodes model family: grey = label-free baselines and Harmony, blue/orange/aqua = foundation models;
the random-weight scGPT control is open blue. Every series also has its own marker, so nothing relies on colour alone.
"""
import argparse
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

EMB = {  # key: (label, colour, marker, filled)
    "X_hvg_pca":      ("HVG + PCA",           "#3d3c39", "o", True),
    "X_harmony":      ("Harmony*",            "#9c9b95", "D", True),
    "X_randproj":     ("Random projection",   "#6e6d68", "s", True),
    "X_scgpt_random": ("scGPT, random wts",   "#2a78d6", "o", False),
    "X_scgpt":        ("scGPT",               "#2a78d6", "o", True),
    "X_geneformer":   ("Geneformer",          "#eb6834", "^", True),
    "X_uce":          ("UCE",                 "#1baf7a", "v", True),
}
FMS = ["X_scgpt", "X_geneformer", "X_uce"]
DATASET = {"cxg_breast_ctm2": "Breast", "cxg_crlm_bdrhapsody": "CRLM", "cxg_follicular_lymphoma": "Foll. lymphoma",
           "cxg_gbm_darmanis": "Glioblastoma", "geo_GSE103322_hnscc": "HNSCC", "geo_GSE115978_melanoma": "Melanoma",
           "geo_GSE131907_luad": "LUAD"}
CANCER = {"breast_cancer": "Breast", "colorectal_liver_metastasis": "CRLM", "follicular_lymphoma": "Foll. lymphoma",
          "glioblastoma": "Glioblastoma", "head_and_neck_SCC": "HNSCC", "lung_adenocarcinoma": "LUAD",
          "melanoma": "Melanoma"}
INK, INK2, RULE = "#0b0b0b", "#52514e", "#d9d8d3"
SEQ = LinearSegmentedColormap.from_list("blue", ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
MM = 1 / 25.4

def style():
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": 7,
        "axes.titlesize": 7, "axes.labelsize": 7, "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5,
        "ytick.major.size": 2.5, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": INK2,
        "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK, "text.color": INK,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300, "figure.dpi": 150, "legend.frameon": False})

def mk(e, s=14, ax=None, **kw):
    lab, c, m, filled = EMB[e]
    return dict(marker=m, color=c, markerfacecolor=c if filled else "white", markeredgecolor=c, **kw)

def panel(ax, letter, dx=0.055):
    """Panel letter at the figure-coordinate left of the axes, level with the title."""
    b = ax.get_position()
    ax.figure.text(b.x0 - dx, b.y1 + 0.012, letter, fontsize=9, fontweight="bold", va="bottom")

def present(df):
    return [e for e in EMB if e in set(df.embedding)]

def load(d):
    d = Path(d)
    return {k: pd.read_csv(d / f"{k}.csv") for k in ["summary", "per_patient"] if (d / f"{k}.csv").exists()}

def strip(ax, pp, probe, title):
    """One point per held-out patient (mean over seeds), no label noise; bar = median."""
    d = pp[(pp.probe == probe) & (pp.label_noise == 0)]
    d = d.groupby(["dataset", "patient", "embedding"], as_index=False).auroc.mean()
    embs = present(d); rng = np.random.default_rng(0)
    for i, e in enumerate(embs):
        y = d[d.embedding == e].auroc.values
        x = i + rng.uniform(-0.28, 0.28, len(y))
        lab, c, m, filled = EMB[e]
        ax.scatter(x, y, s=7, marker=m, facecolor=c if filled else "white", edgecolor=c, linewidth=0.5,
                   alpha=0.75, zorder=2)
        ax.plot([i - 0.36, i + 0.36], [np.median(y)] * 2, color=INK, lw=1.4, solid_capstyle="round", zorder=3)
        ax.text(i, 1.012, f"{np.median(y):.3f}", ha="center", va="bottom", fontsize=5.5, color=INK2)
    ax.set_xticks(range(len(embs)), [EMB[e][0] for e in embs], rotation=40, ha="right")
    ax.set_xlim(-0.6, len(embs) - 0.4); ax.set_ylim(min(0.5, d.auroc.min() - 0.02), 1.04)
    ax.set_ylabel("Held-out-patient AUROC"); ax.axhline(0.5, color=RULE, lw=0.6, zorder=0)
    ax.set_title(f"{title}\n{d.patient.nunique()} patients, {d.dataset.nunique()} datasets", loc="left", pad=10,
                 linespacing=1.4)

def heat(ax, M, fmt="{:.2f}", vmin=0.5, vmax=1.0, cbar_label="AUROC"):
    im = ax.imshow(M.values, cmap=SEQ, vmin=vmin, vmax=vmax, aspect="auto")
    for (r, c), v in np.ndenumerate(M.values):
        if np.isfinite(v):
            ax.text(c, r, fmt.format(v), ha="center", va="center", fontsize=5.5,
                    color="white" if (v - vmin) / (vmax - vmin) > 0.6 else INK)
    ax.set_xticks(range(M.shape[1]), [EMB[e][0] for e in M.columns], rotation=40, ha="right")
    ax.set_yticks(range(M.shape[0]), M.index)
    ax.tick_params(length=0); [s.set_visible(False) for s in ax.spines.values()]
    cb = plt.colorbar(im, ax=ax, fraction=0.05, pad=0.03); cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=5.5, length=2); cb.ax.set_title(cbar_label, fontsize=5.5, pad=3, loc="left")

def legend(fig, embs, y=1.0):
    h = [Line2D([], [], lw=0, markersize=5, markeredgewidth=0.8, **mk(e)) for e in embs]
    fig.legend(h, [EMB[e][0] for e in embs], loc="upper center", ncol=len(embs), bbox_to_anchor=(0.5, y),
               handletextpad=0.3, columnspacing=1.2)

def fig1(R, out):
    ready, matched, t3 = R["ready"], R["matched"], R["t3"]
    fig = plt.figure(figsize=(180 * MM, 150 * MM))
    gs = fig.add_gridspec(2, 6, height_ratios=[1, 1], hspace=0.85, wspace=1.6, left=0.07, right=0.97, top=0.87,
                          bottom=0.1)
    a = fig.add_subplot(gs[0, :3]); b = fig.add_subplot(gs[0, 3:])
    c = fig.add_subplot(gs[1, :2]); d = fig.add_subplot(gs[1, 2:4]); e = fig.add_subplot(gs[1, 4:])
    strip(a, ready["per_patient"], "fewshot10", "Malignant vs all non-malignant, 10-shot probe")
    strip(b, matched["per_patient"], "fewshot10", "Malignant vs lineage-matched normal, 10-shot probe")
    b.set_ylabel("")
    # c: label noise, mean over datasets of per-dataset means (seeds averaged first)
    s = ready["summary"].groupby(["embedding", "label_noise", "dataset"]).lopo_auroc_fs10.mean() \
        .groupby(["embedding", "label_noise"]).mean().reset_index()
    for em in present(s):
        q = s[s.embedding == em].sort_values("label_noise")
        c.plot(100 * q.label_noise, q.lopo_auroc_fs10, lw=1.2, markersize=3.5, markeredgewidth=0.7, **mk(em))
    c.set_xlabel("Training labels flipped (%)"); c.set_ylabel("Mean 10-shot AUROC")
    c.set_xticks(sorted(100 * s.label_noise.unique())); c.set_title("Label noise, 10-shot (ready set)", loc="left")
    # d: T3
    t = t3["summary"].groupby(["held_out_cancer", "embedding"]).auroc_pooled.mean().unstack()
    t = t[[x for x in EMB if x in t.columns]].rename(index=CANCER)
    heat(d, t); d.set_title("Cross-cancer transfer (T3)", loc="left")
    # e: T4
    s4 = ready["summary"][ready["summary"].label_noise == 0].groupby(["embedding", "dataset"])[
        ["normal_same_patient_excess", "malignant_same_patient_excess"]].mean().groupby("embedding").mean()
    for em in present(s4.reset_index()):  # 7 points: identity from the shared legend, not direct labels
        x, y = s4.loc[em]
        e.plot(x, y, lw=0, markersize=5.5, markeredgewidth=0.8, **mk(em))
    e.set_xlabel("Normal cells: same-patient kNN excess\n(lower = less patient batch effect)")
    e.set_ylabel("Malignant cells: same-patient excess\n(higher = tumour individuality kept)")
    e.set_title("Patient structure (ready set)", loc="left")
    e.set_xlim(left=min(-0.02, e.get_xlim()[0])); e.set_ylim(bottom=min(-0.02, e.get_ylim()[0]))
    for ax, l in zip([a, b, c, d, e], "abcde"):
        panel(ax, l, dx=0.06 if l in "ace" else 0.11 if l == "d" else 0.045)
    legend(fig, present(ready["per_patient"]), y=0.995)
    save(fig, out / "fig1_main")

def fig2(R, out):
    tasks = [("ready", "fewshot10", "Ready, 10-shot"), ("ready", "full", "Ready, full probe"),
             ("matched", "fewshot10", "Matched, 10-shot"), ("matched", "full", "Matched, full probe"),
             ("t3", None, "Cross-cancer (T3)")]
    refs = [("hvg_pca", "vs HVG + PCA"), ("randproj", "vs random projection")]
    rows = []
    for ref, _ in refs:
        for st, probe, tl in tasks:
            p = R[st]["dir"] / f"compare_vs_{ref}.csv"
            if not p.exists():
                continue
            c = pd.read_csv(p)
            c = c[c.iloc[:, c.columns.get_loc("embedding") - 1] == "ALL"]  # pooled row (dataset / held_out_cancer)
            if probe is not None:
                c = c[(c.probe == probe) & (c.label_noise == 0)]
            rows.append(c.assign(task=tl, refl=ref))
    C = pd.concat(rows)
    embs = [e for e in ["X_uce", "X_geneformer", "X_scgpt", "X_scgpt_random", "X_harmony", "X_randproj"]
            if e in set(C.embedding)]
    fig, axes = plt.subplots(len(refs), len(tasks), figsize=(180 * MM, 95 * MM), sharey="row")
    for i, (ref, rl) in enumerate(refs):
        for j, (_, _, tl) in enumerate(tasks):
            ax = axes[i, j]; d = C[(C.refl == ref) & (C.task == tl)].set_index("embedding")
            ax.axvline(0, color=INK2, lw=0.6, zorder=0)
            rows_ = [e for e in embs if e != f"X_{ref}"]
            for k, em in enumerate(rows_):
                if em not in d.index:
                    continue
                r = d.loc[em]
                ax.plot([r.mean_diff_lo, r.mean_diff_hi], [k, k], color=EMB[em][1], lw=1.0, solid_capstyle="round")
                ax.plot(r.mean_diff, k, lw=0, markersize=4.5, markeredgewidth=0.8, **mk(em))
                if r.p_holm < 0.05:
                    ax.text(r.mean_diff_hi, k + 0.08, " *", va="center", ha="left", fontsize=7, color=INK)
            ax.set_yticks(range(len(rows_)), [EMB[e][0] for e in rows_]); ax.set_ylim(-0.6, len(rows_) - 0.4)
            ax.tick_params(axis="y", length=0)
            ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(3, symmetric=True))
            lim = max(abs(np.array(ax.get_xlim())).max(), 0.02) * 1.12; ax.set_xlim(-lim, lim)
            if i == 0: ax.set_title(tl, fontsize=6.5)
            if j == 0: ax.set_ylabel(rl, fontsize=7, fontweight="bold")
            if i == len(refs) - 1: ax.set_xlabel("ΔAUROC", fontsize=6)
    fig.text(0.5, 0.005, "Mean paired difference across held-out patients, patient-bootstrap 95% CI. "
             "* Holm-adjusted paired Wilcoxon p < 0.05. Positive = row embedding better.",
             ha="center", fontsize=5.5, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 1), w_pad=0.6, h_pad=1.2)
    save(fig, out / "fig2_paired")

def figS1(R, out):
    ready = R["ready"]["summary"]; ready = ready[ready.label_noise == 0]
    f = ready.groupby(["dataset", "embedding"]).lopo_auroc_fs10.mean().unstack()
    u = ready.groupby(["dataset", "embedding"]).leiden_ari.mean().unstack()
    cols = [x for x in EMB if x in f.columns]
    fig, (a, b) = plt.subplots(1, 2, figsize=(180 * MM, 70 * MM))
    heat(a, f[cols].rename(index=DATASET)); a.set_title("10-shot LOPO AUROC (ready set)", loc="left")
    heat(b, u[cols].rename(index=DATASET), vmin=0, vmax=1, cbar_label="ARI")
    b.set_title("Unsupervised: Leiden ARI vs malignant label", loc="left")
    fig.tight_layout(); panel(a, "a", 0.1); panel(b, "b", 0.1)
    save(fig, out / "figS1_datasets")

def save(fig, p):
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{p}.pdf"); fig.savefig(f"{p}.png"); plt.close(fig); print("wrote", p)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="results/figures")
    args = ap.parse_args()
    style()
    R = {k: {**load(Path(args.results) / k), "dir": Path(args.results) / k} for k in ["ready", "matched", "t3"]}
    out = Path(args.out)
    fig1(R, out); fig2(R, out); figS1(R, out)

if __name__ == "__main__":
    main()
