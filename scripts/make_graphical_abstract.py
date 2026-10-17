"""Graphical abstract (flat, BioRender-like) from the benchmark outputs.
   python scripts/make_graphical_abstract.py --results results --out results/figures
Writes graphical_abstract.svg, and .png (2x) + .pdf when Google Chrome is available for headless rendering.
All numbers shown are read from results/{ready,t3}/summary.csv.
"""
import argparse, math, shutil, subprocess, tempfile
from pathlib import Path
import numpy as np, pandas as pd

W, H = 1600, 790
FONT = "'Helvetica Neue', Helvetica, Arial, sans-serif"
INK, MUTED = "#1f2937", "#4b5563"
MAL, MAL_DARK, NORM, NORM_DARK = "#a78bfa", "#6d28d9", "#7dd3fc", "#0369a1"
MODEL = {  # key: (label, colour)
    "X_uce": ("UCE", "#1baf7a"), "X_geneformer": ("Geneformer", "#eb6834"), "X_scgpt": ("scGPT", "#2a78d6"),
    "X_randproj": ("Random projection", "#6e6d68"), "X_harmony": ("Harmony*", "#9c9b95"),
    "X_hvg_pca": ("HVG + PCA", "#3d3c39"), "X_scgpt_random": ("scGPT, random wts", "#93c5fd"),
}
FMS = ["X_uce", "X_geneformer", "X_scgpt"]


def text(x, y, s, size=14, weight="normal", fill=INK, anchor="start", extra=""):
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" font-weight="{weight}" fill="{fill}" '
            f'text-anchor="{anchor}" {extra}>{s}</text>')


def lines(x, y, rows, size=13, lh=17, **kw):
    return "".join(text(x, y + i * lh, r, size, **kw) for i, r in enumerate(rows))


def blob(cx, cy, r, wobble, phase, n=14):
    """Smooth irregular closed path (Catmull-Rom through polar points)."""
    pts = []
    for i in range(n):
        t = 2 * math.pi * i / n
        rr = r * (1 + wobble * math.sin(3 * t + phase) + 0.5 * wobble * math.sin(5 * t + 2 * phase))
        pts.append((cx + rr * math.cos(t), cy + rr * math.sin(t)))
    d = f"M{pts[0][0]:.1f},{pts[0][1]:.1f}"
    for i in range(n):
        p0, p1, p2, p3 = pts[i - 1], pts[i], pts[(i + 1) % n], pts[(i + 2) % n]
        c1 = (p1[0] + (p2[0] - p0[0]) / 6, p1[1] + (p2[1] - p0[1]) / 6)
        c2 = (p2[0] - (p3[0] - p1[0]) / 6, p2[1] - (p3[1] - p1[1]) / 6)
        d += f" C{c1[0]:.1f},{c1[1]:.1f} {c2[0]:.1f},{c2[1]:.1f} {p2[0]:.1f},{p2[1]:.1f}"
    return d + "Z"


def malignant(cx, cy, r, phase=0.0):
    return (f'<path d="{blob(cx, cy, r, 0.13, phase)}" fill="{MAL}" stroke="{MAL_DARK}" stroke-width="1.5"/>'
            f'<path d="{blob(cx + r * .08, cy - r * .05, r * .5, 0.18, phase + 1)}" fill="{MAL_DARK}" opacity=".85"/>'
            f'<circle cx="{cx + r * .22:.1f}" cy="{cy - r * .12:.1f}" r="{r * .1:.1f}" fill="#ede9fe"/>')


def normal(cx, cy, r, fill=NORM, edge=NORM_DARK):
    return (f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{fill}" stroke="{edge}" stroke-width="1.5"/>'
            f'<circle cx="{cx - r * .1:.1f}" cy="{cy + r * .05:.1f}" r="{r * .38:.1f}" fill="{edge}" opacity=".8"/>')


def fibroblast(cx, cy, r, rot):
    return (f'<g transform="rotate({rot} {cx} {cy})"><ellipse cx="{cx}" cy="{cy}" rx="{r * 1.9}" ry="{r * .6}" '
            f'fill="#bbf7d0" stroke="#15803d" stroke-width="1.5"/><ellipse cx="{cx}" cy="{cy}" rx="{r * .55}" '
            f'ry="{r * .32}" fill="#15803d" opacity=".8"/></g>')


def person(cx, cy, s, fill):
    return (f'<circle cx="{cx}" cy="{cy - s * .55:.1f}" r="{s * .27:.1f}" fill="{fill}"/>'
            f'<path d="M{cx - s * .42:.1f},{cy + s * .45:.1f} Q{cx - s * .42:.1f},{cy - s * .2:.1f} {cx},{cy - s * .2:.1f} '
            f'Q{cx + s * .42:.1f},{cy - s * .2:.1f} {cx + s * .42:.1f},{cy + s * .45:.1f} Z" fill="{fill}"/>')


def network(x, y, col):
    """Tiny transformer/network glyph, 26 x 22."""
    nodes = [(x + dx, y + dy) for dx in (0, 13, 26) for dy in (0, 11, 22)]
    e = "".join(f'<line x1="{a[0]}" y1="{a[1]}" x2="{b[0]}" y2="{b[1]}" stroke="{col}" stroke-width=".9" opacity=".6"/>'
                for a in nodes for b in nodes if b[0] - a[0] == 13)
    return e + "".join(f'<circle cx="{a}" cy="{b}" r="3.2" fill="{col}"/>' for a, b in nodes)


def dice(x, y, col):
    dots = [(6, 6), (18, 6), (12, 12), (6, 18), (18, 18)]
    return (f'<rect x="{x}" y="{y}" width="24" height="24" rx="5" fill="white" stroke="{col}" stroke-width="2"/>'
            + "".join(f'<circle cx="{x + a}" cy="{y + b}" r="2.2" fill="{col}"/>' for a, b in dots))


def chip(x, y, w, label, fill, fg="white", h=34, icon=""):
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{h / 2}" fill="{fill}"/>{icon}'
            + text(x + w / 2 + (10 if icon else 0), y + h / 2 + 5, label, 14, "600", fg, "middle"))


def panel(x, y, w, h, n, title, fill, accent):
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="18" fill="{fill}" filter="url(#sh)"/>'
            f'<circle cx="{x + 30}" cy="{y + 34}" r="16" fill="{accent}"/>'
            + text(x + 30, y + 40, n, 17, "bold", "white", "middle") + text(x + 56, y + 41, title, 20, "bold"))


def arrow(x, y):
    return f'<path d="M{x},{y - 14} L{x + 14},{y - 14} L{x + 26},{y} L{x + 14},{y + 14} L{x},{y + 14} L{x + 10},{y} Z" fill="#94a3b8"/>'


def build(res):
    s = pd.read_csv(res / "ready" / "summary.csv")
    s = s[s.label_noise == 0]
    # the patient is the unit: average seeds within each held-out patient, then average over patients
    pp = pd.read_csv(res / "ready" / "per_patient.csv")
    per_pat = pp.groupby(["probe", "label_noise", "embedding", "dataset", "patient"]).auroc.mean()
    pat_mean = per_pat.groupby(["probe", "label_noise", "embedding"]).mean()
    fs10, full, n10 = pat_mean["fewshot10", 0.0], pat_mean["full", 0.0], pat_mean["fewshot10", 0.1]
    n_pat = per_pat["fewshot10", 0.0, "X_hvg_pca"].size
    normal_excess = s.groupby("embedding").normal_same_patient_excess.mean()
    t3 = pd.read_csv(res / "t3" / "summary.csv")
    t3m = t3.groupby(["held_out_cancer", "embedding"]).auroc_pooled.mean()
    t3_mean = t3m.groupby("embedding").mean()
    n_cancer = t3.held_out_cancer.nunique()
    edge = [fs10[m] - fs10["X_randproj"] for m in FMS]
    share = [(fs10["X_randproj"] - fs10["X_hvg_pca"]) / (fs10[m] - fs10["X_hvg_pca"]) for m in FMS]
    cmp = pd.read_csv(res / "ready" / "compare_vs_randproj.csv")
    p_max = cmp[(cmp.probe == "fewshot10") & (cmp.label_noise == 0) & (cmp.dataset == "ALL")
                & cmp.embedding.isin(FMS)].p_holm.max()
    rng = np.random.default_rng(3)
    o = []

    # title
    o.append(text(W / 2, 50, "Do single-cell foundation models help identify malignant cells?", 30, "bold", INK, "middle"))
    o.append(text(W / 2, 80, "Zero-shot benchmark of scGPT, Geneformer and UCE against simple baselines and random "
                  "controls on tumour scRNA-seq", 17, "normal", MUTED, "middle"))

    # 1 data
    x0 = 30
    o.append(panel(x0, 110, 340, 650, "1", "Tumour scRNA-seq", "#faf5ff", MAL_DARK))
    o.append(f'<ellipse cx="200" cy="262" rx="140" ry="88" fill="#fce7f3" stroke="#f9a8d4" stroke-width="1.5"/>')
    for (cx, cy, r, ph) in [(150, 230, 22, 0), (196, 254, 26, 1.2), (246, 226, 21, 2.1), (240, 288, 20, 3.0),
                            (160, 290, 19, 4.2), (288, 262, 17, 5.1)]:
        o.append(malignant(cx, cy, r, ph))
    for (cx, cy, r) in [(105, 262, 12), (120, 225, 10), (200, 205, 11), (300, 222, 10), (205, 318, 11), (118, 300, 9)]:
        o.append(normal(cx, cy, r))
    o.append(fibroblast(270, 324, 10, -15) + fibroblast(310, 296, 9, 60) + normal(98, 230, 9, "#fde68a", "#b45309"))
    o.append(text(200, 380, "Tumour cells + tumour microenvironment", 14, "600", MUTED, "middle"))
    pal = ["#2a78d6", "#eb6834", "#1baf7a", "#d946ef", "#f59e0b", "#64748b"]
    for i, c in enumerate(pal):
        o.append(person(83 + i * 47, 432, 34, c))
    o.append(text(200, 482, f"{n_pat} patients · {s.dataset.nunique()} datasets · {n_cancer} cancer types",
                  15, "bold", INK, "middle"))
    cx, cy = 48, 498
    for lab in ["Melanoma", "HNSCC", "LUAD", "Breast", "CRLM", "Glioblastoma", "Foll. lymphoma"]:
        w = len(lab) * 7.4 + 22
        if cx + w > 356:
            cx, cy = 48, cy + 34
        o.append(chip(cx, cy, w, lab, "white", MAL_DARK, 26).replace('font-size="14"', 'font-size="12.5"')
                 .replace('<rect', '<rect stroke="#c4b5fd" stroke-width="1.2"', 1))
        cx += w + 7
    o.append(malignant(66, 612, 15, 1.0) + text(92, 617, "Malignant cell", 14, "600"))
    o.append(normal(66, 652, 12) + text(92, 657, "Non-malignant cell", 14, "600"))
    o.append(lines(48, 698, ["Malignant vs all normal cells, and vs", "lineage-matched normal cells"],
                   13, 17, fill=MUTED))

    # 2 embed
    x0 = 400
    o.append(arrow(373, 435))
    o.append(panel(x0, 110, 360, 650, "2", "Embed (zero-shot)", "#eff6ff", "#2a78d6"))
    o.append(text(x0 + 24, 186, "Foundation models", 14, "bold", MUTED))
    for i, (m, dim) in enumerate(zip(["X_scgpt", "X_geneformer", "X_uce"], ["512-d", "768-d", "1280-d"])):
        lab, col = MODEL[m]
        o.append(chip(x0 + 20 + i * 110, 196, 104, lab, col))
        o.append(text(x0 + 72 + i * 110, 248, dim, 12, "normal", MUTED, "middle"))
    o.append(network(x0 + 300, 165, "#2a78d6"))
    o.append(text(x0 + 24, 284, "Baselines", 14, "bold", MUTED)
             + text(x0 + 100, 284, "(*Harmony uses patient labels)", 12, "normal", MUTED))
    o.append(chip(x0 + 20, 294, 150, "HVG + PCA", MODEL["X_hvg_pca"][1]))
    o.append(chip(x0 + 178, 294, 150, "Harmony*", MODEL["X_harmony"][1]))
    o.append(text(x0 + 95, 346, "50-d", 12, "normal", MUTED, "middle") + text(x0 + 253, 346, "50-d", 12, "normal", MUTED, "middle"))
    o.append(text(x0 + 24, 376, "Negative controls", 14, "bold", MUTED))
    o.append(chip(x0 + 20, 386, 160, "Random proj.", "white", INK, 34, dice(x0 + 30, 391, "#6e6d68"))
             .replace('<rect', '<rect stroke="#6e6d68" stroke-width="1.5"', 1))
    o.append(chip(x0 + 188, 386, 152, "scGPT, random", "white", "#2a78d6", 34)
             .replace('<rect', '<rect stroke="#2a78d6" stroke-width="1.5" stroke-dasharray="4 3"', 1))
    o.append(text(x0 + 100, 438, "512-d", 12, "normal", MUTED, "middle") + text(x0 + 264, 438, "512-d, untrained", 12, "normal", MUTED, "middle"))
    o.append(text(x0 + 180, 460, "no fine-tuning · identical cells for every model", 12, "normal", MUTED, "middle"))
    o.append(f'<path d="M{x0 + 170},{468} L{x0 + 190},{468} L{x0 + 190},{478} L{x0 + 200},{478} L{x0 + 180},{494} '
             f'L{x0 + 160},{478} L{x0 + 170},{478} Z" fill="#94a3b8"/>')
    o.append(f'<rect x="{x0 + 30}" y="500" width="300" height="230" rx="14" fill="white" stroke="#bfdbfe"/>')
    for (mx, my, sd, n) in [(185, 560, 16, 70), (130, 640, 14, 45), (250, 600, 13, 45), (200, 672, 11, 35)]:
        is_norm = (mx, my) == (185, 560)
        for px, py in rng.normal([mx, my], sd, (n, 2)):
            px, py = np.clip(px, 45, 315), np.clip(py, 512, 700)
            col = NORM if is_norm else MAL
            o.append(f'<circle cx="{x0 + px:.1f}" cy="{py:.1f}" r="3.3" fill="{col}" '
                     f'stroke="{NORM_DARK if is_norm else MAL_DARK}" stroke-width=".6" opacity=".9"/>')
    o.append(text(x0 + 180, 720, "cell embedding space (schematic)", 12.5, "600", MUTED, "middle"))

    # 3 evaluate
    x0 = 790
    o.append(arrow(763, 435))
    o.append(panel(x0, 110, 320, 650, "3", "Evaluate", "#f0fdf4", "#15803d"))
    cards = [("T1  Unsupervised", ["Leiden clusters & kNN purity", "no labels used"]),
             ("T2  Held-out patient probe", ["leave-one-patient-out; full or", "10/50 cells per class per patient;",
                                             "label noise 0–20 %"]),
             ("T3  Cross-cancer transfer", [f"train on {n_cancer - 1}, test on held-out", "type; 1 study per cancer type",
                                            "(= leave-one-study-out)"]),
             ("T4  Patient structure", ["normal and malignant cells", "scored separately"])]
    for i, (ttl, desc) in enumerate(cards):
        y = 172 + i * 122
        o.append(f'<rect x="{x0 + 16}" y="{y}" width="288" height="110" rx="12" fill="white" stroke="#bbf7d0"/>'
                 f'<rect x="{x0 + 16}" y="{y}" width="6" height="110" rx="3" fill="#15803d"/>')
        ix, iy = x0 + 50, y + 55
        if i == 0:
            for (dx, dy, c) in [(-10, -8, MAL), (10, 10, NORM)]:
                o.append(f'<circle cx="{ix + dx}" cy="{iy + dy}" r="14" fill="none" stroke="#94a3b8" stroke-dasharray="3 2"/>')
                o.extend(f'<circle cx="{ix + dx + a}" cy="{iy + dy + b}" r="3" fill="{c}"/>'
                         for a, b in [(-5, -4), (5, -3), (0, 5), (-4, 4)])
        elif i == 1:
            for k, c in enumerate(["#cbd5e1", "#cbd5e1", "#f59e0b"]):
                o.append(person(ix - 16 + k * 16, iy + 8, 20, c))
            o.append(text(ix + 16, iy - 14, "?", 14, "bold", "#b45309", "middle"))
        elif i == 2:
            o.append(f'<circle cx="{ix - 16}" cy="{iy}" r="9" fill="{MAL}"/><circle cx="{ix + 16}" cy="{iy}" r="9" '
                     f'fill="none" stroke="{MAL_DARK}" stroke-width="2" stroke-dasharray="3 2"/>'
                     f'<line x1="{ix - 5}" y1="{iy}" x2="{ix + 2}" y2="{iy}" stroke="{INK}" stroke-width="1.8" '
                     f'marker-end="url(#ah)"/>')
        else:
            o.append(normal(ix - 10, iy - 10, 7) + normal(ix + 8, iy - 12, 7, "#fde68a", "#b45309")
                     + malignant(ix - 10, iy + 12, 8, 1) + malignant(ix + 10, iy + 12, 8, 2.5))
        o.append(text(x0 + 84, y + 30, ttl, 15, "bold"))
        o.append(lines(x0 + 84, y + 52, desc, 13, 17, fill=MUTED))
    o.append(text(x0 + 160, 678, "Patient = unit of replication", 15, "bold", "#15803d", "middle"))
    o.append(lines(x0 + 160, 700, ["patient-bootstrap 95 % CI ·", "paired Wilcoxon · Holm correction"],
                   13, 17, fill=MUTED, anchor="middle"))

    # 4 findings
    x0 = 1140
    o.append(arrow(1113, 435))
    o.append(panel(x0, 110, 430, 650, "4", "Key findings", "#fff7ed", "#c2410c"))
    o.append(text(x0 + 24, 180, f"10-shot probe AUROC, mean of {n_pat} held-out patients", 14, "bold", INK))
    o.append(text(x0 + 24, 197, "10 training cells per class per patient · no label noise", 12, "normal", MUTED))
    order = FMS + ["X_randproj", "X_harmony", "X_hvg_pca", "X_scgpt_random"]
    bx, bw, lo, y0, rh = x0 + 160, 196, 0.80, 208, 30
    o.append(f'<rect x="{x0 + 14}" y="{y0 - 3}" width="400" height="{4 * rh}" rx="8" fill="#fde68a" opacity=".35"/>')
    for k, m in enumerate(order):
        lab, col = MODEL[m]
        y = y0 + k * rh
        v = fs10[m]
        o.append(text(bx - 8, y + 16, lab, 13, "600" if m in FMS else "normal", INK, "end"))
        o.append(f'<rect x="{bx}" y="{y + 3}" width="{(v - lo) / (1 - lo) * bw:.1f}" height="18" rx="4" fill="{col}"/>')
        o.append(text(bx + (v - lo) / (1 - lo) * bw + 5, y + 17, f"{v:.3f}", 12, "normal", MUTED))
    ya = y0 + 7 * rh + 4
    o.append(f'<line x1="{bx}" y1="{ya}" x2="{bx + bw}" y2="{ya}" stroke="#94a3b8"/>')
    for t in (0.8, 0.9, 1.0):
        tx = bx + (t - lo) / (1 - lo) * bw
        o.append(f'<line x1="{tx}" y1="{ya}" x2="{tx}" y2="{ya + 4}" stroke="#94a3b8"/>'
                 + text(tx, ya + 17, f"{t:.2f}", 11, "normal", MUTED, "middle"))
    o.append(text(bx + bw / 2, ya + 32, "axis starts at 0.80", 11, "normal", MUTED, "middle"))
    bxk = x0 + 404
    o.append(f'<path d="M{bxk - 6},{y0 + 2} H{bxk} V{y0 + 4 * rh - 6} H{bxk - 6}" fill="none" stroke="#c2410c" stroke-width="2"/>')
    p_txt = "Holm p < 0.001" if p_max < 1e-3 else f"Holm p ≤ {p_max:.3f}"
    o.append(lines(x0 + 24, ya + 50, [f"Random projection recovers {min(share):.0%}–{max(share):.0%} of the FM gain",
                                      "over HVG + PCA; FMs keep a small, significant edge",
                                      f"(+{min(edge):.3f} to +{max(edge):.3f}, {p_txt})"],
                   13, 17, weight="bold", fill="#c2410c"))
    nx = normal_excess[FMS]
    finds = [("Standard task is saturated",
              [f"malignant vs all normal, full probe: AUROC ≈ {full['X_randproj']:.2f}",
               "for FMs and baselines, incl. random projection"]),
             ("Cross-cancer transfer splits the models",
              [f"UCE highest label-free (mean {t3_mean['X_uce']:.2f}, tied with Harmony*);",
               f"Geneformer inverted on lymphoma ({t3m['follicular_lymphoma', 'X_geneformer']:.2f}, one study)"]),
             ("Patient effects persist",
              [f"normal-cell patient structure: FMs {nx.min():.2f}–{nx.max():.2f}, HVG + PCA "
               f"{normal_excess['X_hvg_pca']:.2f}", "(technical batch or real biology)"]),
             ("Label noise erodes the edge",
              [f"10 % flipped labels: random projection {n10['X_randproj']:.3f},",
               f"FMs {n10[FMS].min():.3f}–{n10[FMS].max():.3f}"])]
    for i, (hd, desc) in enumerate(finds):
        y = 530 + i * 56
        o.append(f'<circle cx="{x0 + 36}" cy="{y + 8}" r="13" fill="#c2410c"/>'
                 + text(x0 + 36, y + 13, str(i + 1), 14, "bold", "white", "middle"))
        o.append(text(x0 + 60, y + 13, hd, 14.5, "bold") + lines(x0 + 60, y + 30, desc, 12, 15, fill=MUTED))

    defs = ('<defs><filter id="sh" x="-5%" y="-5%" width="110%" height="110%"><feDropShadow dx="0" dy="2" '
            'stdDeviation="3" flood-color="#0f172a" flood-opacity=".10"/></filter>'
            '<marker id="ah" markerWidth="7" markerHeight="7" refX="5" refY="3.5" orient="auto">'
            f'<path d="M0,0 L7,3.5 L0,7 Z" fill="{INK}"/></marker></defs>')
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
            f'font-family="{FONT}"><rect width="{W}" height="{H}" fill="white"/>{defs}{"".join(o)}</svg>')


def chrome():
    for c in ["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", "google-chrome", "chromium", "chrome"]:
        if Path(c).exists() or shutil.which(c):
            return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="results/figures")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    svg = out / "graphical_abstract.svg"
    svg.write_text(build(Path(a.results)))
    print("wrote", svg)
    exe = chrome()
    if not exe:
        print("Chrome not found: PNG/PDF skipped"); return
    with tempfile.TemporaryDirectory() as tmp:
        html = Path(tmp) / "ga.html"
        html.write_text(f'<html><head><style>@page{{size:{W}px {H}px;margin:0}}body{{margin:0}}</style></head>'
                        f'<body>{svg.read_text()}</body></html>')
        common = [exe, "--headless=new", "--disable-gpu", "--hide-scrollbars", f"--user-data-dir={tmp}/prof"]
        subprocess.run(common + [f"--window-size={W},{H}", "--force-device-scale-factor=2",
                                 f"--screenshot={(out / 'graphical_abstract.png').resolve()}", html.as_uri()],
                       check=True, capture_output=True)
        subprocess.run(common + ["--no-pdf-header-footer", f"--print-to-pdf={(out / 'graphical_abstract.pdf').resolve()}",
                                 html.as_uri()], check=True, capture_output=True)
    print("wrote", out / "graphical_abstract.png", "and .pdf")


if __name__ == "__main__":
    main()
