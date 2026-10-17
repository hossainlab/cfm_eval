"""Compute and cache foundation-model embeddings. Run in the FM environment (scGPT/Geneformer pin old torch).
   python scripts/embed_fm.py --model scgpt --model-dir weights/scGPT_human \
       (--model uce --model-dir <UCE repo clone with model_files/>)
       --h5ad data/ready/cxg_gbm_darmanis.h5ad data/ready_matched/cxg_gbm_darmanis.h5ad [--max-cells 1000]
All --h5ad files must share a stem (one dataset); the union of their cells is embedded once and saved as
cache/emb/<dataset>__<model>.npy + .obs_names.txt, which serves both the ready and the lineage-matched task.
FM cell embeddings depend only on each cell's own counts, so embedding the union is equivalent to embedding each set.
--name: cache label (default = --model), e.g. scgpt_random for the random-weight control built by
cfm_eval.embed_fm.make_random_scgpt.
--max-cells: timing pass on a random subset, written to <emb-dir>/timing/ and never used by the benchmark.
--chunk-cells: datasets larger than this are embedded in a fresh subprocess per chunk and then stacked. On Apple
MPS, memory grows with every new batch shape (scGPT pads each batch to its own longest cell) and is only released
when the process exits; 46k cells in one process ran out of memory on a 24 GB Mac. Finished chunks are kept in
<emb-dir>/parts/, so an interrupted dataset resumes where it stopped.
"""
import argparse, subprocess, sys, time
from pathlib import Path
import numpy as np, anndata as ad
from cfm_eval.data import read_std, write_std
from cfm_eval import embed_fm

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["scgpt", "geneformer", "uce"])
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--name", default=None)
    ap.add_argument("--h5ad", nargs="+", required=True)
    ap.add_argument("--emb-dir", default="cache/emb")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-cells", type=int, default=None)
    ap.add_argument("--workdir", default="cache/geneformer_tmp")
    ap.add_argument("--chunk-cells", type=int, default=5000)
    ap.add_argument("--cell-range", type=int, nargs=2, default=None, help=argparse.SUPPRESS)  # set by the parent
    args = ap.parse_args()
    names = {Path(f).stem for f in args.h5ad}
    assert len(names) == 1, f"--h5ad files must be one dataset, got {names}"
    name = names.pop(); label = args.name or args.model
    parts = [read_std(f) for f in args.h5ad]
    a = ad.concat(parts, join="inner", merge="same")
    a = a[~a.obs_names.duplicated()].copy()
    a.var = parts[0].var.loc[a.var_names]
    a.layers["counts"] = a.X.copy()
    emb_dir = Path(args.emb_dir)
    if args.max_cells and args.max_cells < a.n_obs:
        a = a[np.random.default_rng(0).choice(a.n_obs, args.max_cells, replace=False)].copy()
        emb_dir = emb_dir / "timing"
    if args.cell_range:
        a = a[args.cell_range[0]:args.cell_range[1]].copy()
    elif a.n_obs > args.chunk_cells:
        return run_chunked(a, args, name, label, emb_dir)
    print(f"{name}: embedding {a.n_obs} cells x {a.n_vars} genes with {label}", flush=True)
    t0 = time.time()
    if args.model == "scgpt":
        import json
        vocab = json.load(open(Path(args.model_dir) / "vocab.json"))
        cov, _ = embed_fm.scgpt_checkpoint_coverage(args.model_dir)
        print(f"  scGPT vocab match (feature_name): {embed_fm.gene_match(a, vocab, 'feature_name'):.3f}; "
              f"checkpoint fills {cov:.3f} of parameters (only the unused cls_decoder should be missing)")
        E = embed_fm.scgpt_embed(a, args.model_dir, batch_size=args.batch_size, device=args.device)
    elif args.model == "uce":
        E = embed_fm.uce_embed(a, args.model_dir, args.workdir, batch_size=args.batch_size)
    else:
        print(f"  genes with Ensembl ID: {a.var['ensembl_id'].notna().mean():.3f}")
        E = embed_fm.geneformer_embed(a, args.model_dir, args.workdir, batch_size=args.batch_size, device=args.device)
    dt = time.time() - t0
    p = embed_fm.save_cached(E, a.obs_names, emb_dir, name, label)
    print(f"  {E.shape} -> {p} in {dt:.0f}s ({1000 * dt / a.n_obs:.1f}s per 1k cells)")

def run_chunked(a, args, name, label, emb_dir):
    parts = emb_dir / "parts" / f"{name}__{label}"
    src = parts / f"{name}.h5ad"
    if not src.exists():
        parts.mkdir(parents=True, exist_ok=True)
        write_std(a, src)
    starts = range(0, a.n_obs, args.chunk_cells)
    print(f"{name}: {a.n_obs} cells in {len(starts)} chunks of {args.chunk_cells} with {label}", flush=True)
    t0 = time.time()
    for k, i in enumerate(starts):
        out = parts / f"chunk{k:03d}"
        if (out / f"{name}__{label}.npy").exists():
            continue
        cmd = [sys.executable, "-W", "ignore", __file__, "--model", args.model, "--model-dir", args.model_dir,
               "--name", label, "--h5ad", str(src), "--emb-dir", str(out), "--device", args.device,
               "--batch-size", str(args.batch_size), "--workdir", args.workdir,
               "--cell-range", str(i), str(min(i + args.chunk_cells, a.n_obs))]
        subprocess.run(cmd, check=True)
    E = np.vstack([embed_fm.load_cached(a[i:i + args.chunk_cells], parts / f"chunk{k:03d}", name, label)
                   for k, i in enumerate(starts)])
    dt = time.time() - t0
    p = embed_fm.save_cached(E, a.obs_names, emb_dir, name, label)
    print(f"  {E.shape} -> {p} in {dt:.0f}s ({1000 * dt / a.n_obs:.1f}s per 1k cells, incl. resumed chunks)")

if __name__ == "__main__":
    main()
