#!/usr/bin/env bash
# Embed every dataset (ready ∪ ready_matched cells) with one model. Run from the repo root.
#   scripts/embed_all.sh <python-of-FM-env> <model> <model-dir> [<cache label>] [extra embed_fm.py args...]
set -euo pipefail
PY=$1; MODEL=$2; DIR=$3; NAME=${4:-$2}; shift $(( $# < 4 ? $# : 4 ))
export NUMBA_CACHE_DIR=${NUMBA_CACHE_DIR:-/tmp/nb} PYTORCH_ENABLE_MPS_FALLBACK=1
for f in data/ready/*.h5ad; do
  ds=$(basename "$f" .h5ad)
  [ -f "cache/emb/${ds}__${NAME}.npy" ] && { echo "skip $ds (cached)"; continue; }
  files=("$f"); [ -f "data/ready_matched/$ds.h5ad" ] && files+=("data/ready_matched/$ds.h5ad")
  "$PY" -W ignore scripts/embed_fm.py --model "$MODEL" --model-dir "$DIR" --name "$NAME" --h5ad "${files[@]}" "$@"
done
