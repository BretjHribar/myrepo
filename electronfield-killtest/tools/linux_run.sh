#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export BLIS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MAX_JOBS=1
export CMAKE_BUILD_PARALLEL_LEVEL=1 MAKEFLAGS=-j1
export PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0
export TMPDIR=/tmp/xc-project
export XDG_CACHE_HOME=/opt/xc-cache
export TORCH_HOME=/opt/xc-cache/torch
export HF_HOME=/opt/xc-cache/huggingface
export CUDA_CACHE_PATH=/opt/xc-cache/cuda
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"
cd /mnt/c/Users/breth/myrepo/electronfield-killtest
exec taskset -c 0 /opt/xc-env/bin/python "$@"
