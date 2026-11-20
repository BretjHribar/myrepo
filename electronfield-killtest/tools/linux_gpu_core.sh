#!/usr/bin/env bash
# Same as linux_gpu.sh, but the first argument selects which logical core the job is pinned to.
# User allowance, 20 November 2026: three host cores in total (cores 0, 1 and 2), one thread each.
set -euo pipefail
core="$1"
shift
case "$core" in
    0|1|2) ;;
    *) echo "core must be 0, 1 or 2 (three host cores are allowed)" >&2; exit 2 ;;
esac
ulimit -c 0
export XC_GPU_ONLY=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export BLIS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MAX_JOBS=1
export PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0
export TMPDIR=/tmp/xc-project XDG_CACHE_HOME=/opt/xc-cache
export TORCH_HOME=/opt/xc-cache/torch HF_HOME=/opt/xc-cache/huggingface
export CUDA_CACHE_PATH=/opt/xc-cache/cuda
for gpu_lib in /opt/xc-env/lib/python3.12/site-packages/nvidia/*/lib /opt/xc-env/lib/python3.12/site-packages/cutensor/lib; do
    if [[ -d "$gpu_lib" ]]; then
        export LD_LIBRARY_PATH="$gpu_lib:${LD_LIBRARY_PATH:-}"
    fi
done
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"
cd /mnt/c/Users/breth/myrepo/electronfield-killtest
exec taskset -c "$core" /opt/xc-env/bin/python "$@"
