#!/usr/bin/env bash
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export BLIS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MAX_JOBS=1
export CMAKE_BUILD_PARALLEL_LEVEL=1 MAKEFLAGS=-j1
export PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1
export TMPDIR=/tmp/xc-project
mkdir -p "$TMPDIR"
echo 'Installing the compact chemistry environment (single CPU core).'
apt-get update -qq
apt-get install -y -qq --no-install-recommends python3 python3-venv ca-certificates libgomp1
python3 -m venv /opt/xc-env
/opt/xc-env/bin/python -m pip install --quiet --upgrade pip
/opt/xc-env/bin/python -m pip install --quiet 'torch==2.7.1' --index-url https://download.pytorch.org/whl/cu126
/opt/xc-env/bin/python -m pip install --quiet 'pyscf>=2.8,<3' 'cupy-cuda12x==13.4.1' 'cutensor-cu12==2.2.0' gpu4pyscf-cuda12x
/opt/xc-env/bin/python -m pip freeze
apt-get clean
echo 'Chemistry environment installation complete.'
