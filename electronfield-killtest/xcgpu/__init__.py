"""GPU density-functional pipeline (gpu4pyscf) for full-suite evaluation and model training.

All SCF, integral, grid and training numerics run on CUDA. Host work is limited to
orchestration, basis bookkeeping and file I/O on one pinned core.
"""
