"""Call configure() before importing numerical packages."""
import os
from pathlib import Path


def configure(gpu_only_compatible=False):
    if not gpu_only_compatible:
        raise RuntimeError("GPU-only research is active. This legacy entry point may run CPU numerical work; use a CUDA-only entry point.")
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "MAX_JOBS"):
        os.environ[key] = "1"
    os.environ["PYTHONUNBUFFERED"] = "1"
    if hasattr(os, "sched_getaffinity"):
        os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    root = Path(__file__).resolve().parents[2] / ".research"
    root.mkdir(exist_ok=True)
    return root


def torch_setup(seed=20261116):
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by the user's GPU-only instruction; CPU fallback is disabled")
    return "cuda"


def require_speed_authorization():
    """Keep historical speed entry points behind the current user policy."""
    import json
    policy = Path(__file__).resolve().parents[1] / "research/resource_budget.json"
    scope = json.loads(policy.read_text(encoding="utf-8"))["research_scope"]
    if not scope.get("speed_work_authorized", False):
        raise RuntimeError("Further speed work is on hold under the latest user instruction. Use accuracy/research entry points until the accuracy gate is met.")
