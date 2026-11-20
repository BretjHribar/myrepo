"""Record local compute and optionally exercise CUDA and higher derivatives.

Run from any directory. No packages are installed or system settings changed.
The synthetic smoke test establishes readiness, not chemistry performance.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time


def command(args, timeout=30):
    try:
        result = subprocess.run(args, capture_output=True, timeout=timeout)
        raw = result.stdout
        encoding = "utf-16-le" if b"\x00" in raw else "utf-8-sig"
        return {
            "returncode": result.returncode,
            "stdout": raw.decode(encoding, errors="replace").strip(),
            "stderr": result.stderr.decode("utf-8", errors="replace").strip(),
        }
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"error": str(exc)}


def cuda_smoke(torch):
    torch.manual_seed(20261116)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda:0")
    results = {}
    for name, dtype in (("float32", torch.float32), ("float64", torch.float64)):
        torch.cuda.reset_peak_memory_stats(device)
        model = torch.nn.Sequential(
            torch.nn.Linear(4, 32), torch.nn.Tanh(),
            torch.nn.Linear(32, 32), torch.nn.Tanh(),
            torch.nn.Linear(32, 1),
        ).to(device=device, dtype=dtype)
        n = torch.rand(8192, device=device, dtype=dtype).requires_grad_(True)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        def step():
            optimizer.zero_grad(set_to_none=True)
            n.grad = None
            # Local and neighboring values stand in for density descriptors.
            features = torch.stack((n, n.roll(1), n.roll(-1), n.square()), -1)
            energy = (n * model(features).squeeze(-1)).sum() / n.numel()
            potential = torch.autograd.grad(energy, n, create_graph=True)[0]
            # Backward traverses the energy derivative, as field training needs.
            loss = energy.square() + (potential * n.numel()).square().mean()
            loss.backward()
            finite = bool(torch.isfinite(loss).item()) and all(
                p.grad is not None and bool(torch.isfinite(p.grad).all().item())
                for p in model.parameters()
            )
            if not finite:
                raise RuntimeError(f"Nonfinite loss/gradient for {name}")
            optimizer.step()
            return float(loss.detach())

        step()  # Warmup includes optimizer state allocation.
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        for _ in range(5):
            last_loss = step()
        torch.cuda.synchronize(device)
        results[name] = {
            "synthetic_field_training_step_ms": (time.perf_counter() - start) / 5 * 1000,
            "steps_timed": 5,
            "points": 8192,
            "last_loss": last_loss,
            "finite_gradients": True,
            "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20,
        }
        del optimizer, model, n
        torch.cuda.empty_cache()
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    workspace = Path(__file__).resolve().parents[1]
    report = {
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "logical_cpus": os.cpu_count(),
        "workspace_free_gib": shutil.disk_usage(workspace).free / 2**30,
        "packages": {},
    }
    for package in ("torch", "numpy", "scipy", "pyscf", "gpu4pyscf", "cupy"):
        try:
            report["packages"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            report["packages"][package] = None
    gpu = command([
        "nvidia-smi", "--query-gpu=name,memory.total,compute_cap,driver_version",
        "--format=csv,nounits",
    ])
    report["nvidia_smi"] = gpu
    if gpu.get("returncode") == 0:
        report["gpus"] = list(csv.DictReader(io.StringIO(gpu["stdout"]), skipinitialspace=True))
    if sys.platform == "win32":
        host = command([
            "powershell.exe", "-NoProfile", "-Command",
            "$cpu=Get-CimInstance Win32_Processor; "
            "$ram=(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory; "
            "@{cpu_name=$cpu.Name;physical_cores=$cpu.NumberOfCores;"
            "logical_processors=$cpu.NumberOfLogicalProcessors;"
            "physical_memory_bytes=$ram} | ConvertTo-Json -Compress",
        ])
        report["host_query"] = host
        if host.get("returncode") == 0:
            report["host"] = json.loads(host["stdout"])
        report["wsl_distributions"] = command(["wsl.exe", "--list", "--verbose"])
    try:
        import torch
        report["torch"] = {
            "version": torch.__version__,
            "bundled_cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count(),
        }
        if torch.cuda.is_available():
            properties = torch.cuda.get_device_properties(0)
            report["torch"].update({
                "device_name": properties.name,
                "total_memory_bytes": properties.total_memory,
                "compute_capability": [properties.major, properties.minor],
                "multiprocessor_count": properties.multi_processor_count,
            })
            if args.smoke:
                report["cuda_smoke"] = cuda_smoke(torch)
    except Exception as exc:
        report["torch_error"] = f"{type(exc).__name__}: {exc}"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if args.smoke and "cuda_smoke" not in report:
        raise SystemExit("CUDA smoke test did not complete; see the saved report")


if __name__ == "__main__":
    main()
