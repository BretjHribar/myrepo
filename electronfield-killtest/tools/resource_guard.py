"""Launch bounded project work with one-core affinity and storage checks.

Example: python tools/resource_guard.py --reserve-gb 4 --timeout 900 -- command args
Storage polling is a backstop, not a replacement for bounded job outputs.
"""
from __future__ import annotations

import argparse
import ctypes
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "electronfield-killtest/research/resource_budget.json"
THREAD_VARIABLES = (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
    "CMAKE_BUILD_PARALLEL_LEVEL", "MAX_JOBS",
)


def pin_cpu():
    if os.name == "nt":
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = ctypes.c_void_p
        kernel.SetProcessAffinityMask.argtypes = (ctypes.c_void_p, ctypes.c_size_t)
        kernel.SetProcessAffinityMask.restype = ctypes.c_int
        if not kernel.SetProcessAffinityMask(kernel.GetCurrentProcess(), 1):
            raise ctypes.WinError(ctypes.get_last_error())
        return [0]
    allowed = os.sched_getaffinity(0)
    chosen = min(allowed)
    os.sched_setaffinity(0, {chosen})
    return sorted(os.sched_getaffinity(0))


def project_bytes():
    total = 0
    for base, dirs, files in os.walk(ROOT, followlinks=False):
        for name in files:
            path = Path(base) / name
            try:
                total += path.lstat().st_size
            except FileNotFoundError:
                pass  # Temporary file vanished between directory scan and stat.
    disk = json.loads(POLICY.read_text(encoding="utf-8"))["disk"]
    for external in disk.get("external_accounted_paths", []):
        path = Path(external)
        if path.resolve().is_relative_to(ROOT.resolve()):
            continue  # Already counted by the repository walk.
        try:
            if path.is_file():
                total += path.stat().st_size
            elif path.is_dir():
                for base, _, files in os.walk(path, followlinks=False):
                    for name in files:
                        try:
                            total += (Path(base)/name).lstat().st_size
                        except FileNotFoundError:
                            pass
        except (FileNotFoundError, PermissionError):
            pass  # The explicit legacy upper bound below remains reserved.
    return total+disk.get("legacy_external_upper_bound_bytes", 0)


def storage_check(reserve=0):
    policy = json.loads(POLICY.read_text(encoding="utf-8"))["disk"]
    used = project_bytes()
    free = shutil.disk_usage(ROOT).free
    report = {"project_bytes": used, "project_gb": round(used / 1e9, 4),
              "volume_free_gb": round(free / 1e9, 3), "reserved_growth_gb": reserve / 1e9,
              "hard_limit_bytes": policy["hard_limit_bytes"]}
    if used + reserve > policy["working_ceiling_bytes"]:
        raise RuntimeError(f"32 GB working ceiling would be exceeded: {report}")
    if free < reserve + 2_000_000_000:
        raise RuntimeError(f"Insufficient destination space including 2 GB free reserve: {report}")
    return report


def configure_environment():
    for name in THREAD_VARIABLES:
        os.environ[name] = "1"
    store = ROOT / ".research"
    for name, relative in {
        "TMP": "tmp", "TEMP": "tmp", "TMPDIR": "tmp",
        "PIP_CACHE_DIR": "cache/pip", "HF_HOME": "cache/huggingface",
        "TORCH_HOME": "cache/torch", "XDG_CACHE_HOME": "cache",
        "CUDA_CACHE_PATH": "cache/cuda",
    }.items():
        path = store / relative
        path.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(path)
    os.environ["PYTHONUNBUFFERED"] = "1"
    os.environ["PIP_NO_CACHE_DIR"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["MAKEFLAGS"] = "-j1"
    return pin_cpu()


def write_storage_report(destination, report):
    destination = destination.resolve()
    if not destination.is_relative_to(ROOT.resolve()):
        raise ValueError("Resource snapshots must stay inside the project")
    policy = json.loads(POLICY.read_text(encoding="utf-8"))
    snapshot = {"created_utc": datetime.now(timezone.utc).isoformat(), **report,
                "working_ceiling_bytes": policy["disk"]["working_ceiling_bytes"],
                "external_accounted_paths": policy["disk"].get("external_accounted_paths", []),
                "legacy_external_upper_bound_bytes": policy["disk"].get("legacy_external_upper_bound_bytes", 0),
                "wsl_runtime": policy.get("wsl_runtime"), "additional_drive_assumed": False,
                "scientific_numerical_work_device": "CUDA only"}
    destination.write_text(json.dumps(snapshot, indent=2)+"\n", encoding="utf-8")


def stop_tree(process):
    if os.name == "nt":
        subprocess.run(["taskkill.exe", "/PID", str(process.pid), "/T", "/F"],
                       capture_output=True)
    else:
        os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        if os.name != "nt":
            os.killpg(process.pid, signal.SIGKILL)
        process.kill()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reserve-gb", type=float, default=0)
    parser.add_argument("--timeout", type=float, default=3600)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--report", type=Path, help="Save a bounded resource snapshot inside the project")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.reserve_gb < 0 or args.timeout <= 0:
        parser.error("reserve must be nonnegative and timeout positive")
    affinity = configure_environment()
    initial = {"affinity": affinity, **storage_check(int(args.reserve_gb * 1e9))}
    if args.report:
        write_storage_report(args.report, initial)
    print(json.dumps(initial), flush=True)
    if args.check_only:
        return
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a command is required")
    started = time.monotonic()
    process = subprocess.Popen(command, start_new_session=os.name != "nt")
    try:
        while process.poll() is None:
            time.sleep(2)
            storage_check()
            if time.monotonic() - started > args.timeout:
                raise TimeoutError(f"Job exceeded {args.timeout} seconds")
    except BaseException:
        stop_tree(process)
        raise
    final = {"elapsed_seconds": round(time.monotonic() - started, 3),
             "returncode": process.returncode, "affinity": affinity, **storage_check()}
    if args.report:
        write_storage_report(args.report, final)
    print(json.dumps(final), flush=True)
    raise SystemExit(process.returncode)


if __name__ == "__main__":
    main()
