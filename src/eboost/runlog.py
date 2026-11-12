"""Append-only run log: one JSON line per run with config hash, git commit,
data hash, seed, hardware and wall-clock (pre-registration, Provenance)."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np


def config_hash(config: dict[str, Any]) -> str:
    blob = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def data_hash(*arrays: np.ndarray) -> str:
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()[:16]


def git_state(repo: Path | None = None) -> dict[str, Any]:
    if os.environ.get("EBOOST_COMMIT"):  # code shipped as a bundle, no .git
        return {"commit": os.environ["EBOOST_COMMIT"], "dirty": False, "from": "bundle"}
    repo = repo or Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                                capture_output=True, text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "-C", str(repo), "status", "--porcelain"],
                                    capture_output=True, text=True, check=True).stdout.strip())
    except subprocess.CalledProcessError:
        commit, dirty = "none", True
    return {"commit": commit, "dirty": dirty}


def hardware() -> dict[str, Any]:
    info: dict[str, Any] = {"host": platform.node(), "machine": platform.machine(),
                            "cpus": os.cpu_count()}
    try:
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10, check=False)
        if gpu.returncode == 0 and gpu.stdout.strip():
            info["gpus"] = gpu.stdout.strip().splitlines()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return info


def append(log_path: Path, record: dict[str, Any]) -> dict[str, Any]:
    """Append one record. Opens in append mode only, never rewrites."""
    import xgboost

    full = {"time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "git": git_state(), "hardware": hardware(),
            "versions": {"xgboost": xgboost.__version__, "numpy": np.__version__},
            **record}
    if "config" in record:
        full["config_hash"] = config_hash(record["config"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as f:
        f.write(json.dumps(full, sort_keys=True, default=str) + "\n")
    return full
