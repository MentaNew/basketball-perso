"""Experiment log: every training run appends one JSON line to models/metrics.jsonl (roadmap X).

So "did change X help?" is answered from the log, not from memory. Each line records when,
which commit (and whether the code had uncommitted changes), the model, its settings and its
metrics. Read it back with `read_runs()` (e.g. in a notebook).
"""
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
LOG_PATH = ROOT / "models" / "metrics.jsonl"


def _git(*args: str) -> str | None:
    try:
        result = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def log_run(model: str, metrics: dict, params: dict | None = None, path: Path = LOG_PATH) -> dict:
    """Append one run to the log and return it. `metrics`/`params` must be JSON-friendly
    (numpy numbers are converted)."""
    entry = {
        "time": datetime.now(UTC).isoformat(timespec="seconds"),
        "commit": _git("rev-parse", "--short", "HEAD"),
        "uncommitted_changes": bool(_git("status", "--porcelain")),  # results may not be reproducible
        "model": model,
        "params": params or {},
        "metrics": metrics,
    }
    path.parent.mkdir(exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(entry, default=float) + "\n")
    return entry


def read_runs(path: Path = LOG_PATH) -> pd.DataFrame:
    """All logged runs, one row each, with params and metrics flattened into columns."""
    if not path.exists():
        return pd.DataFrame()
    return pd.json_normalize([json.loads(line) for line in path.read_text().splitlines() if line])
