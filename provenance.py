"""provenance.py

Simple utilities to save and load experiment run metadata (runs.json) in
`static/uploads/` so experiments are reproducible and can be benchmarked.
"""
import os
import json
import uuid
from datetime import datetime

RUNS_FILE = os.path.join(os.path.dirname(__file__), "static", "uploads", "runs.json")


def ensure_runs_dir():
    os.makedirs(os.path.dirname(RUNS_FILE), exist_ok=True)


def load_runs():
    ensure_runs_dir()
    if os.path.exists(RUNS_FILE):
        try:
            with open(RUNS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def save_runs(runs_list):
    ensure_runs_dir()
    with open(RUNS_FILE, "w", encoding="utf-8") as f:
        json.dump(runs_list, f, indent=2)


def save_run(run_metadata: dict):
    """Append a run record to the runs file. Adds run_id and timestamps

    Minimal validation: ensures a timestamp and run_id exist.
    """
    runs = load_runs()
    run = dict(run_metadata)
    run.setdefault("run_id", uuid.uuid4().hex)
    run.setdefault("created_at", datetime.utcnow().isoformat())
    runs.insert(0, run)
    save_runs(runs)
    return run


def clear_runs():
    save_runs([])
    return []
