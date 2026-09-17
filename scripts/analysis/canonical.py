"""Loader for the canonical runs behind the paper.

results/paper/<model>/adaptive_evaluation.json holds one evaluated record per run of the four
conditions BASE, SYN, TYPO and UNDER (built by build_snapshot.py from results/experiments/<model>_<condition>/).
Every record carries `_source_experiment` (the run directory) and `_trace_id` (the trace file name
without extension). The raw trace is results/experiments/<_source_experiment>/traces/<_trace_id>.json.

load_traces() returns the records of one model, keeping only the latest trace per
(query, approach, run number) in case a run was restarted.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "results" / "paper"
EXPERIMENTS = ROOT / "results" / "experiments"

MODELS = [("deepseek", "DeepSeek-V3.2"), ("gpt5", "GPT-5.4"), ("qwen", "Qwen3.5-27B")]
SETS = ["BASE", "SYN", "TYPO", "UNDER"]
CONFIGS = [("lexical", "agentic_grep"), ("semantic", "agentic_semantic")]


def raw_trace(record: dict) -> dict:
    path = EXPERIMENTS / record["_source_experiment"] / "traces" / f"{record['_trace_id']}.json"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_traces(folder: str, with_raw: bool = False) -> list[dict]:
    """Return the canonical records of one model (folder: deepseek, gpt5 or qwen)."""
    with open(PAPER / folder / "adaptive_evaluation.json", encoding="utf-8") as f:
        evaluation = json.load(f)
    best: dict[tuple, tuple[str, dict]] = {}
    for record in evaluation["traces"].values():
        raw = raw_trace(record)
        key = (record.get("query_id"), record.get("approach"), raw.get("run_number"), record["_source_experiment"])
        timestamp = raw.get("timestamp", "")
        if key not in best or timestamp > best[key][0]:
            record = dict(record)
            if with_raw:
                record["_raw"] = raw
            best[key] = (timestamp, record)
    return [rec for _, rec in best.values()]


def load_run(run: str) -> list[dict]:
    """Raw traces of one run directory (results/experiments/<run>/traces/*.json)."""
    out = []
    for path in sorted((EXPERIMENTS / run / "traces").glob("*.json")):
        with open(path, encoding="utf-8") as f:
            out.append(json.load(f))
    return out
