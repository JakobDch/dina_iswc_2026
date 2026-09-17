"""Semantic search calls of the runs under semantic retrieval (Section 6.2, "rank of the target").

For the BASE and SYN runs with the embedding-based linking tools, counts the calls of the class and
property search tools, the share of calls that asked for more than the default of k = 5 candidates,
the largest k requested, and the median over runs of the largest k used within a run.

Run from the repository root:  python scripts/analysis/syn_search_stats.py
"""
from __future__ import annotations

import ast
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from canonical import MODELS, load_traces  # noqa: E402

TOOLS = ("search_classes", "search_properties")
DEFAULT_K = 5


def calls_of(raw: dict) -> list[int]:
    ks = []
    for t in raw.get("tool_invocations") or []:
        if t.get("tool_name") not in TOOLS:
            continue
        try:
            args = ast.literal_eval(t.get("input_summary") or "{}")
        except Exception:
            args = {}
        ks.append(int(args.get("top_k") or DEFAULT_K))
    return ks


pooled: dict[str, list[list[int]]] = {"BASE": [], "SYN": []}
print(f"{'backbone':<14}{'set':<5}{'runs':>5}{'calls':>7}{'k>5':>7}{'max k':>6}{'median run max':>15}")
for folder, label in MODELS:
    per_set: dict[str, list[list[int]]] = {"BASE": [], "SYN": []}
    for r in load_traces(folder, with_raw=True):
        if r.get("approach") != "agentic_semantic" or r.get("query_set") not in per_set:
            continue
        per_set[r["query_set"]].append(calls_of(r.get("_raw") or {}))
    for qset, runs in per_set.items():
        pooled[qset].extend(runs)
        ks = [k for run in runs for k in run]
        raised = 100 * sum(k > DEFAULT_K for k in ks) / len(ks) if ks else 0
        run_max = [max(run) for run in runs if run]
        print(f"{label:<14}{qset:<5}{len(runs):>5}{len(ks):>7}{raised:>6.0f}%{max(ks) if ks else 0:>6}"
              f"{statistics.median(run_max) if run_max else 0:>15}")
for qset, runs in pooled.items():
    ks = [k for run in runs for k in run]
    raised = 100 * sum(k > DEFAULT_K for k in ks) / len(ks) if ks else 0
    run_max = [max(run) for run in runs if run]
    print(f"{'all backbones':<14}{qset:<5}{len(runs):>5}{len(ks):>7}{raised:>6.0f}%{max(ks) if ks else 0:>6}"
          f"{statistics.median(run_max) if run_max else 0:>15}")
