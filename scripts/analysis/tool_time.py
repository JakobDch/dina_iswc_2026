"""Tool execution time and run time per retrieval configuration (Section 6.2).

Over the BASE, SYN and TYPO runs of all backbones: the mean summed duration of the tool calls of a run
and the mean total run time, for the lexical and the semantic configuration.

Run from the repository root:  python scripts/analysis/tool_time.py
"""
from __future__ import annotations

import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from canonical import CONFIGS, MODELS, load_traces  # noqa: E402

SETS = ("BASE", "SYN", "TYPO")
tool: dict[str, list[float]] = {key: [] for _, key in CONFIGS}
total: dict[str, list[float]] = {key: [] for _, key in CONFIGS}
for folder, _label in MODELS:
    for r in load_traces(folder, with_raw=True):
        if r.get("query_set") not in SETS or r.get("approach") not in tool:
            continue
        raw = r["_raw"]
        calls = raw.get("tool_invocations") or []
        tool[r["approach"]].append(sum((c.get("duration_ms") or 0) for c in calls) / 1000)
        ms = (raw.get("timing") or {}).get("total_time_ms")
        if ms:
            total[r["approach"]].append(ms / 1000)

print(f"{'configuration':14s}{'runs':>6}{'tool time s/run':>17}{'run time s':>12}")
for label, key in CONFIGS:
    print(f"{label:14s}{len(tool[key]):6d}{statistics.mean(tool[key]):17.2f}{statistics.mean(total[key]):12.1f}")
both = total[CONFIGS[0][1]] + total[CONFIGS[1][1]]
print(f"{'both':14s}{len(both):6d}{'':17s}{statistics.mean(both):12.1f}")
