"""Retry rate per (LLM, condition, retrieval configuration): the share of runs in which the generation
agent handed control back to retrieval (NeedTriple), i.e. runs with more than one retrieval iteration
(Retry column of Table 2).

Run from the repository root:  python scripts/analysis/table_retry.py
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from canonical import CONFIGS, MODELS, SETS, load_traces  # noqa: E402

header = f"{'Set':7}" + "".join(f"{label:>11}" for label, _ in CONFIGS)
for folder, name in MODELS:
    counts: dict[tuple, list[int]] = defaultdict(lambda: [0, 0])
    for r in load_traces(folder, with_raw=True):
        raw = r.get("_raw") or {}
        qset = r.get("query_set")
        if qset not in SETS:
            continue
        for label, key in CONFIGS:
            if r.get("approach") == key:
                cell = counts[(qset, label)]
                cell[1] += 1
                if (raw.get("retrieval_iterations") or 0) > 1:
                    cell[0] += 1
    print(f"=== {name} ===")
    print(header)
    for qset in SETS:
        row = f"{qset:7}"
        for label, _ in CONFIGS:
            retries, total = counts[(qset, label)]
            row += f"{(100.0 * retries / total if total else 0.0):10.1f}%"
        print(row)
    print()
