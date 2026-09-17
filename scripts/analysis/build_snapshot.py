"""Assemble results/paper/<model>/adaptive_evaluation.json from the four condition runs of each model.

Run after re-evaluating the runs (scripts/reevaluate_tiered.py <run> --adaptive):
  python scripts/analysis/build_snapshot.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from canonical import EXPERIMENTS, MODELS, PAPER  # noqa: E402

CONDITIONS = ["base", "syn", "typo", "under"]

for folder, _label in MODELS:
    records: dict[str, dict] = {}
    counts: dict[str, int] = {}
    for cond in CONDITIONS:
        run = f"{folder}_{cond}"
        with open(EXPERIMENTS / run / "adaptive_evaluation.json", encoding="utf-8") as f:
            ev = json.load(f)["traces"]
        for tid, r in ev.items():
            r = dict(r)
            r["_source_experiment"] = run
            r["_trace_id"] = tid
            records[f"{run}/{tid}"] = r
            counts[r.get("query_set", "?")] = counts.get(r.get("query_set", "?"), 0) + 1
    out = PAPER / folder / "adaptive_evaluation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"experiment_name": f"paper/{folder}", "evaluation_mode": "adaptive",
                   "sources": {c.upper(): [f"{folder}_{c}"] for c in CONDITIONS},
                   "total_traces": len(records), "traces": records}, f, indent=1)
    print(folder, counts, "->", out)
