"""Outcome of runs in which the generation agent handed control back to retrieval (NeedTriple).

A run counts as a hand-back when it has more than one retrieval iteration (same definition as the
Retry column, table_retry.py). Reports the mean data F1 of hand-back runs against the rest and the
share of hand-back runs that end with data F1 >= 0.9, over BASE, SYN and TYPO (Section 6.1).

Run from the repository root:  python scripts/analysis/handback_outcome.py
"""
from __future__ import annotations

import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from canonical import MODELS, load_traces  # noqa: E402

SETS = ("BASE", "SYN", "TYPO")
rows = []
for folder, _label in MODELS:
    for r in load_traces(folder, with_raw=True):
        if r.get("query_set") not in SETS:
            continue
        f1 = (r.get("adaptive_metrics") or {}).get("best_f1")
        if f1 is None:
            continue
        rows.append((folder, r.get("approach"), ((r.get("_raw") or {}).get("retrieval_iterations") or 0) > 1, float(f1)))

print(f"hand-back outcome, BASE/SYN/TYPO, n={len(rows)}")
for label, sel in (("all configurations", lambda a: True),
                   ("lexical only", lambda a: a == "agentic_grep"),
                   ("semantic only", lambda a: a == "agentic_semantic")):
    hb = [f for _, a, h, f in rows if h and sel(a)]
    no = [f for _, a, h, f in rows if not h and sel(a)]
    if not hb:
        print(f"  {label}: no hand-back runs")
        continue
    print(f"  {label}: hand-back n={len(hb)} mean F1={st.mean(hb):.3f} (>=0.9: {sum(f >= 0.9 for f in hb)}, "
          f"{100 * sum(f >= 0.9 for f in hb) / len(hb):.1f}%) | no hand-back n={len(no)} mean F1={st.mean(no):.3f}")
for folder, label in MODELS:
    hb = [f for m, _, h, f in rows if h and m == folder]
    no = [f for m, _, h, f in rows if not h and m == folder]
    if hb:
        print(f"  {label}: hand-back n={len(hb)} mean F1={st.mean(hb):.3f} | rest n={len(no)} mean F1={st.mean(no):.3f}")
