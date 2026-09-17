"""Per (LLM, condition, retrieval configuration): schema F1, data F1, schema-triple F1, path coherence,
searches per run and search-failure rate (Table 2 and the F1 values of Figures 3 and 4).

A search is an invocation of one of the three linking tools; it counts as failed when it returned no
usable match (output shorter than the empty-result threshold).

Run from the repository root:  python scripts/analysis/table_conditions.py
"""
from __future__ import annotations

import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from canonical import CONFIGS, MODELS, SETS, load_traces  # noqa: E402

LEXICAL_TOOLS = {"grep_classes", "grep_properties", "grep_data_values"}
SEMANTIC_TOOLS = {"search_classes", "search_properties", "search_data_values"}
EMPTY_OUTPUT = 150  # characters; outputs at or below this size carry no match

for folder, label in MODELS:
    agg: dict[tuple, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for r in load_traces(folder, with_raw=True):
        qset = r.get("query_set")
        if qset not in SETS:
            continue
        cfg = next((c for c, key in CONFIGS if r.get("approach") == key), None)
        if cfg is None:
            continue
        m = r.get("adaptive_metrics") or {}
        a = agg[(cfg, qset)]
        for k, f in (("schema", "schema_f1"), ("data", "best_f1"),
                     ("trF1", "retrieval_triple_f1"), ("pCoh", "retrieval_path_coherence")):
            if m.get(f) is not None:
                a[k].append(m[f])
        calls = (r.get("_raw") or {}).get("tool_invocations") or []
        searches = [x for x in calls if x.get("tool_name") in LEXICAL_TOOLS | SEMANTIC_TOOLS]
        if calls:
            a["search"].append(len(searches))
        if searches:
            a["fail"].append(100.0 * sum((x.get("output_size") or 0) <= EMPTY_OUTPUT for x in searches) / len(searches))
    print("=" * 88)
    print(label)
    print("=" * 88)
    print(f"{'Set':6} {'Cfg':9} {'N':>4} {'SchemaF1':>9} {'DataF1':>8} {'trF1':>7} {'pCoh':>7} {'Srch':>6} {'Fail%':>7}")
    for qset in SETS:
        for cfg, _ in CONFIGS:
            a = agg.get((cfg, qset))
            if not a:
                continue
            mean = lambda k: st.mean(a[k]) if a[k] else float("nan")  # noqa: E731
            print(f"{qset:6} {cfg:9} {len(a['data']):4d} {mean('schema'):9.3f} {mean('data'):8.3f} "
                  f"{mean('trF1'):7.3f} {mean('pCoh'):7.3f} {mean('search'):6.1f} {mean('fail'):6.1f}%")
    print()
