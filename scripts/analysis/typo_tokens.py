"""How often a misspelled token of a TYPO question reaches the linking tools (Section 6.1).

The corpus records every misspelling of a TYPO question as "<correct>-><misspelled>" in its notes. Over the
TYPO runs of all backbones, the script counts the calls of the six linking tools (lexical and semantic
class, property and value search) whose input contains one of the question's misspelled tokens as a whole
word, and how many of those are value searches, where the linker resolves the token against the data.

Run from the repository root:  python scripts/analysis/typo_tokens.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from canonical import MODELS, load_traces  # noqa: E402
from data.queries.experimental_corpus import SET_TYPO  # noqa: E402

LINKING = {"grep_classes", "grep_properties", "grep_data_values",
           "search_classes", "search_properties", "search_data_values"}
VALUE_SEARCH = {"grep_data_values", "search_data_values"}

misspelled: dict[str, set[str]] = {}
for q in SET_TYPO:
    pairs = re.findall(r"([A-Za-z][A-Za-z '\-]*?)\s*->\s*([A-Za-z][A-Za-z '\-]*)", q.notes or "")
    misspelled[q.query_id] = {p[1].strip().lower() for p in pairs}

runs = calls = hits = value_hits = 0
examples: set[str] = set()
for folder, _label in MODELS:
    for r in load_traces(folder, with_raw=True):
        if r.get("query_set") != "TYPO":
            continue
        linking = [t for t in (r["_raw"].get("tool_invocations") or []) if t.get("tool_name") in LINKING]
        if not linking:
            continue
        runs += 1
        for t in linking:
            calls += 1
            text = (t.get("input_summary") or "").lower()
            found = [m for m in misspelled[r["query_id"]] if re.search(r"\b" + re.escape(m) + r"\b", text)]
            if found:
                hits += 1
                if t.get("tool_name") in VALUE_SEARCH:
                    value_hits += 1
                    examples.update(found)

print(f"TYPO runs with linking calls: {runs}")
print(f"linking-tool calls: {calls}")
print(f"calls containing a misspelled token: {hits} ({100 * hits / calls:.1f}%)")
print(f"  of which value searches: {value_hits} (e.g. {', '.join(sorted(examples)[:6])})")
print(f"  of which class or property searches: {hits - value_hits}")
