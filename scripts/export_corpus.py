"""Export the question corpus and its adaptive ground truth to JSON, CSV and Markdown.

The corpus is defined in data/queries/experimental_corpus.py. This script writes
  data/queries/corpus.json      every formulation with all admissible SPARQL readings and column labels
  data/queries/corpus.csv       one row per formulation (id, condition, dataset, question, substitution)
  data/queries/corpus_table.md  the same table as Markdown, pasted into data/queries/README.md

Run from the repository root:  python scripts/export_corpus.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.queries.experimental_corpus import (  # noqa: E402
    ALL_EXPERIMENTAL_QUERIES,
    SET_BASE,
    SET_SYN,
    SET_TYPO,
    SET_UNDER,
    get_baseline_for_variant,
)

OUT = ROOT / "data" / "queries"
PAPER_SETS = {"BASE": SET_BASE, "SYN": SET_SYN, "TYPO": SET_TYPO, "UNDER": SET_UNDER}
LEVEL = {"PREFERRED": "REQUIRED", "ACCEPTABLE": "ACCEPTABLE"}


def relevance(col) -> str:
    name = getattr(col.level, "name", str(col.level))
    return LEVEL.get(name, name)


def record(q, condition: str) -> dict:
    base = get_baseline_for_variant(q.query_id) if condition in ("SYN", "TYPO") else None
    return {
        "id": q.query_id,
        "condition": condition,
        "dataset": q.dataset,
        "question": q.query,
        "base_id": base.query_id if base else None,
        "substitution": getattr(q, "variant_description", "") or (q.notes if condition in ("SYN", "TYPO") else ""),
        "description": q.description,
        "readings": [{"sparql": s.strip(), "columns": columns(q, i)} for i, s in enumerate(q.sparql_queries)],
        "notes": q.notes,
    }


def columns(q, i: int) -> list[dict]:
    """Column labels of reading i (the corpus stores one list for all readings or one list per reading)."""
    cols = q.columns[i] if q.columns and isinstance(q.columns[0], list) else q.columns
    return [
        {"variable": c.var_name, "relevance": relevance(c), "semantic_concept": c.semantic_concept,
         "description": c.description, "is_measurement": c.is_measurement}
        for c in cols
    ]


rows = []
for condition, qs in PAPER_SETS.items():
    for q in qs:
        rows.append(record(q, condition))

(OUT / "corpus.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")

with open(OUT / "corpus.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["id", "condition", "dataset", "base_id", "n_readings", "question", "substitution"])
    for r in rows:
        w.writerow([r["id"], r["condition"], r["dataset"], r["base_id"] or "", len(r["readings"]),
                    r["question"], r["substitution"]])

lines = ["| ID | Condition | Dataset | Readings | Question |", "|---|---|---|---|---|"]
for r in rows:
    q = r["question"].replace("|", "\\|")
    lines.append(f"| {r['id']} | {r['condition']} | {r['dataset']} | {len(r['readings'])} | {q} |")
(OUT / "corpus_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

n_all = len(ALL_EXPERIMENTAL_QUERIES)
print(f"{len(rows)} formulations exported ({', '.join(f'{k}={len(v)}' for k, v in PAPER_SETS.items())}); "
      f"module defines {n_all} entries in total")
