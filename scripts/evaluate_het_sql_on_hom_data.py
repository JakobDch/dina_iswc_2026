"""Run HET CHESS predictions against the homogenized (HOM) DB and re-evaluate.

This isolates the data-availability effect: the SQL that CHESS produced is
unchanged, but each query is now executed against a SQLite snapshot in which
the data has been collapsed into the original-named tables (no schema split).

Queries whose SQL references tables that only exist in the heterogeneous
snapshot (alt_*-style tables) are reported as N/A — the homogenized DB has no
analogue. Aggregates are reported separately for the runnable subset.
"""

import json
import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from run_chess_experiment import (
    build_query_index,
    sql_results_to_sparql_bindings,
    create_trace,
    save_traces,
    evaluate_chess_traces,
    CHESS_MODELS,
)

CHESS_ROOT = PROJECT_ROOT / "tools" / "chess"
DEV_HET_JSON = CHESS_ROOT / "data" / "dev" / "dev_test_syn_only.json"
HET_PREDICTIONS = CHESS_ROOT / "results" / "dev" / "CHESS_gpt54" / "dev_test_syn_only" / "consolidated" / "-predictions.json"
HOM_DB = CHESS_ROOT / "data" / "dev" / "dev_databases" / "merged" / "merged.sqlite"
HET_DB = CHESS_ROOT / "data" / "dev" / "dev_databases" / "merged_heterogeneous" / "merged_heterogeneous.sqlite"
OUTPUT_DIR = PROJECT_ROOT / "results" / "experiments"
EXPERIMENT_NAME = "chess_heterogeneous_sql_on_consolidated"


def clean_sql(s):
    if not isinstance(s, str):
        return ""
    if "\t----- bird -----\t" in s:
        s = s.split("\t----- bird -----\t")[0]
    return s.strip()


def execute(db_path, sql):
    con = sqlite3.connect(str(db_path))
    try:
        cur = con.execute(sql)
        cols = [d[0] for d in cur.description] if cur.description else []
        rows = cur.fetchall()
        return cols, rows, None
    except Exception as e:
        return [], [], str(e)
    finally:
        con.close()


def main():
    model_config = CHESS_MODELS["gpt54"]
    # Build index against HOM dev JSON so db_id maps to "merged"
    # (we'll override db_id to "merged" so eval pipeline knows where to find data)
    dev_het = json.load(DEV_HET_JSON.open(encoding="utf-8"))
    # Re-target each question to the HOM db
    for q in dev_het:
        q["db_id"] = "merged"
    DEV_REWRITTEN = CHESS_ROOT / "data" / "dev" / "dev_test_syn_only_HET_to_HOM_temp.json"
    DEV_REWRITTEN.write_text(json.dumps(dev_het, indent=2), encoding="utf-8")

    query_index = build_query_index(DEV_REWRITTEN)
    raw_pred = json.load(HET_PREDICTIONS.open(encoding="utf-8"))

    # Filter usable predictions
    predictions = {}
    for k, v in raw_pred.items():
        if not isinstance(v, str) or not v.strip() or v.startswith("--"):
            continue
        sql = clean_sql(v)
        if sql:
            predictions[int(k)] = sql

    print(f"Loaded {len(predictions)} HET CHESS predictions")
    print(f"Executing them against HOM DB (full data)...")
    print()

    traces = []
    skipped = []
    for qid, sql in sorted(predictions.items()):
        meta = query_index.get(qid)
        if meta is None:
            print(f"  q{qid}: no metadata, skip")
            continue
        original_id = meta["original_query_id"]

        # First execute against HET DB to get baseline (we already know these)
        het_cols, het_rows, het_err = execute(HET_DB, sql)
        # Then execute against HOM DB
        hom_cols, hom_rows, hom_err = execute(HOM_DB, sql)

        if hom_err:
            print(f"  {original_id} (q{qid}): N/A - SQL fails on HOM DB ({hom_err[:80]})")
            skipped.append((original_id, hom_err))
            # Still save a trace with empty bindings, success=False — eval will give it 0
            bindings = []
            success = False
            error = f"HET-SQL-on-HOM-DB: {hom_err}"
        else:
            bindings = sql_results_to_sparql_bindings(hom_cols, hom_rows)
            success = True
            error = None
            print(f"  {original_id} (q{qid}): HET={len(het_rows)} rows -> HOM={len(hom_rows)} rows")

        trace = create_trace(
            question_meta=meta,
            sql=sql,
            bindings=bindings,
            model_key="gpt54",
            llm_model_name=model_config["llm_model_name"],
            success=success,
            error_message=error,
            chess_history=[],
        )
        traces.append(trace)

    print()
    if skipped:
        print(f"Skipped {len(skipped)} queries (HET SQL references alt-tables not in HOM DB):")
        for q, e in skipped:
            print(f"  {q}: {e[:100]}")
        print()

    if not traces:
        print("No traces!")
        return

    exp_dir = save_traces(traces, OUTPUT_DIR, EXPERIMENT_NAME)
    print(f"Saved {len(traces)} traces to {exp_dir}")
    print()
    print("Running adaptive evaluation...")
    print("=" * 70)
    evaluate_chess_traces(exp_dir)


if __name__ == "__main__":
    main()