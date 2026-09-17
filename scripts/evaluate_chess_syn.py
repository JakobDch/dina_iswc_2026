"""Evaluate CHESS GPT-5.4 SYN experiment from consolidated results.

Loads merged predictions, executes SQL against the heterogeneous SQLite DB,
creates traces, and runs adaptive evaluation.
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from run_chess_experiment import (
    build_query_index,
    execute_sql_on_sqlite,
    sql_results_to_sparql_bindings,
    load_chess_predictions,
    load_chess_execution_history,
    create_trace,
    save_traces,
    evaluate_chess_traces,
    CHESS_MODELS,
)

CHESS_ROOT = PROJECT_ROOT / "tools" / "chess"
DEV_JSON = CHESS_ROOT / "data" / "dev" / "dev_test_syn_only.json"
CONSOLIDATED_DIR = (
    CHESS_ROOT / "results" / "dev" / "CHESS_gpt54" / "dev_test_syn_only" / "consolidated"
)
OUTPUT_DIR = PROJECT_ROOT / "results" / "experiments"
EXPERIMENT_NAME = "chess_heterogeneous"


def main():
    model_config = CHESS_MODELS["gpt54"]
    query_index = build_query_index(DEV_JSON)

    print(f"Loading predictions from: {CONSOLIDATED_DIR}")
    predictions = load_chess_predictions(CONSOLIDATED_DIR)
    print(f"  {len(predictions)} predictions loaded")

    traces = []
    for qid, sql in sorted(predictions.items()):
        meta = query_index.get(qid)
        if meta is None:
            print(f"  WARNING: No metadata for question_id={qid}, skipping")
            continue

        db_id = meta["db_id"]
        original_id = meta["original_query_id"]

        try:
            columns, rows = execute_sql_on_sqlite(db_id, sql)
            bindings = sql_results_to_sparql_bindings(columns, rows)
            success = True
            error = None
            print(f"  {original_id} ({db_id}): {len(bindings)} rows | {sql[:70]}...")
        except Exception as e:
            bindings = []
            success = False
            error = str(e)
            print(f"  {original_id} ({db_id}): ERROR - {e}")

        chess_history = load_chess_execution_history(CONSOLIDATED_DIR, qid, db_id)

        trace = create_trace(
            question_meta=meta,
            sql=sql,
            bindings=bindings,
            model_key="gpt54",
            llm_model_name=model_config["llm_model_name"],
            success=success,
            error_message=error,
            chess_history=chess_history,
        )
        traces.append(trace)

    if not traces:
        print("No traces generated!")
        return

    # Save traces
    exp_dir = save_traces(traces, OUTPUT_DIR, EXPERIMENT_NAME)

    # Summary
    print(f"\n{'=' * 70}")
    print(f"Experiment: {EXPERIMENT_NAME}")
    print(f"  Total traces: {len(traces)}")
    successful = sum(1 for t in traces if t["success"])
    failed = sum(1 for t in traces if not t["success"])
    print(f"  Successful SQL execution: {successful}/{len(traces)}")
    if failed:
        print(f"  Failed SQL execution: {failed}")
        for t in traces:
            if not t["success"]:
                print(f"    {t['query_metadata']['query_id']}: {t['error_message']}")
    print(f"  Output: {exp_dir}")
    print(f"{'=' * 70}")

    # Evaluate
    print(f"\n{'=' * 70}")
    print("EVALUATING TRACES...")
    print(f"{'=' * 70}")
    evaluate_chess_traces(exp_dir)


if __name__ == "__main__":
    main()