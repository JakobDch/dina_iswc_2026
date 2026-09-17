"""Merge run-1 (q0-q3) + run-2 (q4-q16) and evaluate strict + tolerant F1 over all 17.

Reads predictions from both run dirs, executes each SQL against merged.sqlite,
builds traces, runs strict adaptive evaluation, then re-projects with tolerant
matching for the final apples-to-apples Data F1.
"""

import json
import sys
import unicodedata
import urllib.parse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from run_chess_experiment import (
    build_query_index,
    execute_sql_on_sqlite,
    sql_results_to_sparql_bindings,
    create_trace,
    save_traces,
    evaluate_chess_traces,
    CHESS_MODELS,
)

CHESS_ROOT = PROJECT_ROOT / "tools" / "chess"
DEV_JSON = CHESS_ROOT / "data" / "dev" / "dev_test_syn_only_homogeneous.json"
RUN_1 = (
    CHESS_ROOT / "results" / "dev" / "CHESS_gpt54" / "dev_test_syn_only_homogeneous"
    / "2026-04-28_17-53-09"
)
RUN_2 = (
    CHESS_ROOT / "results" / "dev" / "CHESS_gpt54" / "dev_test_syn_only_homogeneous_remaining"
    / "2026-04-28_19-00-13"
)
OUTPUT_DIR = PROJECT_ROOT / "results" / "experiments"
EXPERIMENT_NAME = "chess_consolidated"
GT_CACHE_FILE = PROJECT_ROOT / "data" / "cache" / "ground_truth_full.json"


def load_predictions(run_dir):
    out = {}
    raw = json.load((run_dir / "-predictions.json").open(encoding="utf-8"))
    for k, v in raw.items():
        if not isinstance(v, str) or not v.strip() or v.startswith("--"):
            continue
        sql = v.split("\t----- bird -----\t")[0].strip() if "\t----- bird -----\t" in v else v.strip()
        out[int(k)] = sql
    return out


def load_history(run_dir, qid, db_id):
    p = run_dir / f"{qid}_{db_id}.json"
    if not p.exists():
        return []
    return json.load(p.open(encoding="utf-8"))


def build_traces():
    model_config = CHESS_MODELS["gpt54"]
    query_index = build_query_index(DEV_JSON)
    p1 = load_predictions(RUN_1)
    p2 = load_predictions(RUN_2)
    # Run-2 takes precedence for any q that appears in both (e.g., re-runs of q13/q16)
    merged = {**p1, **p2}
    print(f"Run-1 filled: {sorted(p1.keys())}")
    print(f"Run-2 filled: {sorted(p2.keys())}")
    print(f"Merged set:   {len(merged)}/17 — qids {sorted(merged.keys())}")
    print(f"Missing:      {sorted(set(range(17)) - set(merged.keys()))}")
    print()

    traces = []
    for qid, sql in sorted(merged.items()):
        meta = query_index.get(qid)
        if meta is None:
            print(f"  WARNING: No metadata for q{qid}, skipping")
            continue
        db_id = meta["db_id"]
        original_id = meta["original_query_id"]

        # Pick the run dir that holds the trace for this qid
        run_dir = RUN_2 if qid in p2 else RUN_1

        try:
            cols, rows = execute_sql_on_sqlite(db_id, sql)
            bindings = sql_results_to_sparql_bindings(cols, rows)
            success, error = True, None
            print(f"  {original_id} (q{qid}): {len(bindings)} rows")
        except Exception as e:
            bindings = []
            success, error = False, str(e)
            print(f"  {original_id} (q{qid}): ERROR — {e}")

        history = load_history(run_dir, qid, db_id)
        trace = create_trace(
            question_meta=meta,
            sql=sql,
            bindings=bindings,
            model_key="gpt54",
            llm_model_name=model_config["llm_model_name"],
            success=success,
            error_message=error,
            chess_history=history,
        )
        traces.append(trace)
    return traces


# ---------- tolerant tuple normalization -----------
def _strip_uri(s):
    if not s.startswith(("http://", "https://")):
        return s
    return urllib.parse.unquote(s.rsplit("/", 1)[-1])


def normalize_tolerant(value):
    if value is None:
        return ""
    if isinstance(value, dict):
        raw = str(value.get("value", ""))
    else:
        raw = str(value)
    raw = unicodedata.normalize("NFC", raw.strip())
    if raw == "":
        return ""
    try:
        f = float(raw)
        return f"{abs(f):.6g}"
    except ValueError:
        pass
    return _strip_uri(raw).casefold()


def tolerant_set(tuples, indices=None):
    out = set()
    for t in tuples:
        if indices is not None:
            t = tuple(t[i] for i in indices if i < len(t))
        out.add(tuple(normalize_tolerant(c) for c in t))
    return out


def evaluate_set(gt, llm):
    if not gt and not llm:
        return 0.0, 0.0, 0.0
    tp = len(gt & llm)
    fp = len(llm - gt)
    fn = len(gt - llm)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return r, p, f1


def tolerant_eval_pass(exp_dir):
    eval_file = exp_dir / "adaptive_evaluation.json"
    eval_data = json.load(eval_file.open(encoding="utf-8"))
    gt_cache = json.load(GT_CACHE_FILE.open(encoding="utf-8"))["queries"]

    print()
    print("=" * 80)
    print(f"TOLERANT TUPLE-MATCH RE-EVAL")
    print("=" * 80)
    print(f"{'qid':>6} | {'strict F1':>10} {'toler F1':>10} | {'strict R':>9} {'toler R':>9} | "
          f"{'strict P':>9} {'toler P':>9}")
    print("-" * 80)

    rows_strict = []
    rows_toler = []

    # All 17 queries should appear; for those without traces (q7, q15) we record 0.
    by_qid = {}
    for tk, t in eval_data["traces"].items():
        qid = t.get("query_id")
        by_qid[qid] = (tk, t)

    all_query_ids = sorted(set(by_qid.keys()) | {"SYN10", "SYN20"})  # ensure missing qids accounted

    for qid in all_query_ids:
        if qid not in by_qid:
            # CHESS produced no SQL → F1=0 for all metrics
            rows_strict.append((0.0, 0.0, 0.0))
            rows_toler.append((0.0, 0.0, 0.0))
            print(f"{qid:>6} | {'0.000':>10} {'0.000':>10} | {'0.000':>9} {'0.000':>9} | "
                  f"{'0.000':>9} {'0.000':>9}  [no SQL]")
            continue

        tk, t = by_qid[qid]
        m = t.get("adaptive_metrics") or {}
        strict_f1 = m.get("best_f1") or 0.0
        strict_r = m.get("best_recall") or 0.0
        strict_p = m.get("best_precision") or 0.0

        # Read full LLM rows from saved trace
        trace_path = exp_dir / "traces" / f"{tk}.json"
        trace_full = json.load(trace_path.open(encoding="utf-8"))
        llm_rows = trace_full.get("final_results", [])

        cached = gt_cache.get(qid, {})
        gt_raw = cached.get("gt_tuples", [])

        col_map = m.get("column_mapping") or []
        if not col_map:
            toler_r, toler_p, toler_f1 = 0.0, 0.0, 0.0
        else:
            gt_indices = [c["gt_col_idx"] for c in col_map]
            llm_cols = [c["llm_col_name"] for c in col_map]
            gt_proj = tolerant_set(gt_raw, indices=gt_indices)
            llm_proj_raw = [
                tuple(
                    row.get(c, {}).get("value", "") if isinstance(row.get(c), dict) else row.get(c, "")
                    for c in llm_cols
                )
                for row in llm_rows
            ]
            llm_proj = tolerant_set(llm_proj_raw)
            toler_r, toler_p, toler_f1 = evaluate_set(gt_proj, llm_proj)

        rows_strict.append((strict_f1, strict_r, strict_p))
        rows_toler.append((toler_f1, toler_r, toler_p))

        marker = ""
        delta = toler_f1 - strict_f1
        if delta > 0.05:
            marker = " ↑"
        elif delta < -0.05:
            marker = " ↓"
        print(f"{qid:>6} | {strict_f1:>10.3f} {toler_f1:>10.3f} | {strict_r:>9.3f} {toler_r:>9.3f} | "
              f"{strict_p:>9.3f} {toler_p:>9.3f}{marker}")

    avg = lambda xs, i: sum(x[i] for x in xs) / len(xs) if xs else 0.0
    print("-" * 80)
    print(f"{'mean':>6} | {avg(rows_strict,0):>10.3f} {avg(rows_toler,0):>10.3f} | "
          f"{avg(rows_strict,1):>9.3f} {avg(rows_toler,1):>9.3f} | "
          f"{avg(rows_strict,2):>9.3f} {avg(rows_toler,2):>9.3f}")

    # Schema F1 mean (unchanged by tolerant projection)
    schema_f1s = []
    for tk, t in eval_data["traces"].items():
        m = t.get("adaptive_metrics") or {}
        s = m.get("schema_f1")
        if s is not None:
            schema_f1s.append(s)
    # Add 0 for missing queries
    schema_f1s += [0.0, 0.0]  # q7 and q15
    print(f"\nSchema F1 mean (all 17): {sum(schema_f1s)/len(schema_f1s):.3f}")


def main():
    traces = build_traces()
    if not traces:
        print("No traces!")
        return
    exp_dir = save_traces(traces, OUTPUT_DIR, EXPERIMENT_NAME)
    print(f"\nSaved {len(traces)} traces to {exp_dir}")
    print("\nRunning strict evaluation...\n")
    evaluate_chess_traces(exp_dir)
    tolerant_eval_pass(exp_dir)


if __name__ == "__main__":
    main()