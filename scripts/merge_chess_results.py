"""Merge scattered CHESS GPT-5.4 SYN experiment results into a consolidated directory.

The experiment ran across multiple invocations:
- Batch run (queries 0-3): dev_test_syn_only/2026-04-20_20-40-19/
- Safe wrapper run 1 (queries 4-9): query_4/ through query_9/
- Safe wrapper run 2 (queries 10-16): query_10/ through query_16/

This script merges all predictions and execution histories into one directory
that can be used with run_chess_experiment.py --skip-chess-run.
"""

import json
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHESS_ROOT = PROJECT_ROOT / "tools" / "chess"
RESULTS_BASE = CHESS_ROOT / "results" / "dev" / "CHESS_gpt54"

# Output: consolidated directory
CONSOLIDATED_DIR = RESULTS_BASE / "dev_test_syn_only" / "consolidated"

# Source 1: Batch run (queries 0-3 valid, 4-16 are 0)
BATCH_DIR = RESULTS_BASE / "dev_test_syn_only" / "2026-04-20_20-40-19"

# Source 2+3: Individual query directories (one per safe wrapper query)
# Each has question_id=0 internally (single-query dev.json), must be remapped
INDIVIDUAL_QUERY_DIRS = {
    4: RESULTS_BASE / "query_4" / "2026-04-20_23-54-17",
    5: RESULTS_BASE / "query_5" / "2026-04-21_00-00-24",
    6: RESULTS_BASE / "query_6" / "2026-04-21_00-09-01",
    7: RESULTS_BASE / "query_7" / "2026-04-21_00-16-45",
    8: RESULTS_BASE / "query_8" / "2026-04-21_00-22-00",
    9: RESULTS_BASE / "query_9" / "2026-04-21_00-27-40",
    10: RESULTS_BASE / "query_10" / "2026-04-21_00-42-35",  # retry (first attempt failed)
    11: RESULTS_BASE / "query_11" / "2026-04-21_00-53-08",
    12: RESULTS_BASE / "query_12" / "2026-04-21_00-58-46",
    13: RESULTS_BASE / "query_13" / "2026-04-21_01-03-44",
    14: RESULTS_BASE / "query_14" / "2026-04-21_01-10-16",
    15: RESULTS_BASE / "query_15" / "2026-04-21_01-16-13",
    16: RESULTS_BASE / "query_16" / "2026-04-21_01-25-09",
}


def main():
    print(f"Creating consolidated result directory: {CONSOLIDATED_DIR}")
    CONSOLIDATED_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Load batch run predictions (queries 0-3)
    batch_preds_file = BATCH_DIR / "-predictions.json"
    with open(batch_preds_file, "r", encoding="utf-8") as f:
        batch_preds = json.load(f)

    merged_predictions = {}

    # Take queries 0-3 from batch run
    for qid in range(4):
        qid_str = str(qid)
        if qid_str in batch_preds and batch_preds[qid_str] != 0:
            merged_predictions[qid_str] = batch_preds[qid_str]
            print(f"  Q{qid}: from batch run")
        else:
            print(f"  Q{qid}: MISSING from batch run!")

    # Step 2: Take queries 4-16 from individual runs
    for qid, qdir in sorted(INDIVIDUAL_QUERY_DIRS.items()):
        pred_file = qdir / "-predictions.json"
        if not pred_file.exists():
            print(f"  Q{qid}: MISSING - no predictions file at {qdir}")
            continue

        with open(pred_file, "r", encoding="utf-8") as f:
            individual_preds = json.load(f)

        # Individual runs have question_id=0 (single-query dev.json)
        if "0" in individual_preds and individual_preds["0"] != 0:
            merged_predictions[str(qid)] = individual_preds["0"]
            print(f"  Q{qid}: from {qdir.parent.name}/{qdir.name}")
        else:
            print(f"  Q{qid}: EMPTY prediction in {qdir}")

    # Step 3: Write merged predictions
    preds_out = CONSOLIDATED_DIR / "-predictions.json"
    with open(preds_out, "w", encoding="utf-8") as f:
        json.dump(merged_predictions, f, indent=4)
    print(f"\nMerged predictions: {len(merged_predictions)}/17 queries")
    print(f"  Written to: {preds_out}")

    # Step 4: Copy execution history files
    # From batch run: queries 0-3 (already named {qid}_merged_heterogeneous.json)
    copied = 0
    for qid in range(4):
        src = BATCH_DIR / f"{qid}_merged_heterogeneous.json"
        dst = CONSOLIDATED_DIR / f"{qid}_merged_heterogeneous.json"
        if src.exists():
            shutil.copy2(src, dst)
            copied += 1

    # From individual runs: queries 4-16 (named 0_merged_heterogeneous.json, rename to {qid}_)
    for qid, qdir in sorted(INDIVIDUAL_QUERY_DIRS.items()):
        src = qdir / "0_merged_heterogeneous.json"
        dst = CONSOLIDATED_DIR / f"{qid}_merged_heterogeneous.json"
        if src.exists():
            shutil.copy2(src, dst)
            copied += 1
        else:
            print(f"  WARNING: No execution history for Q{qid}")

    print(f"  Copied {copied} execution history files")

    # Step 5: Copy args and statistics from batch run (for reference)
    for fname in ["-args.json", "-statistics.json"]:
        src = BATCH_DIR / fname
        if src.exists():
            shutil.copy2(src, CONSOLIDATED_DIR / fname)

    # Summary
    print(f"\n{'=' * 60}")
    print(f"CONSOLIDATED RESULTS READY")
    print(f"{'=' * 60}")
    print(f"  Directory: {CONSOLIDATED_DIR}")
    print(f"  Predictions: {len(merged_predictions)}/17")

    # Show SQL preview for each query
    print(f"\nSQL Preview:")
    for qid_str in sorted(merged_predictions.keys(), key=int):
        sql_full = merged_predictions[qid_str]
        sql = sql_full.split("\t----- bird -----\t")[0].strip() if "\t----- bird -----\t" in sql_full else sql_full
        print(f"  Q{qid_str}: {sql[:80]}...")


if __name__ == "__main__":
    main()