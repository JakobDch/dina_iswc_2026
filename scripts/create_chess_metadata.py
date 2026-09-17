"""Create CHESS metadata files (dev.json) from the project's experimental corpus.

Reads AdaptiveGroundTruth queries from experimental_corpus.py (BASE, SYN, TYPO,
LARGE, UNDER, CROSS sets) and produces a BIRD-format dev.json for CHESS Text2SQL.

Cross-dataset queries (CROSS set) are excluded since CHESS operates on
a single database at a time. LARGE dataset queries are excluded since
they use different dataset endpoints (_LARGE suffix).

Usage:
    python scripts/create_chess_metadata.py
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.queries.experimental_corpus import ALL_EXPERIMENTAL_QUERIES

CHESS_DATA_DIR = PROJECT_ROOT / "tools" / "chess" / "data" / "dev"

# Map project dataset names to CHESS db_ids (lowercase)
DATASET_TO_DB_ID = {
    "EDU": "edu",
    "TRN": "trn",
    "NRG": "nrg",
    "BSBM": "bsbm",
    "LCA": "lca",
}


def create_dev_json(output_path: Path, merged: bool = False) -> int:
    """Create dev.json in BIRD format from the experimental corpus.

    Args:
        output_path: Path to write dev.json.
        merged: If True, set all db_id to "merged" and include CROSS queries.

    Returns the number of questions written.
    """
    questions = []
    skipped_cross = 0
    skipped_large = 0
    skipped_unknown = 0

    for idx, q in enumerate(ALL_EXPERIMENTAL_QUERIES):
        # Skip LARGE dataset variants -- different endpoints/data
        dataset_upper = q.dataset.upper()
        if dataset_upper.endswith("_LARGE"):
            skipped_large += 1
            continue

        if merged:
            # Merged mode: all queries use "merged" db_id, including CROSS
            db_id = "merged"
        else:
            # Individual mode: skip cross-dataset queries
            if len(q.datasets) > 1:
                skipped_cross += 1
                continue
            db_id = DATASET_TO_DB_ID.get(dataset_upper)
            if db_id is None:
                print(f"  WARNING: Unknown dataset '{q.dataset}' for query {q.query_id}, skipping")
                skipped_unknown += 1
                continue

        # Map query_set to difficulty
        if q.query_set in ("BASE", "SYN"):
            difficulty = "simple"
        elif q.query_set in ("TYPO", "UNDER"):
            difficulty = "moderate"
        else:
            difficulty = "challenging"

        entry = {
            "question_id": idx,
            "db_id": db_id,
            "question": q.query.strip(),
            "evidence": "",
            "SQL": "",
            "difficulty": difficulty,
            # Extra metadata for traceability back to our evaluation
            "original_query_id": q.query_id,
            "query_set": q.query_set,
        }

        if merged:
            # Preserve original dataset info for traceability
            entry["original_datasets"] = q.datasets
            entry["original_db_id"] = DATASET_TO_DB_ID.get(dataset_upper, dataset_upper.lower())

        questions.append(entry)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(questions, f, indent=2, ensure_ascii=False)

    print(f"  Skipped: {skipped_cross} cross-dataset, {skipped_large} large-dataset, {skipped_unknown} unknown")
    return len(questions)


def main():
    parser = argparse.ArgumentParser(description="Create CHESS metadata (dev.json)")
    parser.add_argument(
        "--merged",
        action="store_true",
        help="Set all db_id to 'merged' and include CROSS queries",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path (default: dev.json or dev_merged.json)",
    )
    args = parser.parse_args()

    if args.output:
        output_path = args.output
    elif args.merged:
        output_path = CHESS_DATA_DIR / "dev_merged.json"
    else:
        output_path = CHESS_DATA_DIR / "dev.json"

    mode_label = "MERGED" if args.merged else "INDIVIDUAL"
    print("=" * 70)
    print(f"Create CHESS Metadata ({mode_label} mode)")
    print("=" * 70)

    count = create_dev_json(output_path, merged=args.merged)

    print(f"\nCreated {output_path}")
    print(f"  {count} questions total")

    # Show breakdown by db_id and query_set
    with open(output_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    by_db = {}
    by_set = {}
    for q in data:
        by_db.setdefault(q["db_id"], []).append(q)
        by_set.setdefault(q["query_set"], []).append(q)

    print("\n  By database:")
    for db_id, qs in sorted(by_db.items()):
        print(f"    {db_id}: {len(qs)} questions")

    print("\n  By query set:")
    for qset, qs in sorted(by_set.items()):
        print(f"    {qset}: {len(qs)} questions")

    print(f"\nOutput: {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
