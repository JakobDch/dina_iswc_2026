"""Safe CHESS runner: runs queries one at a time, stops on API errors.

Prevents token waste by:
1. Testing API connectivity before starting
2. Running each query as a separate CHESS invocation
3. Checking logs after each query for API errors (429, connection errors)
4. Stopping immediately if an infrastructure error is detected
5. Merging all single-query predictions into one combined result

Usage:
    python scripts/run_chess_safe.py --data-path tools/chess/data/dev/dev_test_syn_remaining.json --model gpt54
    python scripts/run_chess_safe.py --data-path tools/chess/data/dev/dev_test_syn_only.json --model gpt54 --start-from 4
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHESS_ROOT = PROJECT_ROOT / "tools" / "chess"
CHESS_SRC = CHESS_ROOT / "src"

API_ERROR_PATTERNS = [
    "insufficient_quota",
    "RateLimitError",
    "APIConnectionError",
    "Connection error",
    "exceeded your current quota",
]


def test_api(model: str = "gpt-5.4") -> bool:
    """Test API connectivity with a minimal call before running expensive queries."""
    print(f"Testing API for {model}...")
    try:
        from dotenv import load_dotenv
        load_dotenv(CHESS_ROOT / ".env")
        import openai
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        kwargs = {"model": model, "messages": [{"role": "user", "content": "OK"}], "max_completion_tokens": 5}
        if model.startswith("gpt-5") or model.startswith("o"):
            kwargs["reasoning_effort"] = "low"
        r = client.chat.completions.create(**kwargs)
        print(f"  API OK: {r.choices[0].message.content.strip()}")
        return True
    except Exception as e:
        print(f"  API ERROR: {type(e).__name__}: {e}")
        return False


def has_api_error(output: str) -> str | None:
    """Check if CHESS output contains API error patterns. Returns the pattern found or None."""
    for pattern in API_ERROR_PATTERNS:
        if pattern in output:
            return pattern
    return None


def run_single_query(query: dict, config_path: Path, temp_dir: Path) -> tuple[str | None, str, bool]:
    """Run CHESS for a single query.

    Returns: (predicted_sql, captured_output, had_api_error)
    """
    # Write single-query JSON
    single_query = [dict(query, question_id=0)]
    query_file = temp_dir / f"query_{query['question_id']}.json"
    with open(query_file, "w", encoding="utf-8") as f:
        json.dump(single_query, f, indent=2, ensure_ascii=False)

    cmd = [
        sys.executable, "-u", str(CHESS_SRC / "main.py"),
        "--data_mode", "dev",
        "--data_path", str(query_file),
        "--config", str(config_path),
        "--num_workers", "1",
        "--pick_final_sql", "True",
    ]

    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    chess_env_file = CHESS_ROOT / ".env"
    if chess_env_file.exists():
        with open(chess_env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, val = line.partition("=")
                    env[key.strip()] = val.strip()

    # Use Popen for real-time output streaming (not capture_output which blocks)
    proc = subprocess.Popen(
        cmd, cwd=str(CHESS_ROOT), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, encoding="utf-8", errors="replace",
    )

    captured_lines = []
    try:
        for line in proc.stdout:
            line_stripped = line.rstrip("\n")
            captured_lines.append(line_stripped)
            # Show CHESS progress with indent
            print(f"    {line_stripped}", flush=True)
        proc.wait(timeout=300)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        raise

    output = "\n".join(captured_lines)

    # Check for API errors
    api_err = has_api_error(output)

    # Read prediction
    predicted_sql = None
    config_name = config_path.stem
    results_base = CHESS_ROOT / "results" / "dev" / config_name
    if results_base.exists():
        stem = query_file.stem
        stem_dir = results_base / stem
        if stem_dir.exists():
            run_dirs = sorted(stem_dir.glob("*"), reverse=True)
            run_dirs = [d for d in run_dirs if d.is_dir()]
            if run_dirs:
                pred_file = run_dirs[0] / "-predictions.json"
                if pred_file.exists():
                    with open(pred_file, "r", encoding="utf-8") as f:
                        preds = json.load(f)
                    val = preds.get("0", 0)
                    if val != 0 and isinstance(val, str):
                        predicted_sql = val

    return predicted_sql, output, bool(api_err)


def main():
    parser = argparse.ArgumentParser(description="Safe CHESS runner - one query at a time with API error detection")
    parser.add_argument("--data-path", type=Path, required=True, help="Path to dev JSON with queries")
    parser.add_argument("--model", default="gpt54", help="Model key (gpt54, gpt4, claude, deepseek)")
    parser.add_argument("--start-from", type=int, default=0, help="Start from this question_id (skip earlier ones)")
    parser.add_argument("--skip-api-test", action="store_true", help="Skip initial API test")
    args = parser.parse_args()

    # Model -> config mapping
    model_configs = {
        "gpt54": ("CHESS_gpt54.yaml", "gpt-5.4"),
        "gpt4": ("CHESS_gpt4o.yaml", "gpt-4o"),
        "claude": ("CHESS_claude.yaml", "claude-sonnet-4-5"),
        "deepseek": ("CHESS_deepseek.yaml", "deepseek-chat"),
    }
    if args.model not in model_configs:
        print(f"Unknown model: {args.model}. Options: {list(model_configs.keys())}")
        sys.exit(1)

    config_file, api_model = model_configs[args.model]
    config_path = CHESS_ROOT / "run" / "configs" / config_file

    # Load queries
    with open(args.data_path, "r", encoding="utf-8") as f:
        all_queries = json.load(f)
    print(f"Loaded {len(all_queries)} queries from {args.data_path}")

    # Filter by start_from
    queries = [q for q in all_queries if q["question_id"] >= args.start_from]
    if len(queries) < len(all_queries):
        print(f"Starting from question_id {args.start_from} ({len(queries)} queries remaining)")

    # Test API
    if not args.skip_api_test:
        if not test_api(api_model):
            print("\nAPI test failed! Fix your API key/quota before running.")
            sys.exit(1)
        print()

    # Run queries one at a time
    results = {}
    succeeded = 0
    failed_chess = 0  # CHESS couldn't generate SQL (valid experimental result)
    failed_api = 0

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        for i, query in enumerate(queries):
            qid = query["question_id"]
            orig_id = query.get("original_query_id", f"Q{qid}")
            dataset = query.get("original_db_id", "?")
            question = query["question"][:60]

            print(f"[{i+1}/{len(queries)}] {orig_id} ({dataset}): {question}...")

            try:
                predicted_sql, output, had_api_error = run_single_query(query, config_path, temp_path)
            except subprocess.TimeoutExpired:
                print(f"  TIMEOUT (300s) - skipping")
                results[qid] = {"status": "timeout", "sql": None, "original_id": orig_id}
                continue

            if had_api_error:
                print(f"  API ERROR detected - stopping immediately!")
                print(f"  Completed {succeeded} queries successfully before this error.")
                failed_api += 1
                results[qid] = {"status": "api_error", "sql": None, "original_id": orig_id}
                break

            if predicted_sql:
                # Extract just the SQL part (before the \t----- bird -----\t separator)
                sql_clean = predicted_sql.split("\t----- bird -----\t")[0].strip() if "\t" in predicted_sql else predicted_sql
                print(f"  OK: {sql_clean[:80]}...")
                results[qid] = {"status": "success", "sql": predicted_sql, "original_id": orig_id}
                succeeded += 1
            else:
                print(f"  CHESS failed to generate SQL (valid result)")
                results[qid] = {"status": "chess_failed", "sql": None, "original_id": orig_id}
                failed_chess += 1

    # Save combined results
    output_file = args.data_path.parent / f"{args.data_path.stem}_safe_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Also save in CHESS predictions format for compatibility
    predictions = {}
    for qid, res in results.items():
        if res["sql"]:
            predictions[str(qid)] = res["sql"]
        else:
            predictions[str(qid)] = 0
    pred_file = args.data_path.parent / f"{args.data_path.stem}_safe_predictions.json"
    with open(pred_file, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=4)

    # Summary
    print(f"\n{'=' * 60}")
    print(f"SAFE RUN COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Total queries:     {len(queries)}")
    print(f"  SQL generated:     {succeeded}")
    print(f"  CHESS failed:      {failed_chess} (valid - CHESS couldn't solve)")
    print(f"  API errors:        {failed_api} (infrastructure)")
    print(f"  Not attempted:     {len(queries) - succeeded - failed_chess - failed_api}")
    print(f"\n  Results: {output_file}")
    print(f"  Predictions: {pred_file}")


if __name__ == "__main__":
    main()