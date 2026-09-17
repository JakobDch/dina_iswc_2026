"""Run CHESS Text2SQL experiment and produce traces compatible with our evaluation.

This script:
1. Runs CHESS preprocessing (LSH + vector DB) if not already done
2. Runs CHESS main pipeline for each LLM model
3. Collects generated SQL from CHESS output
4. Executes SQL against SQLite databases
5. Converts results to SPARQL-binding format
6. Saves trace files compatible with reevaluate_tiered.py

Usage:
    python scripts/run_chess_experiment.py [--models gpt4 claude deepseek] [--skip-preprocess]
    python scripts/run_chess_experiment.py --models gpt4 --experiment-name chess_gpt4_test
"""

import argparse
import importlib.util
import json
import os
import sqlite3
import subprocess
import sys
import uuid
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.queries.experimental_corpus import ALL_EXPERIMENTAL_QUERIES

CHESS_ROOT = PROJECT_ROOT / "tools" / "chess"
CHESS_SRC = CHESS_ROOT / "src"
CHESS_DB_ROOT = CHESS_ROOT / "data" / "dev" / "dev_databases"
CHESS_DEV_JSON = CHESS_ROOT / "data" / "dev" / "dev.json"
CHESS_DEV_JSON_MERGED = CHESS_ROOT / "data" / "dev" / "dev_merged.json"
CHESS_DEV_JSON_HETEROGENEOUS = CHESS_ROOT / "data" / "dev" / "dev_merged_heterogeneous.json"

# Model mapping: our key -> CHESS config file + engine name
CHESS_MODELS = {
    "gpt4": {
        "config": "CHESS_gpt4o.yaml",
        "llm_model_name": "gpt-4o",
    },
    "gpt54": {
        "config": "CHESS_gpt54.yaml",
        "llm_model_name": "gpt-5.4",
    },
    "claude": {
        "config": "CHESS_claude.yaml",
        "llm_model_name": "claude-sonnet-4-5-20250929",
    },
    "deepseek": {
        "config": "CHESS_deepseek.yaml",
        "llm_model_name": "deepseek-chat",
    },
}

# Dataset mapping for our queries
DATASET_TO_DB_ID = {
    "EDU": "edu",
    "TRN": "trn",
    "NRG": "nrg",
    "BSBM": "bsbm",
    "LCA": "lca",
}

# GT cache for evaluation (extracted by extract_ground_truth_cache.py)
GT_CACHE_FILE = PROJECT_ROOT / "data" / "cache" / "ground_truth_full.json"


def _load_tiered_metrics():
    """Load evaluation modules directly, bypassing src.evaluation.__init__.py.

    This avoids the import chain: __init__.py -> runner.py -> OrchestratorAgent
    -> langchain_ollama (incompatible with CHESS's langchain pinning).

    The key issue: calculate_adaptive_metrics() has a lazy import
    `from src.evaluation.column_signatures import extract_column_signatures`
    which triggers __init__.py. We pre-populate sys.modules to prevent this.
    """
    import types

    eval_pkg = PROJECT_ROOT / "src" / "evaluation"

    # 1. Create a fake src.evaluation package in sys.modules (prevents __init__.py)
    if "src.evaluation" not in sys.modules:
        fake_pkg = types.ModuleType("src.evaluation")
        fake_pkg.__path__ = [str(eval_pkg)]
        fake_pkg.__package__ = "src.evaluation"
        sys.modules["src.evaluation"] = fake_pkg

    # 2. Load column_signatures directly and register it
    cs_path = eval_pkg / "column_signatures.py"
    cs_spec = importlib.util.spec_from_file_location(
        "src.evaluation.column_signatures", str(cs_path)
    )
    cs_mod = importlib.util.module_from_spec(cs_spec)
    sys.modules["src.evaluation.column_signatures"] = cs_mod
    cs_spec.loader.exec_module(cs_mod)

    # 3. Load tiered_metrics_tuples directly
    tm_path = eval_pkg / "tiered_metrics_tuples.py"
    tm_spec = importlib.util.spec_from_file_location(
        "src.evaluation.tiered_metrics_tuples", str(tm_path)
    )
    tm_mod = importlib.util.module_from_spec(tm_spec)
    sys.modules["src.evaluation.tiered_metrics_tuples"] = tm_mod
    tm_spec.loader.exec_module(tm_mod)

    return tm_mod


def run_chess_preprocess(db_id: str = "all"):
    """Run CHESS preprocessing (LSH + vector DB) for databases.

    Args:
        db_id: Database ID to preprocess. Use "all" for all individual DBs,
               or a specific db_id like "merged".
    """
    print("=" * 70)
    print(f"Running CHESS preprocessing (db_id={db_id})...")
    print("=" * 70)

    db_root = str(CHESS_DB_ROOT)

    cmd = [
        sys.executable, "-u", str(CHESS_SRC / "preprocess.py"),
        "--db_root_directory", db_root,
        "--signature_size", "100",
        "--n_gram", "3",
        "--threshold", "0.01",
        "--db_id", db_id,
        "--verbose", "True",
    ]

    result = subprocess.run(cmd, cwd=str(CHESS_ROOT), capture_output=False)
    if result.returncode != 0:
        print(f"WARNING: Preprocessing returned code {result.returncode}")
    else:
        print("Preprocessing complete.")


def is_preprocessed(db_id: str) -> bool:
    """Check if a database has been preprocessed."""
    db_dir = CHESS_DB_ROOT / db_id
    preprocessed_dir = db_dir / "preprocessed"
    if not preprocessed_dir.exists():
        return False
    # Check for at least the LSH file
    lsh_files = list(preprocessed_dir.glob("*_lsh.pkl"))
    return len(lsh_files) > 0


def run_chess_pipeline(model_key: str, dev_json_path: Path = None) -> Path:
    """Run CHESS main pipeline for a specific model.

    Args:
        model_key: Key into CHESS_MODELS (e.g., "gpt4", "deepseek").
        dev_json_path: Path to dev.json file. Defaults to CHESS_DEV_JSON.

    Returns the result directory path.
    """
    if dev_json_path is None:
        dev_json_path = CHESS_DEV_JSON

    model_config = CHESS_MODELS[model_key]
    config_path = CHESS_ROOT / "run" / "configs" / model_config["config"]

    print(f"\n{'=' * 70}")
    print(f"Running CHESS pipeline with {model_key} ({model_config['llm_model_name']})")
    print(f"Config: {config_path}")
    print(f"Data: {dev_json_path}")
    print(f"{'=' * 70}")

    cmd = [
        sys.executable, "-u", str(CHESS_SRC / "main.py"),
        "--data_mode", "dev",
        "--data_path", str(Path(dev_json_path).resolve()),
        "--config", str(config_path),
        "--num_workers", "1",
        "--pick_final_sql", "True",
    ]

    env = os.environ.copy()
    env["DB_ROOT_DIRECTORY"] = str(CHESS_DB_ROOT)
    env["PYTHONUTF8"] = "1"  # Fix Windows encoding for CHESS templates

    # Load CHESS .env and forward API keys to subprocess
    chess_env_file = CHESS_ROOT / ".env"
    if chess_env_file.exists():
        with open(chess_env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, val = line.partition("=")
                    env[key.strip()] = val.strip()

    result = subprocess.run(cmd, cwd=str(CHESS_ROOT), env=env, capture_output=False)

    if result.returncode != 0:
        print(f"ERROR: CHESS pipeline failed with code {result.returncode}")

    # Find the most recent result directory
    results_dir = CHESS_ROOT / "results" / "dev"
    setting_name = model_config["config"].replace(".yaml", "")
    setting_dir = results_dir / setting_name

    if not setting_dir.exists():
        print(f"WARNING: Result directory not found: {setting_dir}")
        return None

    # Get most recent run (sorted by timestamp directory name)
    # Structure: CHESS_deepseek/{data_stem}/{timestamp}/
    # Filter by the data_stem to avoid picking up results from other dev.json files
    data_stem = dev_json_path.stem if dev_json_path else "dev"
    data_stem_dir = setting_dir / data_stem
    if data_stem_dir.exists():
        run_dirs = sorted(data_stem_dir.glob("*"), reverse=True)
        run_dirs = [d for d in run_dirs if d.is_dir() and not d.name.startswith("logs")]
    else:
        run_dirs = sorted(setting_dir.glob("*/*"), reverse=True)
        run_dirs = [d for d in run_dirs if d.is_dir() and not d.name.startswith("logs")]
    if not run_dirs:
        # Fallback: flat structure
        run_dirs = sorted(setting_dir.glob("*"), reverse=True)
        run_dirs = [d for d in run_dirs if d.is_dir()]

    if not run_dirs:
        print(f"WARNING: No run directories found in {setting_dir}")
        return None

    return run_dirs[0]


def load_chess_predictions(result_dir: Path) -> dict[int, str]:
    """Load CHESS predictions (question_id -> SQL) from a result directory."""
    predictions_file = result_dir / "-predictions.json"
    if not predictions_file.exists():
        print(f"WARNING: Predictions file not found: {predictions_file}")
        return {}

    with open(predictions_file, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # Parse: keys are question_id strings, values are "SQL\t----- bird -----\tdb_id" or 0
    predictions = {}
    for qid_str, value in raw.items():
        if value == 0:
            continue
        qid = int(qid_str)
        # Extract SQL from the BIRD format
        if isinstance(value, str) and "\t----- bird -----\t" in value:
            sql = value.split("\t----- bird -----\t")[0].strip()
            predictions[qid] = sql
        elif isinstance(value, str):
            predictions[qid] = value.strip()

    return predictions


def load_chess_execution_history(result_dir: Path, question_id: int, db_id: str) -> list[dict]:
    """Load CHESS execution history for a specific question."""
    history_file = result_dir / f"{question_id}_{db_id}.json"
    if not history_file.exists():
        return []
    with open(history_file, "r", encoding="utf-8") as f:
        return json.load(f)


def execute_sql_on_sqlite(db_id: str, sql: str) -> tuple[list[str], list[list]]:
    """Execute SQL against a SQLite database.

    Returns (column_names, rows).
    """
    db_path = CHESS_DB_ROOT / db_id / f"{db_id}.sqlite"
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    conn = sqlite3.connect(str(db_path))
    try:
        cursor = conn.execute(sql)
        columns = [desc[0] for desc in cursor.description] if cursor.description else []
        rows = cursor.fetchall()
        return columns, rows
    finally:
        conn.close()


def sql_results_to_sparql_bindings(columns: list[str], rows: list[list]) -> list[dict]:
    """Convert SQL query results to SPARQL-JSON binding format.

    Each row becomes a dict of {col_name: {"type": "literal", "value": str_value}}.
    """
    bindings = []
    for row in rows:
        binding = {}
        for col, val in zip(columns, row):
            if val is None:
                continue
            binding[col] = {
                "type": "literal",
                "value": str(val),
            }
        bindings.append(binding)
    return bindings


def build_query_index(dev_json_path: Path = None) -> dict[int, dict]:
    """Build index from CHESS question_id to our query metadata.

    Args:
        dev_json_path: Path to dev.json file. Defaults to CHESS_DEV_JSON.

    Returns dict mapping question_id -> {query_id, dataset, datasets, query_text, query_set, ...}
    """
    if dev_json_path is None:
        dev_json_path = CHESS_DEV_JSON

    with open(dev_json_path, "r", encoding="utf-8") as f:
        dev_questions = json.load(f)

    # Also build a lookup from original_query_id to AdaptiveGroundTruth
    agt_lookup = {q.query_id: q for q in ALL_EXPERIMENTAL_QUERIES}

    index = {}
    for q in dev_questions:
        original_id = q["original_query_id"]
        agt = agt_lookup.get(original_id)
        index[q["question_id"]] = {
            "question_id": q["question_id"],
            "original_query_id": original_id,
            "db_id": q["db_id"],
            "question": q["question"],
            "query_set": q["query_set"],
            "dataset": agt.dataset if agt else q["db_id"].upper(),
            "datasets": agt.datasets if agt else [q["db_id"].upper()],
        }
    return index


def create_trace(
    question_meta: dict,
    sql: str,
    bindings: list[dict],
    model_key: str,
    llm_model_name: str,
    success: bool,
    error_message: str | None = None,
    chess_history: list[dict] | None = None,
) -> dict:
    """Create a trace dict compatible with the project's evaluation pipeline."""
    run_id = uuid.uuid4().hex[:8]
    now = datetime.now()

    return {
        "run_id": run_id,
        "timestamp": str(now),
        "approach": "text2sql_chess",
        "llm_model": llm_model_name,
        "run_number": 1,
        "query_metadata": {
            "query_id": question_meta["original_query_id"],
            "query_set": question_meta["query_set"],
            "query_text": question_meta["question"],
            "dataset": question_meta["dataset"],
            "datasets": question_meta["datasets"],
        },
        "retrieval_phases": [],
        "generation_phases": [],
        "final_sparql": None,
        "final_sql": sql,
        "final_results": bindings,
        "final_result_count": len(bindings),
        "success": success,
        "is_unanswerable": False,
        "unanswerable_reason": "",
        "error_message": error_message,
        "result_metrics": {},
        "token_usage": {},
        "timing": {},
        "total_iterations": 0,
        "retrieval_iterations": 0,
        "generation_iterations": 0,
        "total_llm_calls": 0,
        "total_sparql_executions": 0,
        "strategy_used": "text2sql_chess",
        "ground_truth_query_id": question_meta["original_query_id"],
        "chess_execution_history": chess_history or [],
    }


def save_traces(
    traces: list[dict],
    output_dir: Path,
    experiment_name: str,
) -> Path:
    """Save traces and create experiment directory structure."""
    exp_dir = output_dir / experiment_name
    traces_dir = exp_dir / "traces"
    traces_dir.mkdir(parents=True, exist_ok=True)

    for trace in traces:
        query_id = trace["query_metadata"]["query_id"]
        approach = trace["approach"]
        run_id = trace["run_id"]
        filename = f"{query_id}_{approach}_{run_id}.json"
        with open(traces_dir / filename, "w", encoding="utf-8") as f:
            json.dump(trace, f, indent=2, default=str)

    # Create a simple checkpoint file
    checkpoint = {
        "completed_runs": [
            f"{t['query_metadata']['query_id']}_{t['llm_model']}_{t['approach']}_run{t['run_number']}"
            for t in traces
        ],
        "last_updated": str(datetime.now()),
    }
    with open(exp_dir / "checkpoint.json", "w", encoding="utf-8") as f:
        json.dump(checkpoint, f, indent=2)

    print(f"\nSaved {len(traces)} traces to {traces_dir}")
    return exp_dir


def run_experiment_for_model(
    model_key: str,
    experiment_name: str,
    output_dir: Path,
    skip_chess_run: bool = False,
    dev_json_path: Path = None,
) -> list[dict]:
    """Run the full CHESS experiment for one model and return traces."""
    model_config = CHESS_MODELS[model_key]
    query_index = build_query_index(dev_json_path)
    traces = []

    if skip_chess_run:
        # Find existing CHESS results
        setting_name = model_config["config"].replace(".yaml", "")
        results_base = CHESS_ROOT / "results" / "dev" / setting_name
        if results_base.exists():
            run_dirs = sorted(results_base.glob("*/*"), reverse=True)
            if not run_dirs:
                run_dirs = sorted(
                    [d for d in results_base.iterdir() if d.is_dir()], reverse=True
                )
            result_dir = run_dirs[0] if run_dirs else None
        else:
            result_dir = None
    else:
        result_dir = run_chess_pipeline(model_key, dev_json_path=dev_json_path)

    if result_dir is None:
        print(f"ERROR: No CHESS results found for {model_key}")
        return traces

    print(f"\nLoading predictions from: {result_dir}")
    predictions = load_chess_predictions(result_dir)
    print(f"  {len(predictions)} predictions loaded")

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
            print(f"  {original_id} ({db_id}): {len(bindings)} rows")
        except Exception as e:
            bindings = []
            success = False
            error = str(e)
            print(f"  {original_id} ({db_id}): ERROR - {e}")

        chess_history = load_chess_execution_history(result_dir, qid, db_id)

        trace = create_trace(
            question_meta=meta,
            sql=sql,
            bindings=bindings,
            model_key=model_key,
            llm_model_name=model_config["llm_model_name"],
            success=success,
            error_message=error,
            chess_history=chess_history,
        )
        traces.append(trace)

    return traces


def evaluate_chess_traces(exp_dir: Path) -> dict | None:
    """Evaluate CHESS traces using GT cache and adaptive metrics.

    Uses importlib to load evaluation modules directly, bypassing the
    problematic src.evaluation.__init__.py import chain.

    Returns evaluation results dict, or None if evaluation failed.
    """
    from data.queries.experimental_corpus import get_experimental_query

    traces_dir = exp_dir / "traces"
    if not traces_dir.exists():
        print(f"No traces directory found: {traces_dir}")
        return None

    # Load GT cache
    if not GT_CACHE_FILE.exists():
        print(f"GT cache not found: {GT_CACHE_FILE}")
        print("  Run: python scripts/extract_ground_truth_cache.py")
        return None

    with open(GT_CACHE_FILE, "r", encoding="utf-8") as f:
        gt_cache = json.load(f)
    gt_queries = gt_cache.get("queries", {})
    print(f"Loaded GT cache: {len(gt_queries)} queries")

    # Load evaluation module (bypasses __init__.py)
    try:
        tm = _load_tiered_metrics()
    except Exception as e:
        print(f"Failed to load evaluation module: {e}")
        return None

    trace_files = list(traces_dir.glob("*.json"))
    print(f"Evaluating {len(trace_files)} traces...")

    all_metrics = []
    metrics_by_query_set = {}
    results_traces = {}

    for trace_file in sorted(trace_files):
        with open(trace_file, "r", encoding="utf-8") as f:
            trace = json.load(f)

        query_id = trace.get("query_metadata", {}).get("query_id") or trace.get("ground_truth_query_id")
        if not query_id:
            print(f"  Skipping {trace_file.name}: no query_id")
            continue

        ground_truth = get_experimental_query(query_id)
        if not ground_truth:
            print(f"  Skipping {trace_file.name}: no ground truth for {query_id}")
            continue

        cached_gt = gt_queries.get(query_id)
        if not cached_gt or not cached_gt.get("success"):
            print(f"  Skipping {trace_file.name}: no cached GT for {query_id}")
            continue

        generated_results = trace.get("final_results", [])

        # Extract cached GT data
        gt_tuples = {tuple(t) for t in cached_gt.get("gt_tuples", [])}
        gt_col_names = cached_gt.get("gt_columns", [])
        columns_with_nulls = cached_gt.get("null_columns", [])
        values_by_level = cached_gt.get("values_by_level", {})
        essential_values = set(values_by_level.get("PREFERRED", []))
        preferred_values = essential_values
        acceptable_values = set(values_by_level.get("ACCEPTABLE", []))

        # Evaluate all GT variants, pick by best tuple F1 (consistent with inline path)
        cached_variants = cached_gt.get("query_variants", [])
        dataset_id = ground_truth.dataset
        llm_col_vals = tm.extract_llm_column_values(generated_results)
        llm_col_list = [
            k for k in (generated_results[0].keys() if generated_results else [])
            if not k.startswith("_")
        ]

        best_metrics: tm.AdaptiveMetricsResult | None = None
        best_tuple_f1 = -1.0
        num_variants = max(len(ground_truth.sparql_queries), len(cached_variants), 1)

        try:
            for vi in range(num_variants):
                # Load variant-specific cached data
                if vi < len(cached_variants) and cached_variants[vi].get("success"):
                    vi_cached = cached_variants[vi]
                    vi_gt_tuples = {tuple(t) for t in vi_cached.get("gt_tuples", [])}
                    vi_gt_col_names = vi_cached.get("gt_columns", [])
                    vi_columns_with_nulls = vi_cached.get("null_columns", [])
                    vi_values_by_level = vi_cached.get("values_by_level", {})
                elif vi == 0:
                    vi_gt_tuples = gt_tuples
                    vi_gt_col_names = gt_col_names
                    vi_columns_with_nulls = columns_with_nulls
                    vi_values_by_level = cached_gt.get("values_by_level", {})
                else:
                    continue

                if not vi_gt_tuples:
                    continue

                # --- BIND-label column filtering (mirrors _evaluate_single_query) ---
                try:
                    vi_cols_obj = ground_truth.get_columns_for_query(vi)
                except (IndexError, Exception):
                    if vi == 0:
                        vi_cols_obj = ground_truth.get_columns_for_query(0)
                    else:
                        continue

                bind_label_indices = {
                    i for i, col in enumerate(vi_cols_obj) if getattr(col, "is_bind_label", False)
                }

                vi_essential = set(vi_values_by_level.get("PREFERRED", []))
                vi_preferred = vi_essential
                vi_acceptable = set(vi_values_by_level.get("ACCEPTABLE", []))

                if bind_label_indices:
                    keep_indices = [i for i in range(len(vi_cols_obj)) if i not in bind_label_indices]

                    bind_values = set()
                    for t in vi_gt_tuples:
                        for i in bind_label_indices:
                            if i < len(t) and t[i]:
                                bind_values.add(t[i])
                    vi_acceptable = vi_acceptable - bind_values

                    vi_gt_tuples = {tuple(t[i] for i in keep_indices) for t in vi_gt_tuples}
                    vi_gt_col_names = [vi_gt_col_names[i] for i in keep_indices]

                    old_to_new = {old: new for new, old in enumerate(keep_indices)}
                    vi_columns_with_nulls = [old_to_new[c] for c in vi_columns_with_nulls if c in old_to_new]

                    vi_cols_obj = [vi_cols_obj[i] for i in keep_indices]

                # Build column_values for this variant
                vi_num_cols = len(vi_gt_col_names)
                vi_column_values: list[set[str]] = [set() for _ in range(vi_num_cols)]
                for t in vi_gt_tuples:
                    for col_idx in range(min(len(t), vi_num_cols)):
                        val = t[col_idx]
                        if val and val.strip():
                            vi_column_values[col_idx].add(val)

                vi_gt_sparql = ground_truth.sparql_queries[vi] if vi < len(ground_truth.sparql_queries) else None
                vi_measurement = ground_truth.get_measurement_column_indices(query_index=vi)

                # Tuple metrics for this variant
                vi_metrics = tm.calculate_adaptive_metrics(
                    llm_results=generated_results,
                    gt_tuples=vi_gt_tuples,
                    columns_with_nulls=vi_columns_with_nulls,
                    essential_values=vi_essential,
                    preferred_values=vi_preferred,
                    acceptable_values=vi_acceptable,
                    column_values=vi_column_values,
                    measurement_columns=vi_measurement,
                    gt_sparql=vi_gt_sparql,
                    llm_sparql=None,  # CHESS produces SQL, not SPARQL
                    gt_col_names=vi_gt_col_names,
                    dataset_id=dataset_id,
                )
                vi_metrics.llm_columns = llm_col_list

                # Schema metrics for this variant (vi_cols_obj already BIND-filtered)
                vi_levels = [col.level for col in vi_cols_obj]
                vi_is_measurement = [col.is_measurement for col in vi_cols_obj]
                vi_concepts = [col.semantic_concept for col in vi_cols_obj]

                vi_schema = tm.calculate_schema_metrics(
                    llm_col_vals, vi_column_values, vi_levels,
                    vi_is_measurement, vi_concepts,
                )
                vi_metrics.schema_recall = vi_schema[0]
                vi_metrics.schema_precision = vi_schema[1]
                vi_metrics.schema_expected_count = vi_schema[2]
                vi_metrics.schema_matched_count = vi_schema[3]
                vi_metrics.schema_llm_columns = vi_schema[4]
                vi_metrics.schema_llm_matched = vi_schema[5]
                if vi_metrics.schema_recall + vi_metrics.schema_precision > 0:
                    vi_metrics.schema_f1 = (
                        2 * vi_metrics.schema_recall * vi_metrics.schema_precision
                        / (vi_metrics.schema_recall + vi_metrics.schema_precision)
                    )

                # Track best by tuple F1 (consistent with inline path)
                if vi_metrics.best_f1 > best_tuple_f1:
                    best_tuple_f1 = vi_metrics.best_f1
                    best_metrics = vi_metrics

            metrics_dict = best_metrics.to_dict() if best_metrics else {}
        except Exception as e:
            print(f"  Error evaluating {query_id}: {e}")
            metrics_dict = {}

        query_set = ground_truth.query_set if ground_truth else "X"
        print(
            f"  {query_id}: "
            f"Recall={metrics_dict.get('best_recall', 0):.0%} "
            f"Prec={metrics_dict.get('best_precision', 0):.0%} "
            f"F1={metrics_dict.get('best_f1', 0):.0%} | "
            f"Schema F1={metrics_dict.get('schema_f1', 0):.0%}"
        )

        all_metrics.append(metrics_dict)
        metrics_by_query_set.setdefault(query_set, []).append(metrics_dict)

        results_traces[trace_file.stem] = {
            "query_id": query_id,
            "approach": trace.get("approach", "text2sql_chess"),
            "query_set": query_set,
            "dataset": ground_truth.dataset,
            "success": trace.get("success", False),
            "adaptive_metrics": metrics_dict,
        }

    if not all_metrics:
        print("No traces could be evaluated.")
        return None

    # Calculate summary
    def _mean(lst, key):
        vals = [m.get(key, 0) for m in lst]
        return sum(vals) / len(vals) if vals else 0

    def _summary(mlist):
        n = len(mlist)
        return {
            "count": n,
            "best_recall_mean": _mean(mlist, "best_recall"),
            "best_precision_mean": _mean(mlist, "best_precision"),
            "best_f1_mean": _mean(mlist, "best_f1"),
            "schema_recall_mean": _mean(mlist, "schema_recall"),
            "schema_precision_mean": _mean(mlist, "schema_precision"),
            "schema_f1_mean": _mean(mlist, "schema_f1"),
            "llm_row_count_mean": _mean(mlist, "llm_row_count"),
        }

    overall = _summary(all_metrics)
    by_query_set = {qs: _summary(ml) for qs, ml in metrics_by_query_set.items()}

    results = {
        "experiment_name": exp_dir.name,
        "evaluation_mode": "adaptive",
        "reevaluation_timestamp": datetime.now().isoformat(),
        "total_traces": len(trace_files),
        "evaluated_traces": len(all_metrics),
        "traces": results_traces,
        "summary": {
            "overall": overall,
            "by_approach": {"text2sql_chess": _summary(all_metrics)},
            "by_query_set": by_query_set,
        },
    }

    # Save evaluation JSON
    output_file = exp_dir / "adaptive_evaluation.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    # Print summary
    print(f"\n{'=' * 60}")
    print("CHESS ADAPTIVE EVALUATION SUMMARY")
    print(f"{'=' * 60}")
    print(f"\nOverall ({overall['count']} traces):")
    print(f"  Results Metrics (WHERE clause / data correctness):")
    print(f"    Recall:    {overall['best_recall_mean']:.2%}")
    print(f"    Precision: {overall['best_precision_mean']:.2%}")
    print(f"    F1:        {overall['best_f1_mean']:.2%}")
    print(f"  Schema Metrics (SELECT clause / column correctness):")
    print(f"    Recall:    {overall['schema_recall_mean']:.2%}")
    print(f"    Precision: {overall['schema_precision_mean']:.2%}")
    print(f"    F1:        {overall['schema_f1_mean']:.2%}")

    if len(by_query_set) > 1:
        print(f"\nBy Query Set:")
        for qs in sorted(by_query_set):
            s = by_query_set[qs]
            print(f"  {qs}: Results F1={s['best_f1_mean']:.0%}, Schema F1={s['schema_f1_mean']:.0%} (n={s['count']})")

    print(f"\nResults saved to: {output_file}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Run CHESS Text2SQL experiment")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["gpt4"],
        choices=list(CHESS_MODELS.keys()),
        help="LLM models to use (default: gpt4)",
    )
    parser.add_argument(
        "--experiment-name",
        default=None,
        help="Experiment name (default: chess_{models}_{date})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "experiments",
        help="Output directory for traces",
    )
    parser.add_argument(
        "--skip-preprocess",
        action="store_true",
        help="Skip CHESS preprocessing (if already done)",
    )
    parser.add_argument(
        "--skip-chess-run",
        action="store_true",
        help="Skip CHESS pipeline (use existing results)",
    )
    parser.add_argument(
        "--merged",
        action="store_true",
        help="Use merged database (all datasets in one DB, includes CROSS queries)",
    )
    parser.add_argument(
        "--heterogeneous",
        action="store_true",
        help="Use heterogeneous merged database (tables split with renamed columns)",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="Override dev.json path (e.g. for test subsets)",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip automatic evaluation after trace generation",
    )
    args = parser.parse_args()

    # Select dev.json path: explicit override > heterogeneous > merged > default
    if args.data_path is not None:
        dev_json_path = args.data_path
    elif args.heterogeneous:
        dev_json_path = CHESS_DEV_JSON_HETEROGENEOUS
    elif args.merged:
        dev_json_path = CHESS_DEV_JSON_MERGED
    else:
        dev_json_path = CHESS_DEV_JSON

    # Default experiment name
    if args.experiment_name is None:
        models_str = "_".join(args.models)
        date_str = datetime.now().strftime("%Y%m%d")
        if args.heterogeneous:
            db_suffix = "_heterogeneous"
        elif args.merged:
            db_suffix = "_merged"
        else:
            db_suffix = ""
        args.experiment_name = f"chess_{models_str}{db_suffix}_{date_str}"

    mode_label = "MERGED" if args.merged else "INDIVIDUAL"
    print("=" * 70)
    print(f"CHESS Text2SQL Experiment: {args.experiment_name} ({mode_label})")
    print(f"Models: {', '.join(args.models)}")
    print(f"Data: {dev_json_path}")
    print("=" * 70)

    # Step 1: Preprocessing
    if not args.skip_preprocess:
        if args.merged:
            needs_preprocess = not is_preprocessed("merged")
        else:
            needs_preprocess = any(
                not is_preprocessed(db_id) for db_id in DATASET_TO_DB_ID.values()
            )

        if needs_preprocess:
            run_chess_preprocess("merged" if args.merged else "all")
        else:
            print("All databases already preprocessed, skipping.")
    else:
        print("Skipping preprocessing (--skip-preprocess).")

    # Step 2: Run CHESS for each model and collect traces
    all_traces = []
    for model_key in args.models:
        traces = run_experiment_for_model(
            model_key=model_key,
            experiment_name=args.experiment_name,
            output_dir=args.output_dir,
            skip_chess_run=args.skip_chess_run,
            dev_json_path=dev_json_path,
        )
        all_traces.extend(traces)

    # Step 3: Save all traces
    if all_traces:
        exp_dir = save_traces(all_traces, args.output_dir, args.experiment_name)

        # Summary
        print(f"\n{'=' * 70}")
        print(f"Experiment complete: {args.experiment_name}")
        print(f"  Total traces: {len(all_traces)}")
        successful = sum(1 for t in all_traces if t["success"])
        print(f"  Successful: {successful}/{len(all_traces)}")
        print(f"  Output: {exp_dir}")
        print(f"{'=' * 70}")

        # Step 4: Evaluate traces
        if not args.skip_eval:
            print(f"\n{'=' * 70}")
            print("STEP 4: Evaluating traces...")
            print(f"{'=' * 70}")
            evaluate_chess_traces(exp_dir)
        else:
            print(f"\nSkipping evaluation (--skip-eval).")
            print(f"To evaluate later, run:")
            print(f"  python scripts/reevaluate_tiered.py {args.experiment_name}")
    else:
        print("\nNo traces generated. Check CHESS output for errors.")


if __name__ == "__main__":
    main()
