#!/usr/bin/env python3
"""
Full Experiment Runner for VKGQA Evaluation.

Runs experiments with:
- Multiple LLM models (Deepseek, Claude, ChatGPT)
- Multiple approaches (grep, semantic)
- Multiple runs per query for statistical significance
- Automatic F1-Score and Execution Accuracy calculation
- CHECKPOINT/RESUME: Results are saved incrementally, allowing you to stop
  and resume experiments at any time.

Usage:
    python scripts/run_full_experiment.py --help
    python scripts/run_full_experiment.py --models deepseek --queries A01 A02 --runs 3
    python scripts/run_full_experiment.py --models deepseek claude gpt4 --query-sets A B --runs 5

    # Resume a previous experiment:
    python scripts/run_full_experiment.py --resume exp_deepseek_20240315_120000
"""

import argparse
import asyncio
import http.server
import json
import logging
import socketserver
import sys
import threading
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Literal

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tqdm import tqdm

# Query structure from tiered tuples, actual queries from experimental corpus
from data.queries.use_case_queries_tiered_tuples import (
    AdaptiveGroundTruth,
    RelevanceLevel,
    SPARQLComplexity,
    ColumnDef,
    load_large_ground_truth_for_queries,
)
from data.queries.experimental_corpus import (
    SET_BASE, SET_SYN, SET_TYPO, SET_UNDER, SET_CROSS,
    ALL_EXPERIMENTAL_QUERIES,
    PAIRED_QUERIES,
    get_experimental_query,
    get_queries_by_set,
    get_baseline_for_variant,
    CORPUS_STATS,
)
from src.agents.orchestrator import OrchestratorAgent
from src.concurrency.slot_manager import get_slot_manager
from src.config import (
    RESULTS_DIR,
    CACHE_DIR,
    dataset_size_context,
    get_size_for_query_set,
    get_ontop_endpoint,
    slot_context,
)
from src.evaluation.tiered_metrics_tuples import (
    AdaptiveMetricsResult,
    evaluate_adaptive_ground_truth,
    extract_result_values,
    detect_active_levels,
    is_trivial_value,
)
from src.evaluation.retrieval_ground_truth import build_schema_gt_for_query
from src.evaluation.retrieval_metrics import calculate_retrieval_metrics
from src.validation.schema_graph import get_combined_schema
# Import tiered dashboard and reevaluate functions
# Handle both running as script and as module
try:
    from scripts.tiered_dashboard import get_tiered_dashboard_html
    from scripts.reevaluate_tiered import evaluate_trace_adaptive
except ModuleNotFoundError:
    from tiered_dashboard import get_tiered_dashboard_html
    from reevaluate_tiered import evaluate_trace_adaptive

# Global ground truth cache (shared across all experiments)
GLOBAL_GROUND_TRUTH_CACHE = CACHE_DIR / "ground_truth_results.json"
# Lightweight preview cache for dashboard (only first 5 results + count per query)
GROUND_TRUTH_PREVIEW_CACHE = CACHE_DIR / "ground_truth_preview.json"
from src.tools.sparql_tools import execute_sparql, execute_sparql_async
from src.tracing import (
    ExperimentRunTrace,
    ExperimentTraceCollector,
    ResultMetrics,
    TraceSummary,
    convert_use_case_query_to_metadata,
    create_experiment_batch,
    get_tracer,
    trace_run_context,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Suppress noisy loggers
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("langchain").setLevel(logging.WARNING)


# ============================================================================
# LLM Model Configuration
# ============================================================================

LLM_MODELS = {
    "deepseek": "deepseek-chat",
    "claude": "claude-3-5-sonnet-20241022",
    "gpt4": "gpt-4o",
    "gpt4-mini": "gpt-4o-mini",
    "gpt5": "gpt-5.4",
    "qwen": "qwen3.5:27b",
    "local_proxy": "openai/qwen3",
    "qwen35-or": "openrouter/qwen/qwen3.5-27b",
    "qwen36-or": "openrouter/qwen/qwen3.6-27b",
}

APPROACH_TYPES = {
    "grep": "agentic_grep",
    "semantic": "agentic_semantic",
}

# Endpoint mapping for all datasets (for adaptive evaluation)
ENDPOINTS = {
    "EDU": "http://localhost:8080/sparql",
    "TRN": "http://localhost:8081/sparql",
    "NRG": "http://localhost:8082/sparql",
    "BSBM": "http://localhost:8083/sparql",
    "BGEE": "http://localhost:8084/sparql",
    "LCA": "http://localhost:8085/sparql",
}


# ============================================================================
# Query Loading
# ============================================================================

ALL_QUERIES = {
    "BASE": SET_BASE,
    "SYN": SET_SYN,
    "TYPO": SET_TYPO,
    "UNDER": SET_UNDER,
    "CROSS": SET_CROSS,
}


def load_queries(
    query_sets: list[str] | None = None,
    query_ids: list[str] | None = None,
) -> list[AdaptiveGroundTruth]:
    """
    Load queries based on set selection or specific IDs.

    Args:
        query_sets: List of query set names (BASE, SYN, TYPO, UNDER, CROSS)
        query_ids: List of specific query IDs (BASE01, SYN01, TYPO01, etc.)

    Returns:
        List of AdaptiveGroundTruth objects
    """
    queries: list[AdaptiveGroundTruth] = []

    if query_ids:
        # Load specific queries by ID (e.g., BASE01, SYN01, TYPO01)
        query_map = {q.query_id: q for q in ALL_EXPERIMENTAL_QUERIES}
        for qid in query_ids:
            if qid in query_map:
                queries.append(query_map[qid])
            else:
                logger.warning(f"Query ID not found: {qid}")

    elif query_sets:
        # Load entire query sets (BASE, SYN, TYPO, UNDER, CROSS)
        for set_name in query_sets:
            set_name = set_name.upper()
            if set_name in ALL_QUERIES:
                queries.extend(ALL_QUERIES[set_name])
            else:
                logger.warning(f"Query set not found: {set_name}")

    else:
        # Load all queries from experimental corpus
        queries = list(ALL_EXPERIMENTAL_QUERIES)

    # Pre-load cached ground truth for LARGE queries (EDU, TRN)
    # This avoids expensive live SPARQL execution during evaluation
    load_large_ground_truth_for_queries(queries)

    return queries


# ============================================================================
# Checkpoint Management
# ============================================================================

def get_run_key(query_id: str, model_key: str, approach_key: str, run_num: int) -> str:
    """Generate a unique key for a specific run."""
    return f"{query_id}_{model_key}_{approach_key}_run{run_num}"


def load_checkpoint(output_dir: Path) -> dict:
    """
    Load checkpoint data from a previous run.

    Returns:
        Dict with:
        - 'completed_runs': set of run keys
        - 'trace_summaries': list of TraceSummary objects (lightweight, ~1 KB each)

    Note: Ground truth cache is loaded from global cache, not from checkpoint.
    Note: Full traces are NOT loaded into RAM - only lightweight summaries.
    """
    checkpoint_file = output_dir / "checkpoint.json"

    checkpoint = {
        "completed_runs": set(),
        "trace_summaries": [],
    }

    if checkpoint_file.exists():
        try:
            with open(checkpoint_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                checkpoint["completed_runs"] = set(data.get("completed_runs", []))
                logger.info(f"Loaded checkpoint: {len(checkpoint['completed_runs'])} completed runs")
        except Exception as e:
            logger.warning(f"Could not load checkpoint: {e}")

    # Load lightweight trace summaries (NOT full traces - saves ~95% RAM)
    checkpoint["trace_summaries"] = load_trace_summaries(output_dir)

    return checkpoint


def save_checkpoint(
    output_dir: Path,
    completed_runs: set[str],
    ground_truth_cache: dict[str, dict | None] | None = None,
) -> None:
    """Save checkpoint data for resume capability.

    Note: Ground truth cache is stored globally, not in per-experiment checkpoint.
    The ground_truth_cache parameter is kept for backwards compatibility but ignored.
    """
    checkpoint_file = output_dir / "checkpoint.json"

    data = {
        "completed_runs": list(completed_runs),
        # Ground truth is stored in global cache, not here (saves ~640MB per experiment)
        "last_updated": datetime.now().isoformat(),
    }

    with open(checkpoint_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


def load_global_ground_truth_cache() -> dict[str, dict | None]:
    """Load the global ground truth cache (shared across all experiments)."""
    if GLOBAL_GROUND_TRUTH_CACHE.exists():
        try:
            with open(GLOBAL_GROUND_TRUTH_CACHE, "r", encoding="utf-8") as f:
                cache = json.load(f)
                logger.info(f"Loaded {len(cache)} ground truth results from global cache")
                return cache
        except Exception as e:
            logger.warning(f"Could not load global ground truth cache: {e}")
    return {}


def save_global_ground_truth_cache(cache: dict[str, dict | None]) -> None:
    """Save to the global ground truth cache."""
    GLOBAL_GROUND_TRUTH_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(GLOBAL_GROUND_TRUTH_CACHE, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, default=str)
    logger.info(f"Saved {len(cache)} ground truth results to global cache")


def save_trace(output_dir: Path, trace: ExperimentRunTrace) -> TraceSummary:
    """
    Save a single trace immediately after completion.

    Returns:
        TraceSummary for the saved trace (lightweight, ~1 KB)
    """
    traces_dir = output_dir / "traces"
    traces_dir.mkdir(parents=True, exist_ok=True)

    trace_filename = f"{trace.query_metadata.query_id}_{trace.approach}_{trace.run_id}.json"
    trace_file = traces_dir / trace_filename
    with open(trace_file, "w", encoding="utf-8") as f:
        json.dump(trace.model_dump(), f, indent=2, default=str)

    # Create and return lightweight summary
    return trace.to_summary(trace_file=trace_filename)


def load_trace_summaries(output_dir: Path) -> list[TraceSummary]:
    """
    Load lightweight trace summaries from the summaries index file.

    Falls back to creating summaries from full traces if index doesn't exist.
    This is much faster than loading full traces (~1 KB vs ~500 KB+ per trace).
    """
    summaries_file = output_dir / "trace_summaries.json"

    if summaries_file.exists():
        try:
            with open(summaries_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                summaries = [TraceSummary.model_validate(s) for s in data]
                logger.info(f"Loaded {len(summaries)} trace summaries from index")
                return summaries
        except Exception as e:
            logger.warning(f"Could not load trace summaries index: {e}")

    # Fallback: Create summaries from full traces (slower, but works for legacy data)
    traces_dir = output_dir / "traces"
    summaries = []
    if traces_dir.exists():
        for trace_file in traces_dir.glob("*.json"):
            try:
                with open(trace_file, "r", encoding="utf-8") as f:
                    trace_data = json.load(f)
                    trace_obj = ExperimentRunTrace.model_validate(trace_data)
                    summary = trace_obj.to_summary(trace_file=trace_file.name)
                    summaries.append(summary)
            except Exception as e:
                logger.warning(f"Could not create summary from {trace_file}: {e}")

        if summaries:
            # Save index for future loads
            save_trace_summaries(output_dir, summaries)
            logger.info(f"Created {len(summaries)} trace summaries from legacy traces")

    return summaries


def save_trace_summaries(output_dir: Path, summaries: list[TraceSummary]) -> None:
    """Save the trace summaries index file."""
    summaries_file = output_dir / "trace_summaries.json"
    with open(summaries_file, "w", encoding="utf-8") as f:
        json.dump([s.model_dump() for s in summaries], f, indent=2, default=str)


def save_adaptive_evaluation(
    output_dir: Path,
    experiment_name: str,
    adaptive_eval_traces: dict[str, dict],
    models: list[str],
    approaches: list[str],
    expected_total: int = 0,
) -> None:
    """Save adaptive evaluation results in the format expected by tiered_dashboard.py."""
    # Calculate summary statistics
    all_metrics = [t.get("adaptive_metrics", {}) for t in adaptive_eval_traces.values()]

    def calc_mean(key: str) -> float:
        values = [m.get(key, 0) for m in all_metrics if m]
        return sum(values) / len(values) if values else 0.0

    def calc_rate(key: str) -> float:
        values = [t.get(key, False) for t in adaptive_eval_traces.values()]
        return sum(1 for v in values if v) / len(values) if values else 0.0

    # Group by approach
    by_approach: dict[str, list[dict]] = {}
    for trace in adaptive_eval_traces.values():
        approach = trace.get("approach", "unknown")
        if approach not in by_approach:
            by_approach[approach] = []
        by_approach[approach].append(trace.get("adaptive_metrics", {}))

    def _calc_group_summary(metrics_list: list[dict]) -> dict:
        n = len(metrics_list)
        if n == 0:
            return {"count": 0}
        return {
            "count": n,
            # Results metrics (WHERE clause)
            "best_recall_mean": sum(m.get("best_recall", 0) for m in metrics_list) / n,
            "best_precision_mean": sum(m.get("best_precision", 0) for m in metrics_list) / n,
            "best_f1_mean": sum(m.get("best_f1", 0) for m in metrics_list) / n,
            # Schema metrics (SELECT clause)
            "schema_recall_mean": sum(m.get("schema_recall", 0) for m in metrics_list) / n,
            "schema_precision_mean": sum(m.get("schema_precision", 0) for m in metrics_list) / n,
            "schema_f1_mean": sum(m.get("schema_f1", 0) for m in metrics_list) / n,
            # Retrieval metrics (schema-aware)
            "retrieval_path_coherence_mean": sum(m.get("retrieval_path_coherence", 0) for m in metrics_list) / n,
            "retrieval_triple_f1_mean": sum(m.get("retrieval_triple_f1", 0) for m in metrics_list) / n,
            "retrieval_triple_precision_mean": sum(m.get("retrieval_triple_precision", 0) for m in metrics_list) / n,
            "retrieval_triple_recall_mean": sum(m.get("retrieval_triple_recall", 0) for m in metrics_list) / n,
        }

    approach_summaries = {}
    for approach, metrics_list in by_approach.items():
        approach_summaries[approach] = _calc_group_summary(metrics_list)

    # Group by query set
    by_query_set: dict[str, list[dict]] = {}
    for trace in adaptive_eval_traces.values():
        qset = trace.get("query_set", "X")
        if qset not in by_query_set:
            by_query_set[qset] = []
        by_query_set[qset].append(trace.get("adaptive_metrics", {}))

    query_set_summaries = {}
    for qset, metrics_list in by_query_set.items():
        query_set_summaries[qset] = _calc_group_summary(metrics_list)

    evaluation_data = {
        "experiment_name": experiment_name,
        "evaluation_mode": "adaptive",
        "reevaluation_timestamp": datetime.now().isoformat(),
        "total_traces": len(adaptive_eval_traces),
        "expected_total": expected_total,
        "traces": adaptive_eval_traces,
        "summary": {
            "overall": {
                "count": len(all_metrics),
                # Results metrics (WHERE clause)
                "best_recall_mean": calc_mean("best_recall"),
                "best_precision_mean": calc_mean("best_precision"),
                "best_f1_mean": calc_mean("best_f1"),
                # Schema metrics (SELECT clause)
                "schema_recall_mean": calc_mean("schema_recall"),
                "schema_precision_mean": calc_mean("schema_precision"),
                "schema_f1_mean": calc_mean("schema_f1"),
                # Retrieval metrics (schema-aware)
                "retrieval_path_coherence_mean": calc_mean("retrieval_path_coherence"),
                "retrieval_triple_f1_mean": calc_mean("retrieval_triple_f1"),
                "retrieval_triple_precision_mean": calc_mean("retrieval_triple_precision"),
                "retrieval_triple_recall_mean": calc_mean("retrieval_triple_recall"),
                # Level activity rates
                "essential_active_rate": calc_rate("essential_active"),
                "preferred_active_rate": calc_rate("preferred_active"),
                "acceptable_active_rate": calc_rate("acceptable_active"),
                "essential_matches_mean": calc_mean("essential_matches"),
                "preferred_matches_mean": calc_mean("preferred_matches"),
                "acceptable_matches_mean": calc_mean("acceptable_matches"),
                "variants_evaluated_mean": calc_mean("variants_evaluated"),
            },
            "by_approach": approach_summaries,
            "by_query_set": query_set_summaries,
        },
    }

    output_file = output_dir / "adaptive_evaluation.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(evaluation_data, f, indent=2, default=str)


def update_summary(output_dir: Path, experiment_name: str, all_traces: list,
                   queries: list, models: list, approaches: list, runs_per_query: int) -> dict:
    """Update and save the summary file with current results."""
    # Convert trace dicts back to objects for summary creation if needed
    results_by_config: dict[str, list] = {}

    for trace in all_traces:
        # Handle both dict and object formats
        if isinstance(trace, dict):
            model_name = trace.get("llm_model", "unknown")
            approach_name = trace.get("approach", "unknown")
        else:
            model_name = trace.llm_model
            approach_name = trace.approach

        # Find model key
        model_key = "unknown"
        for key, value in LLM_MODELS.items():
            if value == model_name or key in model_name.lower():
                model_key = key
                break

        # Find approach key
        approach_key = "unknown"
        for key, value in APPROACH_TYPES.items():
            if value == approach_name:
                approach_key = key
                break

        config_key = f"{model_key}_{approach_key}"
        if config_key not in results_by_config:
            results_by_config[config_key] = []
        results_by_config[config_key].append(trace)

    summary = create_experiment_summary(
        all_traces=all_traces,
        results_by_config=results_by_config,
        queries=queries,
        models=models,
        approaches=approaches,
        runs_per_query=runs_per_query,
        experiment_name=experiment_name,
    )

    summary_file = output_dir / f"{experiment_name}_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    return summary


# ============================================================================
# Live Dashboard Server
# ============================================================================

DASHBOARD_HTML_TEMPLATE = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Experiment Dashboard - {experiment_name}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            color: #eee;
            padding: 20px;
        }}
        .container {{ max-width: 1600px; margin: 0 auto; }}
        header {{
            text-align: center;
            margin-bottom: 30px;
            padding: 20px;
            background: rgba(255,255,255,0.05);
            border-radius: 15px;
            border: 1px solid rgba(255,255,255,0.1);
        }}
        h1 {{
            font-size: 2rem;
            margin-bottom: 10px;
            background: linear-gradient(90deg, #00d4ff, #7b2cbf);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .status {{
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 10px;
            color: #888;
        }}
        .status-dot {{
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background: #00ff88;
            animation: pulse 2s infinite;
        }}
        @keyframes pulse {{
            0%, 100% {{ opacity: 1; }}
            50% {{ opacity: 0.5; }}
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .grid-2 {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 20px;
            margin-bottom: 30px;
        }}
        @media (max-width: 900px) {{
            .grid-2 {{ grid-template-columns: 1fr; }}
        }}
        .card {{
            background: rgba(255,255,255,0.05);
            border-radius: 15px;
            padding: 25px;
            border: 1px solid rgba(255,255,255,0.1);
        }}
        .card h2 {{
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: #888;
            margin-bottom: 15px;
        }}
        .metric {{
            display: flex;
            justify-content: space-between;
            align-items: baseline;
            margin-bottom: 12px;
            padding-bottom: 12px;
            border-bottom: 1px solid rgba(255,255,255,0.05);
        }}
        .metric:last-child {{ margin-bottom: 0; padding-bottom: 0; border-bottom: none; }}
        .metric-label {{ color: #aaa; font-size: 0.95rem; }}
        .metric-value {{ font-size: 1.4rem; font-weight: 600; }}
        .metric-value.good {{ color: #00ff88; }}
        .metric-value.medium {{ color: #ffaa00; }}
        .metric-value.bad {{ color: #ff4466; }}
        .metric-std {{ font-size: 0.8rem; color: #666; margin-left: 5px; }}
        .progress-bar {{
            height: 8px;
            background: rgba(255,255,255,0.1);
            border-radius: 4px;
            overflow: hidden;
            margin-top: 15px;
        }}
        .progress-fill {{
            height: 100%;
            background: linear-gradient(90deg, #00d4ff, #7b2cbf);
            transition: width 0.5s ease;
        }}
        .progress-text {{
            display: flex;
            justify-content: space-between;
            margin-top: 8px;
            font-size: 0.85rem;
            color: #888;
        }}
        .config-grid {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 10px;
        }}
        .config-item {{
            background: rgba(255,255,255,0.03);
            padding: 10px 15px;
            border-radius: 8px;
        }}
        .config-label {{ font-size: 0.75rem; color: #666; text-transform: uppercase; }}
        .config-value {{ font-size: 1rem; margin-top: 3px; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ padding: 12px 15px; text-align: left; }}
        th {{
            background: rgba(255,255,255,0.05);
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: #888;
        }}
        tr:nth-child(even) {{ background: rgba(255,255,255,0.02); }}
        .table-card {{ overflow-x: auto; }}
        .update-time {{ text-align: center; color: #555; font-size: 0.8rem; margin-top: 20px; }}
        .error-message {{
            background: rgba(255, 68, 102, 0.1);
            border: 1px solid rgba(255, 68, 102, 0.3);
            color: #ff4466;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }}
        .chart-container {{
            position: relative;
            height: 250px;
        }}
        /* Query Details Section */
        .section-title {{
            font-size: 1.2rem;
            margin: 30px 0 20px 0;
            padding-bottom: 10px;
            border-bottom: 1px solid rgba(255,255,255,0.1);
            color: #aaa;
        }}
        .query-card {{
            background: rgba(255,255,255,0.03);
            border-radius: 12px;
            margin-bottom: 15px;
            border: 1px solid rgba(255,255,255,0.08);
            overflow: hidden;
        }}
        .query-header {{
            padding: 15px 20px;
            cursor: pointer;
            display: flex;
            justify-content: space-between;
            align-items: center;
            transition: background 0.2s;
        }}
        .query-header:hover {{
            background: rgba(255,255,255,0.05);
        }}
        .query-header-left {{
            display: flex;
            align-items: center;
            gap: 15px;
        }}
        .query-id {{
            font-weight: 600;
            color: #00d4ff;
            font-size: 1rem;
        }}
        .query-approach {{
            font-size: 0.75rem;
            padding: 3px 8px;
            border-radius: 4px;
            text-transform: uppercase;
        }}
        .query-approach.grep {{ background: rgba(123, 44, 191, 0.3); color: #b388ff; }}
        .query-approach.semantic {{ background: rgba(0, 212, 255, 0.3); color: #00d4ff; }}
        .query-metrics {{
            display: flex;
            gap: 20px;
            font-size: 0.9rem;
        }}
        .query-metric {{
            display: flex;
            align-items: center;
            gap: 5px;
        }}
        .query-metric-label {{ color: #666; }}
        .query-metric-value {{ font-weight: 600; }}
        .expand-icon {{
            color: #666;
            transition: transform 0.3s;
        }}
        .query-card.expanded .expand-icon {{
            transform: rotate(180deg);
        }}
        .query-details {{
            display: none;
            padding: 20px;
            border-top: 1px solid rgba(255,255,255,0.05);
            background: rgba(0,0,0,0.2);
        }}
        .query-card.expanded .query-details {{
            display: block;
        }}
        .detail-section {{
            margin-bottom: 20px;
        }}
        .detail-section:last-child {{
            margin-bottom: 0;
        }}
        .detail-label {{
            font-size: 0.75rem;
            text-transform: uppercase;
            color: #666;
            margin-bottom: 8px;
            letter-spacing: 1px;
        }}
        .nl-query {{
            background: rgba(255,255,255,0.05);
            padding: 15px;
            border-radius: 8px;
            font-style: italic;
            color: #ccc;
            line-height: 1.5;
        }}
        .sparql-query {{
            background: rgba(0,0,0,0.3);
            padding: 15px;
            border-radius: 8px;
            font-family: 'Fira Code', 'Monaco', monospace;
            font-size: 0.85rem;
            color: #7dd3fc;
            white-space: pre-wrap;
            overflow-x: auto;
            line-height: 1.4;
        }}
        .results-table {{
            width: 100%;
            font-size: 0.85rem;
            border-radius: 8px;
            overflow: hidden;
        }}
        .results-table th {{
            background: rgba(0,0,0,0.3);
            padding: 10px;
            font-size: 0.7rem;
        }}
        .results-table td {{
            padding: 8px 10px;
            background: rgba(255,255,255,0.02);
            border-bottom: 1px solid rgba(255,255,255,0.05);
            max-width: 300px;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }}
        .no-results {{
            color: #666;
            font-style: italic;
            padding: 15px;
            text-align: center;
        }}
        .loading-gt {{
            color: #888;
            font-style: italic;
            padding: 15px;
            text-align: center;
            animation: pulse 1.5s infinite;
        }}
        @keyframes pulse {{
            0%, 100% {{ opacity: 0.5; }}
            50% {{ opacity: 1; }}
        }}
        .filter-bar {{
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }}
        .filter-btn {{
            padding: 8px 16px;
            border-radius: 20px;
            border: 1px solid rgba(255,255,255,0.2);
            background: transparent;
            color: #aaa;
            cursor: pointer;
            font-size: 0.85rem;
            transition: all 0.2s;
        }}
        .filter-btn:hover {{
            background: rgba(255,255,255,0.1);
        }}
        .filter-btn.active {{
            background: rgba(0, 212, 255, 0.2);
            border-color: #00d4ff;
            color: #00d4ff;
        }}
        .metrics-detail-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
            gap: 10px;
            margin-top: 10px;
        }}
        .metrics-detail-item {{
            background: rgba(255,255,255,0.03);
            padding: 10px;
            border-radius: 6px;
            text-align: center;
        }}
        .metrics-detail-value {{
            font-size: 1.1rem;
            font-weight: 600;
        }}
        .metrics-detail-label {{
            font-size: 0.7rem;
            color: #666;
            margin-top: 3px;
        }}
        /* Comparison Grid */
        .comparison-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-top: 15px;
        }}
        @media (max-width: 1200px) {{
            .comparison-grid {{ grid-template-columns: 1fr; }}
        }}
        .comparison-col {{
            background: rgba(255,255,255,0.02);
            border-radius: 10px;
            padding: 15px;
        }}
        .comparison-header {{
            font-size: 0.85rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1px;
            padding: 8px 12px;
            border-radius: 6px;
            margin-bottom: 15px;
            text-align: center;
        }}
        .comparison-header.generated {{
            background: rgba(0, 212, 255, 0.2);
            color: #00d4ff;
        }}
        .comparison-header.expected {{
            background: rgba(0, 255, 136, 0.2);
            color: #00ff88;
        }}
        .expected-sparql {{
            color: #86efac !important;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>{experiment_name}</h1>
            <div class="status">
                <div class="status-dot"></div>
                <span id="statusText">Loading...</span>
            </div>
        </header>
        <div id="dashboard"><p style="text-align: center; color: #888;">Loading...</p></div>
        <div class="update-time" id="updateTime"></div>
    </div>
    <script>
        const REFRESH_INTERVAL = 3000;
        let charts = {{}};
        let currentFilter = 'all';
        let tracesData = [];

        function fmt(v, std) {{
            if (v == null) return '-';
            let s = (v * 100).toFixed(1) + '%';
            if (std > 0) s += ' <span class="metric-std">(+/-' + (std * 100).toFixed(1) + '%)</span>';
            return s;
        }}
        function fmtPct(v) {{
            if (v == null) return '-';
            return (v * 100).toFixed(1) + '%';
        }}
        function clr(v) {{ return v >= 0.7 ? 'good' : v >= 0.4 ? 'medium' : 'bad'; }}
        function fmtTime(ms) {{
            if (!ms) return '-';
            return ms < 1000 ? ms.toFixed(0)+'ms' : ms < 60000 ? (ms/1000).toFixed(1)+'s' : (ms/60000).toFixed(1)+'min';
        }}

        function renderCharts(summary, traces) {{
            // F1 by Configuration Bar Chart
            const configCtx = document.getElementById('configChart');
            if (configCtx) {{
                const bc = summary.by_config || {{}};
                const labels = Object.keys(bc);
                const f1Data = labels.map(k => (bc[k].f1_score?.mean || 0) * 100);
                const execData = labels.map(k => (bc[k].execution_accuracy?.mean || 0) * 100);

                if (charts.config) charts.config.destroy();
                charts.config = new Chart(configCtx, {{
                    type: 'bar',
                    data: {{
                        labels: labels,
                        datasets: [
                            {{ label: 'F1-Score', data: f1Data, backgroundColor: 'rgba(0, 212, 255, 0.7)' }},
                            {{ label: 'Exec Accuracy', data: execData, backgroundColor: 'rgba(123, 44, 191, 0.7)' }}
                        ]
                    }},
                    options: {{
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {{ legend: {{ labels: {{ color: '#888' }} }} }},
                        scales: {{
                            y: {{ beginAtZero: true, max: 100, ticks: {{ color: '#666' }}, grid: {{ color: 'rgba(255,255,255,0.05)' }} }},
                            x: {{ ticks: {{ color: '#888' }}, grid: {{ display: false }} }}
                        }}
                    }}
                }});
            }}

            // F1 Distribution Chart
            const distCtx = document.getElementById('distributionChart');
            if (distCtx && traces.length > 0) {{
                const f1Values = traces.map(t => (t.result_metrics?.f1_score || 0) * 100);
                const bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100];
                const histogram = new Array(bins.length - 1).fill(0);
                f1Values.forEach(v => {{
                    for (let i = 0; i < bins.length - 1; i++) {{
                        if (v >= bins[i] && v < bins[i + 1]) {{ histogram[i]++; break; }}
                        if (i === bins.length - 2 && v === 100) {{ histogram[i]++; break; }}
                    }}
                }});

                if (charts.dist) charts.dist.destroy();
                charts.dist = new Chart(distCtx, {{
                    type: 'bar',
                    data: {{
                        labels: bins.slice(0, -1).map((b, i) => b + '-' + bins[i + 1] + '%'),
                        datasets: [{{ label: 'Queries', data: histogram, backgroundColor: 'rgba(0, 255, 136, 0.6)' }}]
                    }},
                    options: {{
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {{ legend: {{ display: false }} }},
                        scales: {{
                            y: {{ beginAtZero: true, ticks: {{ color: '#666' }}, grid: {{ color: 'rgba(255,255,255,0.05)' }} }},
                            x: {{ ticks: {{ color: '#888', font: {{ size: 10 }} }}, grid: {{ display: false }} }}
                        }}
                    }}
                }});
            }}
        }}

        const gtCache = {{}};  // Cache for ground truth results

        async function loadGroundTruth(queryId, containerId) {{
            const container = document.getElementById(containerId);
            if (!container) return;

            // Check cache first
            if (gtCache[queryId]) {{
                const cached = gtCache[queryId];
                container.innerHTML = renderResultsTable(cached.bindings, cached.totalCount);
                return;
            }}

            try {{
                const res = await fetch('/api/ground-truth/' + queryId);
                const data = await res.json();
                const results = data.bindings || data.results?.bindings || [];
                const totalCount = data.total_count || results.length;
                gtCache[queryId] = {{ bindings: results, totalCount: totalCount }};
                container.innerHTML = renderResultsTable(results, totalCount);
            }} catch (e) {{
                container.innerHTML = '<div class="no-results">Could not load ground truth</div>';
            }}
        }}

        function toggleQuery(id) {{
            const card = document.getElementById('query-' + id);
            if (card) {{
                const wasExpanded = card.classList.contains('expanded');
                card.classList.toggle('expanded');

                // Load ground truth when expanding
                if (!wasExpanded) {{
                    const gtContainers = card.querySelectorAll('.gt-container');
                    gtContainers.forEach(c => {{
                        const queryId = c.dataset.queryId;
                        if (queryId && c.querySelector('.loading-gt')) {{
                            loadGroundTruth(queryId, c.id);
                        }}
                    }});
                }}
            }}
        }}

        function setFilter(filter) {{
            currentFilter = filter;
            document.querySelectorAll('.filter-btn').forEach(btn => {{
                btn.classList.toggle('active', btn.dataset.filter === filter);
            }});
            renderQueryList(tracesData);
        }}

        function renderQueryList(traces) {{
            tracesData = traces;
            let filtered = traces;
            // Support both TraceSummary (direct fields) and full trace (nested fields)
            const getF1 = t => t.f1_score ?? t.result_metrics?.f1_score;
            if (currentFilter === 'grep') filtered = traces.filter(t => t.approach?.includes('grep'));
            else if (currentFilter === 'semantic') filtered = traces.filter(t => t.approach?.includes('semantic'));
            else if (currentFilter === 'success') filtered = traces.filter(t => getF1(t) > 0);
            else if (currentFilter === 'failed') filtered = traces.filter(t => !getF1(t));

            const container = document.getElementById('queryList');
            if (!container) return;

            let html = '';
            // Support both TraceSummary (query_id) and full trace (query_metadata.query_id)
            const getQid = t => t.query_id ?? t.query_metadata?.query_id ?? 'Unknown';
            filtered.sort((a, b) => getQid(a).localeCompare(getQid(b)));

            for (const t of filtered) {{
                // Support both TraceSummary and full trace formats
                const qid = t.query_id ?? t.query_metadata?.query_id ?? 'Unknown';
                const approach = t.approach?.includes('grep') ? 'grep' : 'semantic';
                const f1 = t.f1_score ?? t.result_metrics?.f1_score;
                const prec = t.precision ?? t.result_metrics?.precision;
                const rec = t.recall ?? t.result_metrics?.recall;
                const execAcc = t.execution_accuracy ?? t.result_metrics?.execution_accuracy;
                const truePos = t.true_positives ?? t.result_metrics?.true_positives ?? 0;
                const actualCount = t.actual_result_count ?? t.result_metrics?.actual_result_count ?? 0;
                const nlQuery = t.query_text ?? t.query_metadata?.query_text ?? 'N/A';
                const sparql = t.final_sparql || 'No query generated';
                const results = t.results_sample ?? t.final_results ?? [];
                const expectedSparql = t.expected_sparql ?? t.query_metadata?.expected_sparql ?? 'N/A';
                const uid = qid + '_' + approach + '_' + (t.run_id || '1');

                html += `
                <div class="query-card" id="query-${{uid}}">
                    <div class="query-header" onclick="toggleQuery('${{uid}}')">
                        <div class="query-header-left">
                            <span class="query-id">${{qid}}</span>
                            <span class="query-approach ${{approach}}">${{approach}}</span>
                        </div>
                        <div class="query-metrics">
                            <div class="query-metric">
                                <span class="query-metric-label">F1:</span>
                                <span class="query-metric-value ${{clr(f1)}}">${{fmtPct(f1)}}</span>
                            </div>
                            <div class="query-metric">
                                <span class="query-metric-label">P:</span>
                                <span class="query-metric-value">${{fmtPct(prec)}}</span>
                            </div>
                            <div class="query-metric">
                                <span class="query-metric-label">R:</span>
                                <span class="query-metric-value">${{fmtPct(rec)}}</span>
                            </div>
                            <span class="expand-icon">&#9660;</span>
                        </div>
                    </div>
                    <div class="query-details">
                        <div class="detail-section">
                            <div class="detail-label">Natural Language Query</div>
                            <div class="nl-query">${{nlQuery}}</div>
                        </div>
                        <div class="detail-section">
                            <div class="detail-label">Metrics</div>
                            <div class="metrics-detail-grid">
                                <div class="metrics-detail-item">
                                    <div class="metrics-detail-value ${{clr(f1)}}">${{fmtPct(f1)}}</div>
                                    <div class="metrics-detail-label">F1-Score</div>
                                </div>
                                <div class="metrics-detail-item">
                                    <div class="metrics-detail-value">${{fmtPct(prec)}}</div>
                                    <div class="metrics-detail-label">Precision</div>
                                </div>
                                <div class="metrics-detail-item">
                                    <div class="metrics-detail-value">${{fmtPct(rec)}}</div>
                                    <div class="metrics-detail-label">Recall</div>
                                </div>
                                <div class="metrics-detail-item">
                                    <div class="metrics-detail-value ${{clr(execAcc)}}">${{fmtPct(execAcc)}}</div>
                                    <div class="metrics-detail-label">Exec Accuracy</div>
                                </div>
                                <div class="metrics-detail-item">
                                    <div class="metrics-detail-value">${{truePos}}</div>
                                    <div class="metrics-detail-label">True Positives</div>
                                </div>
                                <div class="metrics-detail-item">
                                    <div class="metrics-detail-value">${{actualCount}}</div>
                                    <div class="metrics-detail-label">Results</div>
                                </div>
                            </div>
                        </div>
                        <div class="comparison-grid">
                            <div class="comparison-col">
                                <div class="comparison-header generated">Generated</div>
                                <div class="detail-section">
                                    <div class="detail-label">Generated SPARQL</div>
                                    <div class="sparql-query">${{sparql.replace(/</g, '&lt;').replace(/>/g, '&gt;')}}</div>
                                </div>
                                <div class="detail-section">
                                    <div class="detail-label">Generated Results (first 5 rows)</div>
                                    ${{renderResultsTable(results)}}
                                </div>
                            </div>
                            <div class="comparison-col">
                                <div class="comparison-header expected">Ground Truth</div>
                                <div class="detail-section">
                                    <div class="detail-label">Expected SPARQL</div>
                                    <div class="sparql-query expected-sparql">${{expectedSparql.replace(/</g, '&lt;').replace(/>/g, '&gt;')}}</div>
                                </div>
                                <div class="detail-section">
                                    <div class="detail-label">Expected Results (first 5 rows)</div>
                                    <div id="gt-${{uid}}" class="gt-container" data-query-id="${{qid}}">
                                        <div class="loading-gt">Loading ground truth...</div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>`;
            }}
            container.innerHTML = html || '<p style="color:#666;text-align:center;">No queries completed yet</p>';
        }}

        function renderResultsTable(results, totalCount) {{
            if (!results || results.length === 0) {{
                return '<div class="no-results">No results returned</div>';
            }}
            const rows = results.slice(0, 5);
            const total = totalCount || results.length;
            const keys = [...new Set(rows.flatMap(r => Object.keys(r)))].filter(k => !k.startsWith('_'));
            if (keys.length === 0) return '<div class="no-results">No data columns</div>';

            let html = '<table class="results-table"><thead><tr>';
            keys.forEach(k => html += '<th>' + k + '</th>');
            html += '</tr></thead><tbody>';
            rows.forEach(row => {{
                html += '<tr>';
                keys.forEach(k => {{
                    let val = row[k];
                    if (typeof val === 'object' && val !== null) val = val.value || JSON.stringify(val);
                    html += '<td title="' + (val || '').toString().replace(/"/g, '&quot;') + '">' + (val || '-') + '</td>';
                }});
                html += '</tr>';
            }});
            html += '</tbody></table>';
            if (total > 5) html += '<p style="color:#666;font-size:0.8rem;margin-top:8px;">...and ' + (total - 5) + ' more rows</p>';
            return html;
        }}

        function render(summary, traces) {{
            const o = summary.overall || {{}}, c = summary.config || {{}}, bc = summary.by_config || {{}};
            const total = c.total_runs || 0;
            const expected = (c.total_queries||0) * (c.models?.length||1) * (c.approaches?.length||1) * (c.runs_per_query||1);
            const pct = expected > 0 ? total / expected : 0;

            let h = `
            <div class="grid">
                <div class="card">
                    <h2>Progress</h2>
                    <div class="metric">
                        <span class="metric-label">Completed</span>
                        <span class="metric-value">${{total}} / ${{expected}}</span>
                    </div>
                    <div class="progress-bar"><div class="progress-fill" style="width:${{(pct*100).toFixed(1)}}%"></div></div>
                    <div class="progress-text"><span>${{(pct*100).toFixed(1)}}%</span><span>${{expected-total}} remaining</span></div>
                </div>
                <div class="card">
                    <h2>Overall Metrics</h2>
                    <div class="metric">
                        <span class="metric-label">F1-Score</span>
                        <span class="metric-value ${{clr(o.f1_score?.mean)}}">${{fmt(o.f1_score?.mean, o.f1_score?.std)}}</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Exec Accuracy</span>
                        <span class="metric-value ${{clr(o.execution_accuracy?.mean)}}">${{fmt(o.execution_accuracy?.mean, o.execution_accuracy?.std)}}</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Success Rate</span>
                        <span class="metric-value ${{clr(o.success_rate)}}">${{fmt(o.success_rate)}}</span>
                    </div>
                </div>
                <div class="card">
                    <h2>Config</h2>
                    <div class="config-grid">
                        <div class="config-item"><div class="config-label">Models</div><div class="config-value">${{c.models?.join(', ')||'-'}}</div></div>
                        <div class="config-item"><div class="config-label">Approaches</div><div class="config-value">${{c.approaches?.join(', ')||'-'}}</div></div>
                        <div class="config-item"><div class="config-label">Queries</div><div class="config-value">${{c.total_queries||0}}</div></div>
                        <div class="config-item"><div class="config-label">Runs/Query</div><div class="config-value">${{c.runs_per_query||0}}</div></div>
                    </div>
                </div>
            </div>

            <div class="grid-2">
                <div class="card">
                    <h2>Results by Configuration</h2>
                    <div class="chart-container"><canvas id="configChart"></canvas></div>
                </div>
                <div class="card">
                    <h2>F1-Score Distribution</h2>
                    <div class="chart-container"><canvas id="distributionChart"></canvas></div>
                </div>
            </div>

            <div class="card table-card" style="margin-bottom:30px;">
                <h2>Detailed Results by Configuration</h2>
                <table>
                    <thead><tr><th>Config</th><th>F1-Score</th><th>Precision</th><th>Recall</th><th>Exec Acc</th><th>Success</th><th>Avg Time</th><th>Runs</th></tr></thead>
                    <tbody>`;
            for (const [k, m] of Object.entries(bc)) {{
                // Calculate precision/recall from traces
                const configTraces = traces.filter(t => {{
                    const isGrep = k.includes('grep') && t.approach?.includes('grep');
                    const isSem = k.includes('semantic') && t.approach?.includes('semantic');
                    return isGrep || isSem;
                }});
                const avgPrec = configTraces.length > 0 ? configTraces.reduce((s,t) => s + (t.result_metrics?.precision||0), 0) / configTraces.length : 0;
                const avgRec = configTraces.length > 0 ? configTraces.reduce((s,t) => s + (t.result_metrics?.recall||0), 0) / configTraces.length : 0;

                h += `<tr>
                    <td><strong>${{k}}</strong></td>
                    <td class="${{clr(m.f1_score?.mean)}}">${{fmt(m.f1_score?.mean, m.f1_score?.std)}}</td>
                    <td>${{fmtPct(avgPrec)}}</td>
                    <td>${{fmtPct(avgRec)}}</td>
                    <td class="${{clr(m.execution_accuracy?.mean)}}">${{fmt(m.execution_accuracy?.mean, m.execution_accuracy?.std)}}</td>
                    <td>${{fmt(m.success_rate)}}</td>
                    <td>${{fmtTime(m.avg_time_ms)}}</td>
                    <td>${{m.total_runs||0}}</td>
                </tr>`;
            }}
            h += '</tbody></table></div>';

            h += `
            <h3 class="section-title">Individual Query Results</h3>
            <div class="filter-bar">
                <button class="filter-btn ${{currentFilter==='all'?'active':''}}" data-filter="all" onclick="setFilter('all')">All</button>
                <button class="filter-btn ${{currentFilter==='grep'?'active':''}}" data-filter="grep" onclick="setFilter('grep')">Grep</button>
                <button class="filter-btn ${{currentFilter==='semantic'?'active':''}}" data-filter="semantic" onclick="setFilter('semantic')">Semantic</button>
                <button class="filter-btn ${{currentFilter==='success'?'active':''}}" data-filter="success" onclick="setFilter('success')">F1 > 0</button>
                <button class="filter-btn ${{currentFilter==='failed'?'active':''}}" data-filter="failed" onclick="setFilter('failed')">F1 = 0</button>
            </div>
            <div id="queryList"></div>`;

            document.getElementById('dashboard').innerHTML = h;

            // Render charts after DOM update
            setTimeout(() => {{
                renderCharts(summary, traces);
                renderQueryList(traces);
            }}, 50);

            const done = total >= expected && expected > 0;
            document.getElementById('statusText').textContent = done ? 'Completed' : 'Running - press F5 to refresh';
            document.querySelector('.status-dot').style.background = done ? '#888' : '#00ff88';
            document.querySelector('.status-dot').style.animation = done ? 'none' : 'pulse 2s infinite';
            document.getElementById('updateTime').textContent = 'Updated: ' + new Date().toLocaleTimeString();
        }}

        async function fetch_data() {{
            try {{
                const [summaryRes, tracesRes] = await Promise.all([
                    fetch('/api/summary'),
                    fetch('/api/traces')
                ]);
                if (!summaryRes.ok) throw new Error('Failed to fetch summary');
                const summary = await summaryRes.json();
                const traces = tracesRes.ok ? await tracesRes.json() : [];
                render(summary, traces);
            }} catch(e) {{
                document.getElementById('dashboard').innerHTML = '<div class="error-message">Waiting for experiment data...</div>';
            }}
        }}
        fetch_data();
        // Manual refresh only - no auto-refresh
    </script>
</body>
</html>'''


class DashboardHandler(http.server.BaseHTTPRequestHandler):
    """HTTP handler for the live adaptive evaluation dashboard."""

    experiment_dir: Path = None
    experiment_name: str = None

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            # Serve the new tiered/adaptive dashboard
            self.send_response(200)
            self.send_header("Content-type", "text/html; charset=utf-8")
            self.end_headers()
            html = get_tiered_dashboard_html(self.experiment_name)
            self.wfile.write(html.encode("utf-8"))
        elif self.path.startswith("/api/tiered_evaluation"):
            # Serve adaptive evaluation data for tiered dashboard
            eval_file = self.experiment_dir / "adaptive_evaluation.json"
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            if eval_file.exists():
                with open(eval_file, "r", encoding="utf-8") as f:
                    self.wfile.write(f.read().encode())
            else:
                empty = {
                    "experiment_name": self.experiment_name,
                    "evaluation_mode": "adaptive",
                    "traces": {},
                    "summary": {"overall": {}, "by_approach": {}, "by_query_set": {}},
                }
                self.wfile.write(json.dumps(empty).encode())
        elif self.path == "/api/summary":
            summary_file = self.experiment_dir / f"{self.experiment_name}_summary.json"
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            if summary_file.exists():
                with open(summary_file, "r", encoding="utf-8") as f:
                    self.wfile.write(f.read().encode())
            else:
                empty = {"experiment_name": self.experiment_name, "config": {}, "overall": {}, "by_config": {}}
                self.wfile.write(json.dumps(empty).encode())
        elif self.path == "/api/traces":
            # Load lightweight trace summaries (~5 MB instead of ~2 GB)
            summaries_file = self.experiment_dir / "trace_summaries.json"
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()

            if summaries_file.exists():
                # Fast path: use pre-built summaries index
                with open(summaries_file, "r", encoding="utf-8") as f:
                    self.wfile.write(f.read().encode())
            else:
                # Fallback: build summaries from full traces (slower, for legacy data)
                traces_dir = self.experiment_dir / "traces"
                summaries = []
                if traces_dir.exists():
                    for trace_file in sorted(traces_dir.glob("*.json")):
                        try:
                            with open(trace_file, "r", encoding="utf-8") as f:
                                trace_data = json.load(f)
                                query_meta = trace_data.get("query_metadata", {})
                                result_metrics = trace_data.get("result_metrics") or {}
                                final_results = trace_data.get("final_results", [])
                                # Extract summary fields including preview data
                                summaries.append({
                                    "run_id": trace_data.get("run_id", ""),
                                    "query_id": query_meta.get("query_id", ""),
                                    "approach": trace_data.get("approach", ""),
                                    "llm_model": trace_data.get("llm_model", ""),
                                    "run_number": trace_data.get("run_number", 1),
                                    # Query info
                                    "query_text": query_meta.get("query_text", ""),
                                    "expected_sparql": query_meta.get("expected_sparql", ""),
                                    # Metrics
                                    "f1_score": result_metrics.get("f1_score"),
                                    "precision": result_metrics.get("precision"),
                                    "recall": result_metrics.get("recall"),
                                    "execution_accuracy": result_metrics.get("execution_accuracy"),
                                    "true_positives": result_metrics.get("true_positives", 0),
                                    "actual_result_count": result_metrics.get("actual_result_count", 0),
                                    "expected_result_count": result_metrics.get("expected_result_count", 0),
                                    # Timing
                                    "total_time_ms": trace_data.get("timing", {}).get("total_time_ms", 0.0),
                                    "total_tokens": trace_data.get("token_usage", {}).get("total_tokens", 0),
                                    # Status
                                    "success": trace_data.get("success", False),
                                    "is_unanswerable": trace_data.get("is_unanswerable", False),
                                    # Output preview (first 5 results)
                                    "final_sparql": trace_data.get("final_sparql"),
                                    "results_sample": final_results[:5] if isinstance(final_results, list) else [],
                                    "trace_file": trace_file.name,
                                })
                        except Exception:
                            pass
                self.wfile.write(json.dumps(summaries).encode())

        elif self.path.startswith("/api/trace/"):
            # Load a single full trace on-demand
            trace_id = self.path.split("/")[-1]
            traces_dir = self.experiment_dir / "traces"

            # Find matching trace file
            trace_file = None
            for f in traces_dir.glob(f"*{trace_id}*.json"):
                trace_file = f
                break

            if trace_file and trace_file.exists():
                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.send_header("Cache-Control", "max-age=3600")  # Cache for 1 hour
                self.end_headers()
                with open(trace_file, "r", encoding="utf-8") as f:
                    self.wfile.write(f.read().encode())
            else:
                self.send_error(404, f"Trace not found: {trace_id}")

        elif self.path.startswith("/api/ground-truth/"):
            # Load ground truth from lightweight preview cache (136 KB vs 1 GB)
            query_id = self.path.split("/")[-1]

            # Use class-level cache to avoid reloading on every request
            if not hasattr(DashboardHandler, '_gt_preview_cache'):
                if GROUND_TRUTH_PREVIEW_CACHE.exists():
                    with open(GROUND_TRUTH_PREVIEW_CACHE, "r", encoding="utf-8") as f:
                        DashboardHandler._gt_preview_cache = json.load(f)
                else:
                    DashboardHandler._gt_preview_cache = {}

            result = DashboardHandler._gt_preview_cache.get(query_id, {"bindings": [], "total_count": 0})
            result["preview_only"] = True

            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.send_header("Cache-Control", "max-age=86400")  # Cache for 24 hours
            self.end_headers()
            self.wfile.write(json.dumps(result).encode())

        else:
            self.send_error(404)

    def log_message(self, format, *args):
        pass  # Suppress HTTP logs


def start_dashboard_server(experiment_dir: Path, experiment_name: str, port: int = 8050) -> tuple[socketserver.TCPServer | None, int]:
    """Start the dashboard server in a background thread."""
    DashboardHandler.experiment_dir = experiment_dir
    DashboardHandler.experiment_name = experiment_name

    for try_port in range(port, port + 20):
        try:
            httpd = socketserver.TCPServer(("", try_port), DashboardHandler)
            httpd.allow_reuse_address = True

            thread = threading.Thread(target=httpd.serve_forever, daemon=True)
            thread.start()

            return httpd, try_port
        except OSError:
            continue

    return None, 0


# ============================================================================
# Ground Truth Execution
# ============================================================================

def execute_ground_truth(query: AdaptiveGroundTruth) -> dict | None:
    """
    Execute the ground-truth SPARQL query and return results (sync version).

    Args:
        query: AdaptiveGroundTruth with expected_sparql (sparql_query)

    Returns:
        Results dict or None if execution failed
    """
    if not query.expected_sparql:
        return None

    try:
        # Clean up the query (remove comments)
        sparql = query.expected_sparql
        lines = [
            line for line in sparql.split("\n")
            if not line.strip().startswith("#")
        ]
        sparql = "\n".join(lines)

        # Execute on the correct dataset endpoints
        result = execute_sparql.invoke({
            "query": sparql,
            "dataset_ids": query.datasets,
        })
        if result.get("success"):
            return result.get("results", {})
        else:
            logger.warning(f"Ground truth execution failed for {query.id}: {result.get('error_message')}")
            return None
    except Exception as e:
        logger.warning(f"Ground truth execution error for {query.id}: {e}")
        return None


async def execute_ground_truth_async(query: AdaptiveGroundTruth) -> dict | None:
    """
    Async version: Execute the ground-truth SPARQL query and return results.

    Args:
        query: AdaptiveGroundTruth with expected_sparql (sparql_query)

    Returns:
        Results dict or None if execution failed
    """
    if not query.expected_sparql:
        return None

    try:
        # Clean up the query (remove comments)
        sparql = query.expected_sparql
        lines = [
            line for line in sparql.split("\n")
            if not line.strip().startswith("#")
        ]
        sparql = "\n".join(lines)

        # Execute on the correct dataset endpoints using async version
        result = await execute_sparql_async(
            query=sparql,
            dataset_ids=query.datasets,
        )
        if result.get("success"):
            return result.get("results", {})
        else:
            logger.warning(f"Ground truth execution failed for {query.id}: {result.get('error_message')}")
            return None
    except Exception as e:
        logger.warning(f"Ground truth execution error for {query.id}: {e}")
        return None


# ============================================================================
# Single Run Execution
# ============================================================================

async def run_single_experiment(
    query: AdaptiveGroundTruth,
    llm_model: str,
    approach: Literal["agentic_grep", "agentic_semantic"],
    run_number: int,
) -> dict:
    """
    Run a single experiment iteration with adaptive evaluation.

    Args:
        query: The AdaptiveGroundTruth query to test (also serves as ground truth)
        llm_model: LLM model to use
        approach: Retrieval approach
        run_number: Run number (1-indexed)

    Returns:
        Dict with trace, metrics, and adaptive evaluation data
    """
    tracer = get_tracer()

    # Create collector with unique run_id for event filtering
    query_metadata = convert_use_case_query_to_metadata(query)
    collector = ExperimentTraceCollector(
        query_metadata=query_metadata,
        approach=approach,
        llm_model=llm_model,
        run_number=run_number,
    )
    tracer.add_listener(collector)

    try:
        # Use trace_run_context to associate all events with this collector's run_id
        # This prevents cross-contamination when running multiple experiments in parallel
        with trace_run_context(collector.run_id):
            # Run the orchestrator
            agent = OrchestratorAgent()
            state = await agent.run(
                user_query=query.query,
                approach=approach,
                llm_model=llm_model,
            )

        # Extract results
        final_query = state.get("final_query")
        final_results_raw = state.get("final_results")
        is_unanswerable = state.get("is_unanswerable", False)
        unanswerable_reason = state.get("unanswerable_reason", "")

        # Extract bindings from results (handle dict/list formats)
        if isinstance(final_results_raw, dict):
            # Handle nested format: {"results": {"bindings": [...]}}
            if "results" in final_results_raw and isinstance(final_results_raw["results"], dict):
                final_results = final_results_raw["results"].get("bindings", [])
            elif "bindings" in final_results_raw:
                final_results = final_results_raw["bindings"]
            else:
                final_results = [final_results_raw]  # Single result as dict
        elif isinstance(final_results_raw, list):
            final_results = final_results_raw
        else:
            final_results = []

        # Update collector with final results
        collector.final_sparql = final_query
        collector.final_results = final_results
        collector.success = final_query is not None and not is_unanswerable
        collector.is_unanswerable = is_unanswerable
        collector.unanswerable_reason = unanswerable_reason

        # Store ground truth reference (query IS the ground truth now)
        collector.ground_truth_query_id = query.id

        # Calculate adaptive metrics using evaluate_trace_adaptive (same as reevaluate_tiered.py)
        result_metrics = None
        adaptive_eval_data = None
        # Use get_ontop_endpoint() which respects dataset_size context (small/large)
        endpoint = get_ontop_endpoint(query.dataset)
        if endpoint:
            try:
                # Create trace dict in the format expected by evaluate_trace_adaptive
                trace_dict = {
                    "final_results": final_results,
                    "final_sparql": final_query,  # CRITICAL: Include SPARQL for signature-based column matching
                }

                # Use the same evaluation function as reevaluate_tiered.py
                # query is AdaptiveGroundTruth, so it serves as both query and ground truth
                adaptive_eval_data = evaluate_trace_adaptive(
                    trace_dict, query, endpoint
                )

                # Extract metrics for ResultMetrics
                adaptive_metrics = adaptive_eval_data.get("adaptive_metrics", {})
                result_metrics = ResultMetrics(
                    execution_accuracy=1.0 if adaptive_metrics.get("best_f1", 0) == 1.0 else 0.0,
                    f1_score=adaptive_metrics.get("best_f1", 0.0),
                    precision=adaptive_metrics.get("best_precision", 0.0),
                    recall=adaptive_metrics.get("best_recall", 0.0),
                    true_positives=0,  # Not directly available in adaptive metrics
                    actual_result_count=adaptive_metrics.get("llm_row_count", 0),
                    expected_result_count=adaptive_metrics.get("best_recall_gt_size", 0),
                )

            except Exception as e:
                logger.warning(f"Adaptive evaluation failed for {query.id}: {e}")

        # Retrieval evaluation (schema-aware metrics)
        retrieval_metrics_dict = None
        try:
            schema_gt = build_schema_gt_for_query(query)
            if schema_gt.triples:
                turtle_strings = list(collector.all_retrieved_triples) if collector.all_retrieved_triples else []
                dataset_ids = query.datasets or [query.dataset]
                schema = get_combined_schema(dataset_ids)
                retrieval_result = calculate_retrieval_metrics(
                    turtle_strings, schema, schema_gt,
                )
                retrieval_metrics_dict = retrieval_result.to_dict()
        except Exception as e:
            logger.debug(f"Retrieval evaluation failed for {query.id}: {e}")

        # Inject retrieval metrics into adaptive_eval_data
        if adaptive_eval_data is not None and retrieval_metrics_dict is not None:
            adaptive_eval_data["retrieval_metrics"] = retrieval_metrics_dict
            # Also inject summary values into adaptive_metrics for aggregation
            am = adaptive_eval_data.get("adaptive_metrics", {})
            am["retrieval_path_coherence"] = retrieval_metrics_dict.get("schema_path_coherence", 0)
            am["retrieval_triple_f1"] = retrieval_metrics_dict.get("schema_triple_f1", 0)
            am["retrieval_triple_precision"] = retrieval_metrics_dict.get("schema_triple_precision", 0)
            am["retrieval_triple_recall"] = retrieval_metrics_dict.get("schema_triple_recall", 0)

        # Finalize trace
        trace = collector.finalize(result_metrics=result_metrics)

        return {
            "trace": trace,
            "success": collector.success,
            "metrics": result_metrics,
            "adaptive_eval": adaptive_eval_data,
        }

    except Exception as e:
        logger.error(f"Run failed for {query.id}: {e}")
        collector.error_message = str(e)
        trace = collector.finalize()
        return {
            "trace": trace,
            "success": False,
            "metrics": None,
            "adaptive_eval": None,
            "error": str(e),
        }

    finally:
        tracer.remove_listener(collector)


# ============================================================================
# Experiment Runner
# ============================================================================

async def run_experiment(
    queries: list[AdaptiveGroundTruth],
    models: list[str],
    approaches: list[str],
    runs_per_query: int,
    output_dir: Path,
    experiment_name: str,
    resume: bool = False,
    concurrency: int = 5,
) -> dict:
    """
    Run the full experiment with checkpoint/resume capability and parallel execution.

    Results are saved incrementally after each run, allowing you to stop
    and resume the experiment at any time.

    Args:
        queries: List of queries to test
        models: List of LLM model keys (deepseek, claude, gpt4)
        approaches: List of approach keys (grep, semantic)
        runs_per_query: Number of runs per query/model/approach combination
        output_dir: Directory to save results
        experiment_name: Name of the experiment
        resume: Whether to resume from a previous checkpoint
        concurrency: Number of parallel runs (default: 5)

    Returns:
        Summary dict with all results
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load checkpoint if resuming
    checkpoint = load_checkpoint(output_dir) if resume else {
        "completed_runs": set(),
        "trace_summaries": [],
    }
    completed_runs = checkpoint["completed_runs"]
    # Use lightweight summaries instead of full traces (saves ~95% RAM)
    trace_summaries: list[TraceSummary] = checkpoint["trace_summaries"]

    # Locks for thread-safe access to shared state
    completed_runs_lock = asyncio.Lock()
    summaries_lock = asyncio.Lock()
    adaptive_eval_lock = asyncio.Lock()
    checkpoint_counter = 0
    checkpoint_counter_lock = asyncio.Lock()
    CHECKPOINT_BATCH_SIZE = 5

    # Storage for adaptive evaluation results (for dashboard)
    # Load existing traces if resuming
    adaptive_eval_traces: dict[str, dict] = {}
    if resume:
        adaptive_eval_file = output_dir / "adaptive_evaluation.json"
        if adaptive_eval_file.exists():
            try:
                with open(adaptive_eval_file, "r", encoding="utf-8") as f:
                    existing_data = json.load(f)
                    adaptive_eval_traces = existing_data.get("traces", {})
                    logger.info(f"Loaded {len(adaptive_eval_traces)} existing adaptive evaluation traces")
            except Exception as e:
                logger.warning(f"Could not load existing adaptive evaluation: {e}")

    # Queries are now AdaptiveGroundTruth objects - they ARE the ground truth
    logger.info(f"Loaded {len(queries)} AdaptiveGroundTruth queries")

    save_checkpoint(output_dir, completed_runs)

    # Calculate total and remaining iterations
    total_iterations = len(queries) * len(models) * len(approaches) * runs_per_query
    remaining_iterations = total_iterations - len(completed_runs)

    if remaining_iterations == 0:
        logger.info("All runs already completed!")
        return update_summary(output_dir, experiment_name, trace_summaries,
                              queries, models, approaches, runs_per_query)

    logger.info(
        f"Experiment: {len(queries)} queries x {len(models)} models x "
        f"{len(approaches)} approaches x {runs_per_query} runs = {total_iterations} total"
    )
    logger.info(f"Running with concurrency={concurrency}")
    if len(completed_runs) > 0:
        logger.info(f"Resuming: {len(completed_runs)} already done, {remaining_iterations} remaining")

    # Progress bar
    pbar = tqdm(total=total_iterations, initial=len(completed_runs), desc="Running experiments")

    # Build list of pending tasks
    pending_tasks: list[tuple[str, dict]] = []
    for query in queries:
        for model_key in models:
            llm_model = LLM_MODELS.get(model_key, model_key)
            for approach_key in approaches:
                approach = APPROACH_TYPES.get(approach_key, approach_key)
                for run_num in range(1, runs_per_query + 1):
                    run_key = get_run_key(query.id, model_key, approach_key, run_num)
                    if run_key not in completed_runs:
                        pending_tasks.append((run_key, {
                            "query": query,  # AdaptiveGroundTruth serves as both query and ground truth
                            "llm_model": llm_model,
                            "approach": approach,
                            "run_number": run_num,
                            "model_key": model_key,
                            "approach_key": approach_key,
                        }))

    # Semaphore for controlling concurrency
    run_semaphore = asyncio.Semaphore(concurrency)

    # Automatically enable slot-based container isolation when concurrency > 1
    # This prevents Mapping Optimizer restarts from affecting other concurrent queries
    slot_manager = None
    if concurrency > 1:
        from src.config import MAPPINGS_DIR_ROOT
        import src.config as config_module

        # Check if slot infrastructure exists
        slot_mappings_dir = MAPPINGS_DIR_ROOT.parent / "mappings-slots"
        if not slot_mappings_dir.exists():
            raise RuntimeError(
                f"\n{'='*70}\n"
                f"SLOT INFRASTRUCTURE NOT FOUND\n"
                f"{'='*70}\n"
                f"Running with concurrency={concurrency} requires slot-based containers.\n\n"
                f"Please run these commands first:\n\n"
                f"  1. Initialize slot mappings:\n"
                f"     python scripts/initialize_slot_mappings.py --slots {concurrency}\n\n"
                f"  2. Start slot containers:\n"
                f"     docker compose -f docker-compose.slots.yml up -d\n\n"
                f"Or use --concurrency 1 to run without slots.\n"
                f"{'='*70}"
            )

        # Check if we have enough slots
        available_slots = len([d for d in slot_mappings_dir.iterdir() if d.is_dir() and d.name.startswith("slot")])
        if available_slots < concurrency:
            raise RuntimeError(
                f"\n{'='*70}\n"
                f"NOT ENOUGH SLOTS\n"
                f"{'='*70}\n"
                f"Requested concurrency={concurrency} but only {available_slots} slots found.\n\n"
                f"Please run:\n"
                f"  python scripts/initialize_slot_mappings.py --slots {concurrency} --clean\n"
                f"  docker compose -f docker-compose.slots.yml up -d\n"
                f"{'='*70}"
            )

        # Enable slot containers
        config_module.USE_SLOT_CONTAINERS = True
        slot_manager = get_slot_manager(concurrency)
        await slot_manager.initialize()
        logger.info(f"Slot-based container isolation enabled (concurrency={concurrency}, slots={available_slots})")

    async def run_with_semaphore(run_key: str, params: dict) -> tuple[str, dict | Exception]:
        """Execute a single run with semaphore-controlled concurrency and slot isolation."""
        nonlocal checkpoint_counter

        async with run_semaphore:
            # Acquire exclusive slot for container isolation (if enabled)
            if slot_manager is not None:
                async with slot_manager.acquire_slot() as slot_id:
                    return await _execute_run(run_key, params, slot_id)
            else:
                return await _execute_run(run_key, params, slot_id=None)

    async def _execute_run(run_key: str, params: dict, slot_id: int | None) -> tuple[str, dict | Exception]:
        """Execute a single run with optional slot isolation."""
        nonlocal checkpoint_counter

        try:
            # Auto-select dataset size based on query set
            query = params["query"]
            size = get_size_for_query_set(query.query_set)

            # Use both dataset_size_context and slot_context (if slot_id is set)
            if slot_id is not None:
                with dataset_size_context(size), slot_context(slot_id):
                    result = await run_single_experiment(
                        query=query,
                        llm_model=params["llm_model"],
                        approach=params["approach"],
                        run_number=params["run_number"],
                    )
            else:
                with dataset_size_context(size):
                    result = await run_single_experiment(
                        query=query,
                        llm_model=params["llm_model"],
                        approach=params["approach"],
                        run_number=params["run_number"],
                    )

            trace = result["trace"]

            # Save trace immediately and get lightweight summary
            summary = save_trace(output_dir, trace)

            # Thread-safe: add summary (NOT full trace) and mark completed
            async with summaries_lock:
                trace_summaries.append(summary)

            async with completed_runs_lock:
                completed_runs.add(run_key)

            # Store adaptive evaluation data for dashboard
            # Use exact same structure as reevaluate_tiered.py (lines 476-485)
            if result.get("adaptive_eval"):
                # query already defined above from params["query"]
                async with adaptive_eval_lock:
                    # Determine baseline_id for paired query grouping in dashboard
                    baseline_id = None
                    if query.query_set in ("SYN", "TYPO"):
                        base_query = get_baseline_for_variant(query.query_id)
                        baseline_id = base_query.query_id if base_query else None
                    elif query.query_set == "BASE":
                        baseline_id = query.query_id

                    # Base metadata + all fields from evaluate_trace_adaptive
                    adaptive_eval_traces[run_key] = {
                        "query_id": query.id,
                        "approach": params["approach"],
                        "query_set": query.query_set,
                        "dataset": query.dataset,
                        "query_text": query.query,
                        "final_sparql": trace.final_sparql,
                        "success": result["success"],
                        "baseline_id": baseline_id,  # For paired query grouping
                        **result["adaptive_eval"],  # Include all fields from evaluate_trace_adaptive
                    }

            # Trace object can now be garbage collected - only summary stays in RAM
            del trace

            # Batch checkpoint and summary writes
            async with checkpoint_counter_lock:
                checkpoint_counter += 1
                if checkpoint_counter % CHECKPOINT_BATCH_SIZE == 0:
                    save_checkpoint(output_dir, completed_runs)
                    # Save summaries index for dashboard
                    save_trace_summaries(output_dir, trace_summaries)
                    # Update summary for dashboard
                    update_summary(output_dir, experiment_name, trace_summaries,
                                   queries, models, approaches, runs_per_query)
                    # Save adaptive evaluation for tiered dashboard
                    save_adaptive_evaluation(output_dir, experiment_name,
                                             adaptive_eval_traces, models, approaches,
                                             expected_total=total_iterations)

            # Log progress
            status = "OK" if result["success"] else "FAIL"
            metrics_str = ""
            if result.get("metrics"):
                m = result["metrics"]
                metrics_str = f" F1={m.f1_score:.3f}"

            logger.info(
                f"[{status}] {params['query'].id} | {params['model_key']}/{params['approach_key']} | "
                f"Run {params['run_number']}{metrics_str}"
            )

            pbar.update(1)
            return (run_key, result)

        except Exception as e:
            logger.error(f"Error in {run_key}: {e}")
            pbar.update(1)
            return (run_key, e)

    # Execute all pending tasks in parallel with semaphore control
    try:
        tasks = [run_with_semaphore(run_key, params) for run_key, params in pending_tasks]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle any top-level exceptions from gather
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Task failed with exception: {result}")

    except KeyboardInterrupt:
        logger.warning("\nInterrupted! Progress has been saved. Resume with --resume flag.")
        pbar.close()
        # Save final checkpoint and summary before exit
        save_checkpoint(output_dir, completed_runs)
        save_trace_summaries(output_dir, trace_summaries)
        save_adaptive_evaluation(output_dir, experiment_name, adaptive_eval_traces, models, approaches,
                                 expected_total=total_iterations)
        summary = update_summary(output_dir, experiment_name, trace_summaries,
                                 queries, models, approaches, runs_per_query)
        return summary

    pbar.close()

    # Final checkpoint save
    save_checkpoint(output_dir, completed_runs)
    save_trace_summaries(output_dir, trace_summaries)

    # Save adaptive evaluation for tiered dashboard
    save_adaptive_evaluation(output_dir, experiment_name, adaptive_eval_traces, models, approaches,
                             expected_total=total_iterations)

    # Create final summary
    summary = update_summary(output_dir, experiment_name, trace_summaries,
                             queries, models, approaches, runs_per_query)

    logger.info(f"Results saved to {output_dir}")
    logger.info(f"Completed {len(completed_runs)} / {total_iterations} runs")

    return summary


def _get_metric(trace, metric_name: str) -> float | None:
    """
    Extract a metric from either TraceSummary or ExperimentRunTrace.

    TraceSummary has metrics directly (f1_score, execution_accuracy, etc.)
    ExperimentRunTrace has them nested in result_metrics.
    """
    # TraceSummary: direct attributes
    if hasattr(trace, metric_name) and not hasattr(trace, 'result_metrics'):
        return getattr(trace, metric_name, None)

    # ExperimentRunTrace: nested in result_metrics
    if hasattr(trace, 'result_metrics') and trace.result_metrics is not None:
        return getattr(trace.result_metrics, metric_name, None)

    return None


def _get_tokens(trace) -> int:
    """Extract total tokens from either TraceSummary or ExperimentRunTrace."""
    # TraceSummary: direct attribute
    if hasattr(trace, 'total_tokens') and not hasattr(trace, 'token_usage'):
        return trace.total_tokens

    # ExperimentRunTrace: nested in token_usage
    if hasattr(trace, 'token_usage'):
        return trace.token_usage.total_tokens

    return 0


def _get_time_ms(trace) -> float:
    """Extract total time from either TraceSummary or ExperimentRunTrace."""
    # TraceSummary: direct attribute
    if hasattr(trace, 'total_time_ms') and not hasattr(trace, 'timing'):
        return trace.total_time_ms

    # ExperimentRunTrace: nested in timing
    if hasattr(trace, 'timing'):
        return trace.timing.total_time_ms

    return 0.0


def create_experiment_summary(
    all_traces: list,
    results_by_config: dict[str, list],
    queries: list[AdaptiveGroundTruth],
    models: list[str],
    approaches: list[str],
    runs_per_query: int,
    experiment_name: str,
) -> dict:
    """
    Create experiment summary with aggregated metrics.

    Works with both TraceSummary (lightweight) and ExperimentRunTrace (full).
    """

    def safe_mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    def safe_std(values: list[float]) -> float:
        if len(values) < 2:
            return 0.0
        m = safe_mean(values)
        variance = sum((x - m) ** 2 for x in values) / len(values)
        return variance ** 0.5

    # Build query_id -> complexity lookup
    query_complexity_map = {q.id: q.complexity.value for q in queries}

    # Aggregate metrics by configuration
    config_metrics = {}

    for config_key, traces in results_by_config.items():
        f1_scores = [f for t in traces if (f := _get_metric(t, 'f1_score')) is not None]
        exec_accs = [e for t in traces if (e := _get_metric(t, 'execution_accuracy')) is not None]
        success_count = sum(1 for t in traces if t.success)
        total_tokens = [_get_tokens(t) for t in traces]
        total_times = [_get_time_ms(t) for t in traces]

        config_metrics[config_key] = {
            "f1_score": {
                "mean": safe_mean(f1_scores),
                "std": safe_std(f1_scores),
                "n": len(f1_scores),
            },
            "execution_accuracy": {
                "mean": safe_mean(exec_accs),
                "std": safe_std(exec_accs),
                "n": len(exec_accs),
            },
            "success_rate": success_count / len(traces) if traces else 0.0,
            "total_runs": len(traces),
            "avg_tokens": safe_mean(total_tokens),
            "avg_time_ms": safe_mean(total_times),
        }

    # Aggregate by model
    model_metrics = {}
    for model_key in models:
        model_traces = [
            t for t in all_traces
            if model_key in t.llm_model.lower() or LLM_MODELS.get(model_key, "") == t.llm_model
        ]

        f1_scores = [f for t in model_traces if (f := _get_metric(t, 'f1_score')) is not None]
        exec_accs = [e for t in model_traces if (e := _get_metric(t, 'execution_accuracy')) is not None]

        model_metrics[model_key] = {
            "f1_score": {"mean": safe_mean(f1_scores), "std": safe_std(f1_scores)},
            "execution_accuracy": {"mean": safe_mean(exec_accs), "std": safe_std(exec_accs)},
            "total_runs": len(model_traces),
        }

    # Aggregate by approach
    approach_metrics = {}
    for approach_key in approaches:
        approach_name = APPROACH_TYPES.get(approach_key, approach_key)
        approach_traces = [t for t in all_traces if t.approach == approach_name]

        f1_scores = [f for t in approach_traces if (f := _get_metric(t, 'f1_score')) is not None]
        exec_accs = [e for t in approach_traces if (e := _get_metric(t, 'execution_accuracy')) is not None]

        approach_metrics[approach_key] = {
            "f1_score": {"mean": safe_mean(f1_scores), "std": safe_std(f1_scores)},
            "execution_accuracy": {"mean": safe_mean(exec_accs), "std": safe_std(exec_accs)},
            "total_runs": len(approach_traces),
        }

    # Aggregate by query complexity (using query_id lookup)
    complexity_metrics = {}
    for complexity in ["SIMPLE", "MEDIUM", "HARD", "VERY_HARD"]:
        # Get query_id from TraceSummary or ExperimentRunTrace
        complexity_traces = []
        for t in all_traces:
            query_id = t.query_id if hasattr(t, 'query_id') else t.query_metadata.query_id
            if query_complexity_map.get(query_id) == complexity:
                complexity_traces.append(t)

        f1_scores = [f for t in complexity_traces if (f := _get_metric(t, 'f1_score')) is not None]

        if f1_scores:
            complexity_metrics[complexity] = {
                "f1_score": {"mean": safe_mean(f1_scores), "std": safe_std(f1_scores)},
                "total_runs": len(complexity_traces),
            }

    # Overall metrics
    all_f1 = [f for t in all_traces if (f := _get_metric(t, 'f1_score')) is not None]
    all_exec = [e for t in all_traces if (e := _get_metric(t, 'execution_accuracy')) is not None]

    return {
        "experiment_name": experiment_name,
        "timestamp": datetime.now().isoformat(),
        "config": {
            "models": models,
            "approaches": approaches,
            "runs_per_query": runs_per_query,
            "total_queries": len(queries),
            "total_runs": len(all_traces),
        },
        "overall": {
            "f1_score": {"mean": safe_mean(all_f1), "std": safe_std(all_f1)},
            "execution_accuracy": {"mean": safe_mean(all_exec), "std": safe_std(all_exec)},
            "success_rate": sum(1 for t in all_traces if t.success) / len(all_traces) if all_traces else 0.0,
        },
        "by_config": config_metrics,
        "by_model": model_metrics,
        "by_approach": approach_metrics,
        "by_complexity": complexity_metrics,
    }


def print_summary(summary: dict) -> None:
    """Print a formatted summary of experiment results."""
    print("\n" + "=" * 70)
    print(f"EXPERIMENT: {summary['experiment_name']}")
    print("=" * 70)

    config = summary["config"]
    print(f"\nConfiguration:")
    print(f"  Models: {', '.join(config['models'])}")
    print(f"  Approaches: {', '.join(config['approaches'])}")
    print(f"  Runs per query: {config['runs_per_query']}")
    print(f"  Total queries: {config['total_queries']}")
    print(f"  Total runs: {config['total_runs']}")

    overall = summary["overall"]
    print(f"\nOverall Results:")
    print(f"  F1-Score:       {overall['f1_score']['mean']:.3f} +/- {overall['f1_score']['std']:.3f}")
    print(f"  Exec Accuracy:  {overall['execution_accuracy']['mean']:.3f} +/- {overall['execution_accuracy']['std']:.3f}")
    print(f"  Success Rate:   {overall['success_rate']:.1%}")

    print(f"\nResults by Configuration:")
    print("-" * 70)
    print(f"{'Config':<25} {'F1-Score':<16} {'Exec Accuracy':<16} {'Success':<10}")
    print("-" * 70)

    for config_key, metrics in summary["by_config"].items():
        f1 = metrics["f1_score"]
        ea = metrics["execution_accuracy"]
        sr = metrics["success_rate"]
        print(f"{config_key:<25} {f1['mean']:.3f}+/-{f1['std']:.3f}    {ea['mean']:.3f}+/-{ea['std']:.3f}    {sr:.1%}")

    print("-" * 70)

    if summary.get("by_complexity"):
        print(f"\nResults by Complexity:")
        for complexity, metrics in summary["by_complexity"].items():
            f1 = metrics["f1_score"]
            print(f"  {complexity}: F1 = {f1['mean']:.3f} +/- {f1['std']:.3f} (n={metrics['total_runs']})")


# ============================================================================
# CLI Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run VKGQA experiments with multiple LLM models and approaches.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with Deepseek on specific queries, 3 runs each
  python scripts/run_full_experiment.py --models deepseek --queries BASE01 SYN01 TYPO01 --runs 3

  # Run all models on paired queries (BASE, SYN, TYPO)
  python scripts/run_full_experiment.py --models deepseek claude gpt4 --query-sets BASE SYN TYPO --runs 5

  # Run semantic approach only with all models on all queries
  python scripts/run_full_experiment.py --approaches semantic --runs 3

Available models: deepseek, claude, gpt4, gpt4-mini
Available approaches: grep, semantic
Query sets:
  BASE  - Baseline queries (16) with exact matches
  SYN   - Synonym variants (16) - same GT as BASE
  TYPO  - Typo variants (16) - same GT as BASE
  UNDER - Underspecified (5) - vague terms like "top", "recent"
  CROSS - Cross-dataset (5) - multi-endpoint queries
        """,
    )

    parser.add_argument(
        "--models", "-m",
        nargs="+",
        default=["deepseek"],
        choices=list(LLM_MODELS.keys()),
        help="LLM models to use (default: deepseek)",
    )

    parser.add_argument(
        "--approaches", "-a",
        nargs="+",
        default=["grep", "semantic"],
        choices=list(APPROACH_TYPES.keys()),
        help="Retrieval approaches to test (default: both grep and semantic)",
    )

    parser.add_argument(
        "--queries", "-q",
        nargs="+",
        help="Specific query IDs to test (e.g., BASE01 SYN01 TYPO01)",
    )

    parser.add_argument(
        "--query-sets", "-s",
        nargs="+",
        choices=["BASE", "SYN", "TYPO", "UNDER", "CROSS"],
        help="Query sets to test (e.g., BASE SYN TYPO)",
    )

    parser.add_argument(
        "--runs", "-r",
        type=int,
        default=3,
        help="Number of runs per query/model/approach (default: 3)",
    )

    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=RESULTS_DIR / "experiments",
        help="Output directory for results",
    )

    parser.add_argument(
        "--name", "-n",
        type=str,
        default=None,
        help="Experiment name (default: auto-generated)",
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        metavar="EXPERIMENT_NAME",
        help="Resume a previous experiment by name (e.g., exp_deepseek_20240315_120000)",
    )

    parser.add_argument(
        "--no-dashboard",
        action="store_true",
        help="Disable the live dashboard (enabled by default)",
    )

    parser.add_argument(
        "--keep-dashboard",
        action="store_true",
        help="Keep dashboard running after experiment completes (press Enter to stop)",
    )

    parser.add_argument(
        "--concurrency", "-c",
        type=int,
        default=5,
        help="Number of parallel runs (default: 5, adjust for API rate limits)",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Handle resume mode
    resume_mode = args.resume is not None
    if resume_mode:
        experiment_name = args.resume
        # Check if experiment directory exists
        experiment_dir = args.output / experiment_name
        if not experiment_dir.exists():
            print(f"Error: Experiment '{experiment_name}' not found in {args.output}")
            print(f"Available experiments:")
            for exp_dir in args.output.iterdir():
                if exp_dir.is_dir():
                    print(f"  - {exp_dir.name}")
            sys.exit(1)
        print(f"\nResuming experiment: {experiment_name}")
    else:
        # Generate experiment name
        if args.name:
            experiment_name = args.name
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            models_str = "_".join(args.models)
            experiment_name = f"exp_{models_str}_{timestamp}"

    # Load queries
    queries = load_queries(
        query_sets=args.query_sets,
        query_ids=args.queries,
    )

    if not queries:
        print("No queries selected. Use --queries or --query-sets.")
        sys.exit(1)

    print(f"\nExperiment: {experiment_name}")
    print(f"Queries: {len(queries)}")
    print(f"Models: {args.models}")
    print(f"Approaches: {args.approaches}")
    print(f"Runs per query: {args.runs}")
    print(f"Concurrency: {args.concurrency}")
    total_iterations = len(queries) * len(args.models) * len(args.approaches) * args.runs
    print(f"Total iterations: {total_iterations}")

    if resume_mode:
        print("Mode: RESUME (will skip completed runs)")
    print()

    # Start live dashboard
    dashboard_server = None
    dashboard_port = None
    experiment_dir = args.output / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    if not args.no_dashboard:
        dashboard_server, dashboard_port = start_dashboard_server(experiment_dir, experiment_name)
        if dashboard_server:
            url = f"http://localhost:{dashboard_port}"
            print(f"Live Dashboard: {url}")
            print()
            # Open browser after a short delay
            threading.Timer(1.0, lambda: webbrowser.open(url)).start()
        else:
            print("Warning: Could not start dashboard server")

    # Run experiment
    try:
        summary = asyncio.run(
            run_experiment(
                queries=queries,
                models=args.models,
                approaches=args.approaches,
                runs_per_query=args.runs,
                output_dir=experiment_dir,
                experiment_name=experiment_name,
                resume=resume_mode,
                concurrency=args.concurrency,
            )
        )

        # Print summary
        print_summary(summary)

        print(f"\nResults saved to: {experiment_dir}")
        if not resume_mode:
            print(f"\nTo resume this experiment later, use:")
            print(f"  python scripts/run_full_experiment.py --resume {experiment_name}")

        # Keep dashboard running after completion if requested
        if args.keep_dashboard and dashboard_server:
            print(f"\n{'='*60}")
            print("Experiment completed! Dashboard remains available.")
            print(f"Dashboard URL: http://localhost:{dashboard_port}")
            print("Press Enter to stop the dashboard and exit...")
            print(f"{'='*60}\n")
            try:
                input()
            except EOFError:
                # Handle non-interactive environments
                pass

    finally:
        # Shutdown dashboard server
        if dashboard_server:
            dashboard_server.shutdown()


if __name__ == "__main__":
    main()
