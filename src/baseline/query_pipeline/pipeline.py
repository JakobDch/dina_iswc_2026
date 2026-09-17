"""
Main SPARQL pipeline orchestration.

Simplified version for experiment - no workspace DB dependencies.
Works directly with file paths for semantic models.
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Union, Optional, List, Dict, Any

import httpx

from ..config import get_settings, AppSettings, LLM_PROFILES
from ..llm_services import OllamaLLM, OpenAILLM, DeepSeekLLM, LLMResponse
from ..comparison_utils import canonicalize_results_value_multiset, calculate_variable_level_metrics
from ..utils import parse_ttl_to_clean_triples

from .llm_invocation import (
    PipelineTokenUsage,
    identify_keywords_from_query_with_usage
)
from .sparql_generation import (
    generate_sparql_query,
    generate_sparql_query_with_usage,
    format_sparql_query_complete
)
from .sparql_execution import (
    execute_sparql_query,
    execute_sparql_in_ontop,
    validate_sparql_syntax_with_ontop,
    _add_limit_to_sparql_query
)
from .sparql_analysis import (
    calculate_diff_cost_deterministically,
    calculate_max_edit_cost_deterministically
)

logger = logging.getLogger(__name__)


def _create_llm_instance_from_profile(
    profile_name: str,
    settings: AppSettings
) -> Optional[Union[OpenAILLM, OllamaLLM, DeepSeekLLM]]:
    """Create an LLM instance based on profile configuration."""
    if profile_name not in LLM_PROFILES:
        logger.error(f"LLM profile '{profile_name}' not found")
        return None

    profile = LLM_PROFILES[profile_name]
    provider = profile["provider"]
    model = profile["model"]
    base_url = profile.get("base_url")
    requires_api_key = profile.get("requires_api_key", False)

    api_key = None
    if requires_api_key:
        api_key_env = profile.get("api_key_env")
        if api_key_env:
            api_key = getattr(settings, api_key_env.lower(), None)
        if not api_key:
            logger.error(f"API key for profile '{profile_name}' not found")
            return None

    if provider == "openai":
        return OpenAILLM(api_key=api_key, model=model, base_url=base_url, timeout=float(settings.llm_timeout_seconds))
    elif provider == "ollama":
        return OllamaLLM(model=model, base_url=base_url, timeout=float(settings.llm_timeout_seconds))
    else:
        logger.error(f"Unsupported provider '{provider}'")
        return None


def calculate_result_based_metrics(
    actual_results_json: dict,
    expected_results_json: dict,
    request_id: str = "metrics"
) -> Dict[str, float]:
    """
    Calculate result-based metrics for SPARQL query evaluation.

    Returns execution_accuracy, precision, recall, f1_score, and variable-level metrics.
    """
    log_prefix = f"[{request_id}]"

    canonical_actual = canonicalize_results_value_multiset(actual_results_json)
    canonical_expected = canonicalize_results_value_multiset(expected_results_json)

    if canonical_actual is None or canonical_expected is None:
        return {
            "execution_accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0
        }

    execution_accuracy = 1.0 if canonical_actual == canonical_expected else 0.0

    tp_counter = canonical_actual & canonical_expected
    tp_count = sum(tp_counter.values())

    fp_counter = canonical_actual - canonical_expected
    fp_count = sum(fp_counter.values())

    fn_counter = canonical_expected - canonical_actual
    fn_count = sum(fn_counter.values())

    if tp_count + fp_count > 0:
        precision = tp_count / (tp_count + fp_count)
    else:
        precision = 0.0 if fn_count > 0 else 1.0

    if tp_count + fn_count > 0:
        recall = tp_count / (tp_count + fn_count)
    else:
        recall = 1.0

    if precision + recall > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
    else:
        f1_score = 0.0

    result = {
        "execution_accuracy": execution_accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score
    }

    try:
        var_metrics = calculate_variable_level_metrics(actual_results_json, expected_results_json)
        result.update(var_metrics)
    except Exception as e:
        logger.warning(f"{log_prefix} Variable-level metrics calculation failed: {e}")

    return result


async def run_baseline_pipeline(
    user_query: str,
    semantic_model_paths: List[Path],
    llm_instance: Union[OllamaLLM, OpenAILLM, DeepSeekLLM],
    few_shot_prompting_enabled: bool = False,
    request_id: str = "baseline"
) -> Dict[str, Any]:
    """
    Run the simplified baseline SPARQL generation pipeline.

    Args:
        user_query: Natural language query
        semantic_model_paths: List of paths to semantic model TTL files
        llm_instance: LLM instance to use
        few_shot_prompting_enabled: Whether to use few-shot examples
        request_id: Request ID for logging

    Returns:
        Dictionary with generated SPARQL query and metadata
    """
    log_prefix = f"[{request_id}]"
    logger.info(f"{log_prefix} Starting baseline pipeline")

    token_usage = PipelineTokenUsage()

    # Step 1: Load semantic model content
    model_content_blocks = []
    for model_path in semantic_model_paths:
        if model_path.exists():
            content = parse_ttl_to_clean_triples(model_path, request_id)
            if content:
                model_content_blocks.append(f"Model: {model_path.name}\n{content}")
        else:
            logger.warning(f"{log_prefix} Model file not found: {model_path}")

    if not model_content_blocks:
        return {
            "status": "error",
            "error": "No semantic models could be loaded",
            "generated_sparql_query": None,
            "token_usage": token_usage.get_summary()
        }

    model_info_blocks = "\n\n".join(model_content_blocks)

    # Step 2: Generate SPARQL query
    try:
        gen_result, gen_response = await generate_sparql_query_with_usage(
            user_query=user_query,
            model_info_blocks=model_info_blocks,
            model_check_hints="",
            llm_instance=llm_instance,
            workspace_id="experiment",
            few_shot_prompting_enabled=few_shot_prompting_enabled,
            adaptive_few_shot_enabled=False,
            instance_search_results=None,
            request_id=f"{request_id}_gen",
            is_sparql_only=True
        )
        token_usage.add_step_usage("generate_sparql", gen_response)
        generated_query = gen_result.get("sparql_query", "")

    except Exception as e:
        logger.error(f"{log_prefix} SPARQL generation failed: {e}")
        return {
            "status": "error",
            "error": f"SPARQL generation failed: {e}",
            "generated_sparql_query": None,
            "token_usage": token_usage.get_summary()
        }

    return {
        "status": "success",
        "generated_sparql_query": format_sparql_query_complete(generated_query),
        "reasoning": gen_result.get("reasoning", ""),
        "models_used": [p.name for p in semantic_model_paths],
        "token_usage": token_usage.get_summary()
    }


async def execute_and_evaluate(
    generated_query: str,
    expected_results: Dict[str, Any],
    request_id: str = "eval"
) -> Dict[str, Any]:
    """
    Execute a generated SPARQL query and evaluate against expected results.

    Args:
        generated_query: Generated SPARQL query string
        expected_results: Expected results in SPARQL JSON format
        request_id: Request ID for logging

    Returns:
        Dictionary with execution results and metrics
    """
    log_prefix = f"[{request_id}]"

    # Execute query
    try:
        result = await execute_sparql_query(generated_query)

        if result is None:
            return {
                "execution_success": False,
                "error": "Query execution returned None",
                "metrics": None
            }

        if "error_details" in result:
            return {
                "execution_success": False,
                "error": result["error_details"],
                "metrics": None
            }

        # Calculate metrics
        metrics = calculate_result_based_metrics(result, expected_results, request_id)

        return {
            "execution_success": True,
            "result": result,
            "metrics": metrics
        }

    except Exception as e:
        logger.error(f"{log_prefix} Query execution failed: {e}")
        return {
            "execution_success": False,
            "error": str(e),
            "metrics": None
        }


async def run_baseline_with_evaluation(
    user_query: str,
    semantic_model_paths: List[Path],
    expected_results: Dict[str, Any],
    llm_instance: Union[OllamaLLM, OpenAILLM, DeepSeekLLM],
    few_shot_prompting_enabled: bool = False,
    request_id: str = "baseline_eval"
) -> Dict[str, Any]:
    """
    Run the baseline pipeline and evaluate the generated query.

    Args:
        user_query: Natural language query
        semantic_model_paths: List of paths to semantic model TTL files
        expected_results: Expected results in SPARQL JSON format
        llm_instance: LLM instance to use
        few_shot_prompting_enabled: Whether to use few-shot examples
        request_id: Request ID for logging

    Returns:
        Dictionary with generation results, execution results, and metrics
    """
    # Generate SPARQL
    gen_result = await run_baseline_pipeline(
        user_query=user_query,
        semantic_model_paths=semantic_model_paths,
        llm_instance=llm_instance,
        few_shot_prompting_enabled=few_shot_prompting_enabled,
        request_id=request_id
    )

    if gen_result["status"] != "success" or not gen_result.get("generated_sparql_query"):
        return {
            **gen_result,
            "execution_success": False,
            "metrics": None
        }

    # Execute and evaluate
    eval_result = await execute_and_evaluate(
        generated_query=gen_result["generated_sparql_query"],
        expected_results=expected_results,
        request_id=f"{request_id}_eval"
    )

    return {
        **gen_result,
        **eval_result
    }
