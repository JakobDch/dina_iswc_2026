"""
Baseline pipeline from chat2sparql-main.

Simplified version for experiment - no workspace DB dependencies.
Only supports OnTop SPARQL endpoint.
"""

from .config import get_settings, AppSettings, LLM_PROFILES
from .llm_services import OpenAILLM, OllamaLLM, DeepSeekLLM, LLMResponse
from .query_pipeline.pipeline import (
    run_baseline_pipeline,
    run_baseline_with_evaluation,
    execute_and_evaluate,
    calculate_result_based_metrics,
    _create_llm_instance_from_profile
)
from .query_pipeline.sparql_execution import (
    execute_sparql_query,
    execute_sparql_in_ontop,
    validate_sparql_syntax_with_ontop
)
from .query_pipeline.sparql_generation import (
    generate_sparql_query,
    generate_sparql_query_with_usage,
    format_sparql_query_complete
)

__all__ = [
    # Config
    "get_settings",
    "AppSettings",
    "LLM_PROFILES",
    # LLM Services
    "OpenAILLM",
    "OllamaLLM",
    "DeepSeekLLM",
    "LLMResponse",
    # Pipeline
    "run_baseline_pipeline",
    "run_baseline_with_evaluation",
    "execute_and_evaluate",
    "calculate_result_based_metrics",
    "_create_llm_instance_from_profile",
    # SPARQL Execution
    "execute_sparql_query",
    "execute_sparql_in_ontop",
    "validate_sparql_syntax_with_ontop",
    # SPARQL Generation
    "generate_sparql_query",
    "generate_sparql_query_with_usage",
    "format_sparql_query_complete",
]
