"""
SPARQL query generation from natural language.

Provides functions to generate SPARQL queries using LLM with semantic model
context and optional few-shot examples.
"""

import logging
import re
from typing import Union, Optional, List, Dict, Any

from ..llm_services import OllamaLLM, OpenAILLM, DeepSeekLLM, LLMResponse
from ..prompts import SPARQL_GENERATOR_PROMPT_MULTI_MODEL, OUTPUT_INSTRUCTIONS_GENERATE_SPARQL
from .llm_invocation import (
    _invoke_llm_and_parse_with_adaptive_few_shot,
    _invoke_llm_and_parse_with_adaptive_few_shot_with_usage
)

logger = logging.getLogger(__name__)

CORRECT_SPARQL_PREFIXES = """
PREFIX owl:    <http://www.w3.org/2002/07/owl#>
PREFIX plasma: <http://plasma.uni-wuppertal.de/ontology#>
PREFIX plcm:   <http://plasma.uni-wuppertal.de/cm#>
PREFIX plsm:   <http://plasma.uni-wuppertal.de/sm/>
PREFIX rdf:    <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs:   <http://www.w3.org/2000/01/rdf-schema#>
PREFIX xsd:    <http://www.w3.org/2001/XMLSchema#>
PREFIX local:  <https://local.ontology#>
PREFIX eno:   <http://example.org/ontology/energy#>
PREFIX enoptl:    <http://example.org/ontology/energy-ptl#>
"""


def _add_correct_prefixes(query_string: str) -> str:
    """Replace all PREFIX declarations with standardized prefixes."""
    if not query_string:
        return ""

    prefix_pattern = re.compile(r"^(\s*PREFIX\s+[\w\-]*:\s*<[^>]*>\s*)+", flags=re.IGNORECASE)
    query_body = prefix_pattern.sub("", query_string.strip()).strip()

    return f"{CORRECT_SPARQL_PREFIXES.strip()}\n\n{query_body}"


def format_sparql_query_complete(query: str) -> str:
    """Format SPARQL query with standardized prefixes and formatting."""
    if not query:
        return ""

    prefix_corrected = _add_correct_prefixes(query)

    lines = prefix_corrected.split('\n')
    formatted_lines = []
    indent_level = 0
    in_prefix_section = True

    for line in lines:
        stripped = line.strip()

        if not stripped:
            if not in_prefix_section:
                formatted_lines.append("")
            continue

        if stripped.upper().startswith('PREFIX'):
            formatted_lines.append(stripped)
            continue

        in_prefix_section = False

        if stripped.startswith('}'):
            indent_level = max(0, indent_level - 1)

        indent = "  " * indent_level
        formatted_lines.append(f"{indent}{stripped}")

        if stripped.endswith('{'):
            indent_level += 1

    return '\n'.join(formatted_lines)


def _robust_sparql_sanitize(sparql_candidate: str, request_id: str) -> str:
    """Remove markdown code blocks and other formatting artifacts from SPARQL."""
    if not sparql_candidate:
        return ""
    sanitized = re.sub(r'^\s*```sparql\s*(.*?)\s*```\s*$', r'\1', sparql_candidate,
                       flags=re.IGNORECASE | re.DOTALL | re.MULTILINE)
    sanitized = re.sub(r'^\s*```\s*(.*?)\s*```\s*$', r'\1', sanitized,
                       flags=re.IGNORECASE | re.DOTALL | re.MULTILINE)
    sanitized = sanitized.strip()
    if not re.search(r"(?i)^(PREFIX|SELECT|CONSTRUCT|ASK|DESCRIBE)", sanitized):
        logger.warning(f"[{request_id}] No SPARQL keyword found in sanitized output")
    return sanitized


async def generate_sparql_query(
    user_query: str,
    model_info_blocks: str,
    model_check_hints: str,
    llm_instance: Union[OllamaLLM, OpenAILLM, DeepSeekLLM],
    workspace_id: str,
    few_shot_prompting_enabled: bool = False,
    adaptive_few_shot_enabled: bool = False,
    instance_search_results: Optional[Dict[str, Any]] = None,
    request_id: str = "N/A",
    is_sparql_only: bool = False,
    exclude_template_ids: Optional[List[str]] = None
) -> dict:
    """Generate SPARQL query from user query using LLM."""
    logger.info(f"[{request_id}] Generating SPARQL for: '{user_query[:100]}...'")

    instance_data_info = ""
    if instance_search_results and not instance_search_results.get("error"):
        found_terms = instance_search_results.get("found_terms", [])
        results_by_term = instance_search_results.get("results_by_term", {})

        if found_terms:
            instance_data_info = "**FOUND RDF TRIPLES FROM DATA:**\n\n"
            for term in found_terms:
                term_data = results_by_term.get(term, {})
                example_matches = term_data.get("example_matches", [])[:2]
                if example_matches:
                    instance_data_info += f"**Search term '{term}' - Found triples:**\n"
                    for example in example_matches:
                        semantic_structure = example.get('semantic_structure', '')
                        if semantic_structure:
                            instance_data_info += f"```\n{semantic_structure}\n```\n"
                    instance_data_info += "\n"

    optional_instance_data_instructions = ""
    if instance_data_info:
        optional_instance_data_instructions = """
- IMPORTANT: The provided instance triples show the exact structure of real RDF data
- Use these structures as a template for your SPARQL query
- For string matching use FILTER with CONTAINS() and LCASE() for case-insensitive search
"""

    prompt_params = {
        "query": user_query,
        "model_info_blocks": model_info_blocks,
        "model_check_hints": model_check_hints,
        "instance_data_info": instance_data_info,
        "optional_instance_data_instructions": optional_instance_data_instructions
    }

    reasoning, structured_output, raw_response, adaptive_examples_metadata, operator_token_usage = \
        await _invoke_llm_and_parse_with_adaptive_few_shot(
            llm_instance,
            SPARQL_GENERATOR_PROMPT_MULTI_MODEL,
            prompt_params,
            OUTPUT_INSTRUCTIONS_GENERATE_SPARQL,
            few_shot_prompting_enabled,
            adaptive_few_shot_enabled,
            request_id,
            "SPARQLGeneration",
            workspace_id,
            is_sparql_only,
            exclude_template_ids
        )

    sanitized_output = _robust_sparql_sanitize(structured_output, request_id)
    final_sparql_query = _add_correct_prefixes(sanitized_output)

    return {
        "sparql_query": final_sparql_query,
        "reasoning": reasoning,
        "raw_response": raw_response,
        "adaptive_few_shot_examples": adaptive_examples_metadata,
        "operator_token_usage": operator_token_usage
    }


async def generate_sparql_query_with_usage(
    user_query: str,
    model_info_blocks: str,
    model_check_hints: str,
    llm_instance: Union[OllamaLLM, OpenAILLM, DeepSeekLLM],
    workspace_id: str,
    few_shot_prompting_enabled: bool = False,
    adaptive_few_shot_enabled: bool = False,
    instance_search_results: Optional[Dict[str, Any]] = None,
    request_id: str = "N/A",
    is_sparql_only: bool = False,
    exclude_template_ids: Optional[List[str]] = None
) -> tuple[dict, LLMResponse]:
    """Generate SPARQL query with token usage tracking."""
    logger.info(f"[{request_id}] Generating SPARQL (with usage) for: '{user_query[:100]}...'")

    instance_data_info = ""
    if instance_search_results and not instance_search_results.get("error"):
        found_terms = instance_search_results.get("found_terms", [])
        results_by_term = instance_search_results.get("results_by_term", {})

        if found_terms:
            instance_data_info = "**FOUND RDF TRIPLES FROM DATA:**\n\n"
            for term in found_terms:
                term_data = results_by_term.get(term, {})
                example_matches = term_data.get("example_matches", [])[:2]
                if example_matches:
                    instance_data_info += f"**Search term '{term}' - Found triples:**\n"
                    for example in example_matches:
                        semantic_structure = example.get('semantic_structure', '')
                        if semantic_structure:
                            instance_data_info += f"```\n{semantic_structure}\n```\n"
                    instance_data_info += "\n"

    optional_instance_data_instructions = ""
    if instance_data_info:
        optional_instance_data_instructions = """
- IMPORTANT: The provided instance triples show the exact structure of real RDF data
- Use these structures as a template for your SPARQL query
- For string matching use FILTER with CONTAINS() and LCASE() for case-insensitive search
"""

    prompt_params = {
        "query": user_query,
        "model_info_blocks": model_info_blocks,
        "model_check_hints": model_check_hints,
        "instance_data_info": instance_data_info,
        "optional_instance_data_instructions": optional_instance_data_instructions
    }

    reasoning, structured_output, raw_response, llm_response, adaptive_examples_metadata, operator_token_usage = \
        await _invoke_llm_and_parse_with_adaptive_few_shot_with_usage(
            llm_instance,
            SPARQL_GENERATOR_PROMPT_MULTI_MODEL,
            prompt_params,
            OUTPUT_INSTRUCTIONS_GENERATE_SPARQL,
            few_shot_prompting_enabled,
            adaptive_few_shot_enabled,
            request_id,
            "SPARQLGeneration",
            workspace_id,
            is_sparql_only,
            exclude_template_ids
        )

    sanitized_output = _robust_sparql_sanitize(structured_output, request_id)
    final_sparql_query = _add_correct_prefixes(sanitized_output)

    return {
        "sparql_query": final_sparql_query,
        "reasoning": reasoning,
        "raw_response": raw_response,
        "adaptive_few_shot_examples": adaptive_examples_metadata,
        "operator_token_usage": operator_token_usage
    }, llm_response
