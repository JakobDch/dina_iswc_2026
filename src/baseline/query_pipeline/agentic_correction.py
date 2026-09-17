"""
Agentic correction for SPARQL queries.

Provides functions for syntax error correction, empty results correction,
and general agentic reasoning to fix problematic SPARQL queries.
"""

import asyncio
import json
import logging
import re
from typing import Union, Optional, List, Dict, Any

import httpx

from ..llm_services import OllamaLLM, OpenAILLM, DeepSeekLLM, LLMResponse
from .sparql_analysis import robust_json_parse
from .sparql_execution import execute_sparql_query
from .sparql_generation import _add_correct_prefixes, _robust_sparql_sanitize
from .formatting import (
    format_hierarchical_decomposition_for_llm,
    format_instance_search_for_prompt,
    format_few_shot_examples_for_prompt
)

logger = logging.getLogger(__name__)


def classify_syntax_error(error_message: str) -> str:
    """Classify a syntax error based on GraphDB error message."""
    error_msg_lower = error_message.lower()

    error_patterns = {
        "missing_quotes": ["unexpected token", "literal", "string", "quote"],
        "missing_brackets": ["expected", "bracket", "<", ">", "uri"],
        "missing_dot": ["expected", ".", "dot", "statement"],
        "missing_curly_braces": ["expected", "{", "}", "brace", "where"],
        "unbalanced_parentheses": ["expected", "(", ")", "parenthes", "balanced"],
        "wrong_variable_prefix": ["variable", "$", "?", "prefix"],
        "keyword_case_error": ["select", "where", "from", "keyword"],
        "missing_semicolon": ["expected", ";", "semicolon"],
        "invalid_uri_format": ["uri", "invalid", "format", "namespace"],
        "trailing_comma": ["unexpected", ",", "comma"],
        "whitespace_error": ["space", "whitespace", "expected"],
        "invalid_prefix": ["prefix", "namespace", "declaration"],
        "duplicate_variable_alias": ["bind", "alias", "previously", "used"],
        "malformed_query": ["malformed query"],
        "unexpected_token": ["unexpected token"],
        "syntax_error": ["syntax error"],
        "parse_error": ["parse error"]
    }

    for error_type, keywords in error_patterns.items():
        if all(keyword in error_msg_lower for keyword in keywords[:2]):
            return error_type

    return "unknown_syntax_error"


async def collect_syntax_errors(
    generated_query: str,
    workspace_id: str,
    request_id: str = "N/A"
) -> Dict[str, Any]:
    """
    Collect syntax errors from a SPARQL query without correction attempts.

    Validates against GraphDB and classifies any found errors.
    """
    logger.info(f"[{request_id}] Checking syntax errors")

    syntax_errors_found = []

    from ..db import get_session
    from ..models import Workspace

    db_session = next(get_session())
    try:
        workspace = db_session.get(Workspace, workspace_id)
        if not workspace:
            return {
                "success": False,
                "syntax_errors_found": [],
                "error": f"Workspace {workspace_id} not found"
            }
    finally:
        db_session.close()

    async with httpx.AsyncClient(timeout=15.0) as client:
        result = await execute_sparql_query(client, workspace, generated_query)

        if result is None or "error_details" not in result:
            return {
                "success": True,
                "syntax_errors_found": [],
                "query_results": result
            }

        error_details = result.get("error_details", {})
        if error_details.get("error_type") != "syntax_error":
            return {
                "success": False,
                "syntax_errors_found": [],
                "error": f"Non-syntax error: {error_details.get('error_message', 'Unknown')}"
            }

        syntax_error_msg = error_details.get("error_message", "")
        error_type = classify_syntax_error(syntax_error_msg)
        syntax_errors_found.append(error_type)

        return {
            "success": False,
            "syntax_errors_found": syntax_errors_found,
            "error_message": syntax_error_msg,
            "error": f"Syntax error: {error_type}"
        }


async def fix_syntax_errors_only(
    generated_query: str,
    groundtruth_query: str,
    syntax_error_details: str,
    llm_instance: Union[OllamaLLM, OpenAILLM, DeepSeekLLM],
    request_id: str = "N/A"
) -> Dict[str, Any]:
    """
    Fix only syntax errors without changing query semantics.

    Uses groundtruth query as reference for correct syntax patterns only.
    """
    logger.info(f"[{request_id}] Starting syntax-only correction")

    syntax_correction_prompt = f"""You are a SPARQL syntax expert. Fix ONLY syntax errors without changing semantics.

RULES:
1. NEVER change variable names, property URIs, class URIs, or logical structures
2. Fix ONLY syntax problems like missing brackets, quotes, dots, etc.
3. Use groundtruth query ONLY as reference for correct syntax patterns
4. Preserve original semantics and logic

PROBLEMATIC QUERY:
```sparql
{generated_query}
```

SYNTAX ERROR DETAILS:
{syntax_error_details}

GROUNDTRUTH REFERENCE (syntax patterns only):
```sparql
{groundtruth_query}
```

Respond in JSON format:
{{
    "analysis": "Brief description of syntax problems found",
    "syntax_fixes": ["list_of_applied_syntax_fixes"],
    "corrected_query": "syntax_corrected_sparql_query"
}}"""

    try:
        llm_response = await asyncio.to_thread(llm_instance.invoke_with_usage, syntax_correction_prompt)
        raw_response = llm_response.content

        json_match = re.search(r'\{[\s\S]*\}', raw_response.strip())
        if not json_match:
            return {
                "error": "LLM response format invalid",
                "corrected_query": None,
                "syntax_fixes": []
            }

        parsed_response = json.loads(json_match.group(0))

        return {
            "corrected_query": parsed_response.get("corrected_query", "").strip(),
            "syntax_fixes": parsed_response.get("syntax_fixes", []),
            "analysis": parsed_response.get("analysis", ""),
            "error": None
        }

    except Exception as e:
        logger.error(f"[{request_id}] Syntax correction error: {e}")
        return {
            "error": str(e),
            "corrected_query": None,
            "syntax_fixes": []
        }


async def apply_agentic_reasoning_to_sparql(
    generated_sparql_query: str,
    user_query: str,
    model_info_blocks: str,
    model_check_hints: str,
    llm_instance: Union[OllamaLLM, OpenAILLM, DeepSeekLLM],
    error_context: str,
    request_id: str = "N/A"
) -> tuple[str, str, str, LLMResponse]:
    """
    Apply agentic reasoning to identify and fix SPARQL query issues.

    Returns tuple of (assessment, justification, corrected_query, llm_response).
    """
    logger.info(f"[{request_id}] Starting agentic reasoning")

    formatted_prompt = f"""You are a SPARQL expert. The generated query has a problem.

User Query: {user_query}

Semantic Models:
{model_info_blocks}

Model Check Hints: {model_check_hints}

Generated SPARQL query:
{generated_sparql_query}

Problem: {error_context}

Analyze and fix the SPARQL query. Check:
1. Properties and classes used match semantic models
2. Correct triple connections
3. Possible filter problems
4. Syntax errors

Respond in JSON format:
{{"assessment": "faulty", "justification": "Problem explanation", "corrected_query": "corrected SPARQL query"}}"""

    try:
        agentic_response = await asyncio.to_thread(
            llm_instance.invoke_with_usage,
            formatted_prompt
        )

        agentic_data = robust_json_parse(agentic_response.content, request_id)

        if agentic_data:
            assessment = agentic_data.get("assessment", "faulty")
            justification = agentic_data.get("justification", "No justification.")
            corrected_query = agentic_data.get("corrected_query", "")

            return assessment, justification, corrected_query, agentic_response
        else:
            return "error", "JSON parsing failed", "", agentic_response

    except Exception as e:
        logger.error(f"[{request_id}] Agentic reasoning error: {e}")
        return "error", str(e), "", LLMResponse(content="", input_tokens=0, output_tokens=0, total_tokens=0)


async def apply_empty_results_correction(
    generated_sparql_query: str,
    user_query: str,
    model_info_blocks: str,
    model_check_hints: str,
    llm_instance: Union[OllamaLLM, OpenAILLM, DeepSeekLLM],
    previous_attempts: List[str],
    instance_search_results: Optional[Dict[str, Any]] = None,
    adaptive_few_shot_examples: Optional[List[Dict[str, Any]]] = None,
    decomposition_analysis_context: str = "",
    request_id: str = "N/A"
) -> tuple[bool, str, str, LLMResponse]:
    """
    Apply correction for SPARQL queries that returned empty results.

    Returns tuple of (should_correct, justification, corrected_query, llm_response).
    """
    logger.info(f"[{request_id}] Starting empty results correction")

    instance_data_section = format_instance_search_for_prompt(instance_search_results)
    few_shot_section = format_few_shot_examples_for_prompt(adaptive_few_shot_examples)

    previous_attempts_section = ""
    if previous_attempts:
        previous_attempts_section = "**PREVIOUS CORRECTION ATTEMPTS (FAILED):**\n\n"
        for i, attempt in enumerate(previous_attempts[-3:], 1):
            previous_attempts_section += f"Attempt {i}:\n```sparql\n{attempt[:500]}...\n```\n\n"
        previous_attempts_section += "Do NOT repeat these queries. Try a different approach.\n\n"

    formatted_prompt = f"""You are a SPARQL debugging expert. The query returned empty results.

**USER QUERY:** {user_query}

**SEMANTIC MODELS:**
{model_info_blocks}

**MODEL HINTS:** {model_check_hints}

**GENERATED QUERY (EMPTY RESULTS):**
```sparql
{generated_sparql_query}
```

{instance_data_section}

{few_shot_section}

{decomposition_analysis_context}

{previous_attempts_section}

**TASK:**
1. Decide if empty results are LEGITIMATE (query correct but no matching data)
2. Or if query needs CORRECTION (wrong structure, missing connections, etc.)

If legitimate: {{"should_correct": false, "justification": "explanation"}}
If needs fix: {{"should_correct": true, "justification": "explanation", "corrected_query": "fixed_sparql"}}

Common issues to check:
- Wrong property/class URIs
- Missing type constraints
- Over-restrictive FILTERs
- Wrong join variables
- String literal format mismatches"""

    try:
        llm_response = await asyncio.to_thread(
            llm_instance.invoke_with_usage,
            formatted_prompt
        )

        parsed = robust_json_parse(llm_response.content, request_id)

        if parsed:
            should_correct = parsed.get("should_correct", False)
            justification = parsed.get("justification", "No justification")
            corrected_query = parsed.get("corrected_query", "") if should_correct else ""

            if corrected_query:
                corrected_query = _robust_sparql_sanitize(corrected_query, request_id)
                corrected_query = _add_correct_prefixes(corrected_query)

            return should_correct, justification, corrected_query, llm_response
        else:
            return False, "JSON parsing failed", "", llm_response

    except Exception as e:
        logger.error(f"[{request_id}] Empty results correction error: {e}")
        return False, str(e), "", LLMResponse(content="", input_tokens=0, output_tokens=0, total_tokens=0)
