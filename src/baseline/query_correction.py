"""
Agent-based query correction for evaluation.

This module provides functions to correct generated SPARQL queries
using the groundtruth as reference, while preserving the original
structure as much as possible.

Supports iterative correction with feedback from execution results.
"""

import asyncio
import logging
import re
from typing import Any

from .llm_services import LLMInstance, LLMResponse
from .sparql_analysis import robust_json_parse
from .metrics import canonicalize_results

logger = logging.getLogger(__name__)


CORRECTION_SYSTEM_MESSAGE = """You are a SPARQL query correction expert.
Your task is to correct a generated SPARQL query to produce the correct results,
while keeping the structure as close to the original as possible.

Key principles:
1. Only make changes that are necessary for correctness
2. Preserve variable names from the original when possible
3. Preserve the query structure (SELECT/WHERE pattern, etc.)
4. Only change properties, classes, or filters that are actually wrong
5. Do not add unnecessary elements"""


async def correct_query_for_evaluation(
    generated_query: str,
    groundtruth_query: str,
    llm_instance: LLMInstance,
    user_query: str = "",
    request_id: str = "",
) -> dict[str, Any]:
    """
    Correct a generated SPARQL query using groundtruth as reference.

    The goal is to create a corrected query that:
    1. Produces the correct results (like groundtruth)
    2. Is structurally as close as possible to the generated query

    This is used for evaluation to measure how much the generated query
    needs to be changed to be correct.

    Args:
        generated_query: The generated SPARQL query to correct
        groundtruth_query: The correct SPARQL query (reference)
        llm_instance: The LLM instance to use for correction
        user_query: Optional natural language query for context
        request_id: Optional request ID for logging

    Returns:
        Dictionary with:
            - corrected_query: The corrected SPARQL query
            - changes_made: List of changes applied
            - error: Error message if correction failed
    """
    log_prefix = f"[{request_id}]" if request_id else ""
    logger.info(f"{log_prefix} Starting query correction for evaluation")

    user_context = f"\nOriginal user question: {user_query}\n" if user_query else ""

    correction_prompt = f"""Correct the following SPARQL query to produce the correct results.

IMPORTANT RULES:
1. Keep the structure as close to the original generated query as possible
2. Only change what is necessary for correctness
3. If variable names differ but the query is otherwise correct, prefer keeping original variable names
4. Do not add unnecessary elements (comments, extra triples, etc.)
5. The corrected query must produce the same results as the groundtruth
{user_context}
GENERATED QUERY (to be corrected):
```sparql
{generated_query}
```

GROUNDTRUTH QUERY (reference for correct results):
```sparql
{groundtruth_query}
```

Analyze the differences and create a corrected version of the generated query.
The corrected query should produce the same results as groundtruth but maintain
the generated query's structure where possible.

Respond in JSON format:
{{
    "analysis": "Brief analysis of what needs to be changed",
    "changes_made": ["list", "of", "specific", "changes"],
    "corrected_query": "the corrected SPARQL query"
}}"""

    try:
        llm_response: LLMResponse = await asyncio.to_thread(
            llm_instance.invoke_with_usage,
            correction_prompt,
            CORRECTION_SYSTEM_MESSAGE,
        )

        parsed = robust_json_parse(llm_response.content, request_id)

        if parsed:
            corrected_query = parsed.get("corrected_query", "").strip()
            changes_made = parsed.get("changes_made", [])
            analysis = parsed.get("analysis", "")

            # Clean up the query (remove markdown formatting if present)
            corrected_query = _clean_query_response(corrected_query)

            logger.info(
                f"{log_prefix} Query correction successful. "
                f"Changes: {len(changes_made)}"
            )

            return {
                "corrected_query": corrected_query,
                "changes_made": changes_made,
                "analysis": analysis,
                "error": None,
                "token_usage": {
                    "input_tokens": llm_response.input_tokens,
                    "output_tokens": llm_response.output_tokens,
                    "total_tokens": llm_response.total_tokens,
                },
            }
        else:
            logger.error(f"{log_prefix} Failed to parse LLM response as JSON")
            return {
                "corrected_query": groundtruth_query,  # Fallback to groundtruth
                "changes_made": ["Fallback: used groundtruth directly"],
                "analysis": "LLM response parsing failed",
                "error": "JSON parsing failed",
            }

    except Exception as e:
        logger.error(f"{log_prefix} Query correction error: {e}")
        return {
            "corrected_query": groundtruth_query,  # Fallback to groundtruth
            "changes_made": ["Fallback: used groundtruth directly"],
            "analysis": "",
            "error": str(e),
        }


def _clean_query_response(query: str) -> str:
    """Clean SPARQL query from LLM response artifacts."""
    # Remove markdown code blocks
    query = re.sub(r"```(?:sparql)?\s*\n?", "", query)
    query = re.sub(r"\n?```", "", query)

    # Remove zero-width spaces
    query = query.replace("\u200B", "")

    return query.strip()


async def batch_correct_queries(
    query_pairs: list[tuple[str, str]],
    llm_instance: LLMInstance,
    user_queries: list[str] | None = None,
    request_id_prefix: str = "",
) -> list[dict[str, Any]]:
    """
    Correct multiple query pairs in batch.

    Args:
        query_pairs: List of (generated_query, groundtruth_query) tuples
        llm_instance: The LLM instance to use
        user_queries: Optional list of user queries for context
        request_id_prefix: Prefix for request IDs

    Returns:
        List of correction results
    """
    results = []

    for i, (generated, groundtruth) in enumerate(query_pairs):
        user_query = user_queries[i] if user_queries and i < len(user_queries) else ""
        request_id = f"{request_id_prefix}_{i}" if request_id_prefix else str(i)

        result = await correct_query_for_evaluation(
            generated_query=generated,
            groundtruth_query=groundtruth,
            llm_instance=llm_instance,
            user_query=user_query,
            request_id=request_id,
        )
        results.append(result)

    return results


def correct_query_sync(
    generated_query: str,
    groundtruth_query: str,
    llm_instance: LLMInstance,
    user_query: str = "",
    request_id: str = "",
) -> dict[str, Any]:
    """
    Synchronous wrapper for correct_query_for_evaluation.

    Use this when you're not in an async context.
    """
    return asyncio.run(
        correct_query_for_evaluation(
            generated_query=generated_query,
            groundtruth_query=groundtruth_query,
            llm_instance=llm_instance,
            user_query=user_query,
            request_id=request_id,
        )
    )


# =============================================================================
# Iterative Correction with Validation
# =============================================================================

CORRECTION_PROMPT_WITH_FEEDBACK = """You are correcting a SPARQL query. {iteration_context}

ORIGINAL GENERATED QUERY (to be corrected):
```sparql
{generated_query}
```

GROUNDTRUTH QUERY (reference for correct results):
```sparql
{groundtruth_query}
```

{user_context}
{previous_attempts_section}
{feedback_section}

CRITICAL - VARIABLE NAME NORMALIZATION:
You MUST preserve variable names from the ORIGINAL GENERATED QUERY wherever possible!
- If the generated query uses ?person and groundtruth uses ?x for the same concept, use ?person
- If the generated query uses ?name and groundtruth uses ?label for the same binding, use ?name
- Only introduce new variable names if the generated query doesn't have an equivalent

IMPORTANT RULES:
1. PRESERVE variable names from the original generated query
2. Keep the structure as close to the original generated query as possible
3. Only change what is necessary for correctness (properties, classes, filters)
4. The corrected query MUST produce the same results as the groundtruth
5. {iteration_specific_instruction}

Respond in JSON format:
{{
    "reasoning": "Brief explanation of your corrections",
    "variable_mapping": {{"groundtruth_var": "generated_var", ...}},
    "corrected_query": "the corrected SPARQL query"
}}"""


FINAL_ITERATION_NORMALIZATION_PROMPT = """You are performing ONLY variable name normalization on a SPARQL query.

This is the FINAL attempt. Previous correction attempts failed to produce correct results.
Your task now is DIFFERENT: Take the GROUNDTRUTH query and ONLY change the variable names
to match those used in the ORIGINAL GENERATED query.

ORIGINAL GENERATED QUERY (source for variable names):
```sparql
{generated_query}
```

GROUNDTRUTH QUERY (to be normalized):
```sparql
{groundtruth_query}
```

{user_context}

YOUR TASK:
1. Identify which variables in the groundtruth correspond to which variables in the generated query
2. Create a mapping: groundtruth variable -> generated variable
3. Replace ALL variable names in the groundtruth with the corresponding generated variable names
4. DO NOT change anything else - no properties, no classes, no filters, no structure
5. If a groundtruth variable has no clear equivalent in generated, keep it as-is

Example:
- Generated uses: ?person, ?name, ?age
- Groundtruth uses: ?x, ?label, ?years
- Mapping: ?x->?person, ?label->?name, ?years->?age
- Result: Groundtruth query with ?person, ?name, ?age instead

Respond in JSON format:
{{
    "variable_mapping": {{"?x": "?person", "?label": "?name", ...}},
    "corrected_query": "the groundtruth query with normalized variable names"
}}"""


def generate_correction_feedback(
    actual_results: dict | None,
    expected_results: dict,
    error_type: str | None = None,
    error_message: str | None = None,
) -> dict[str, Any]:
    """
    Generate detailed feedback for the next correction attempt.

    Args:
        actual_results: Results from executing the corrected query (None if execution failed)
        expected_results: Expected results from groundtruth
        error_type: Type of error if execution failed
        error_message: Error message if execution failed

    Returns:
        Dictionary with feedback details
    """
    # Handle execution errors
    if error_type == "syntax_error":
        return {
            "error_type": "syntax_error",
            "message": f"Query has syntax errors: {error_message}",
            "missing_rows": [],
            "extra_rows": [],
            "row_count_actual": 0,
            "row_count_expected": 0,
        }

    if error_type in ["connection_error", "http_error", "timeout"]:
        return {
            "error_type": error_type,
            "message": f"Query execution failed: {error_message}",
            "missing_rows": [],
            "extra_rows": [],
            "row_count_actual": 0,
            "row_count_expected": 0,
        }

    if actual_results is None:
        return {
            "error_type": "execution_failed",
            "message": "Query execution returned no results object",
            "missing_rows": [],
            "extra_rows": [],
            "row_count_actual": 0,
            "row_count_expected": 0,
        }

    # Canonicalize results for comparison
    actual_canonical = canonicalize_results(actual_results)
    expected_canonical = canonicalize_results(expected_results)

    if actual_canonical is None or expected_canonical is None:
        return {
            "error_type": "parse_error",
            "message": "Could not parse query results for comparison",
            "missing_rows": [],
            "extra_rows": [],
            "row_count_actual": 0,
            "row_count_expected": 0,
        }

    actual_count = sum(actual_canonical.values())
    expected_count = sum(expected_canonical.values())

    # Check for empty results
    if actual_count == 0 and expected_count > 0:
        return {
            "error_type": "empty_results",
            "message": f"Query returned 0 rows, but expected {expected_count} rows. "
            "Check if properties, classes, or filters are correct.",
            "missing_rows": [],
            "extra_rows": [],
            "row_count_actual": 0,
            "row_count_expected": expected_count,
        }

    # Calculate differences
    missing = expected_canonical - actual_canonical
    extra = actual_canonical - expected_canonical

    missing_count = sum(missing.values())
    extra_count = sum(extra.values())

    # Format sample rows for feedback
    missing_samples = [str(row) for row in list(missing.elements())[:3]]
    extra_samples = [str(row) for row in list(extra.elements())[:3]]

    if missing_count == 0 and extra_count == 0:
        return {
            "error_type": "success",
            "message": "Results match exactly",
            "missing_rows": [],
            "extra_rows": [],
            "row_count_actual": actual_count,
            "row_count_expected": expected_count,
        }

    message_parts = []
    if missing_count > 0:
        message_parts.append(f"Missing {missing_count} expected rows")
    if extra_count > 0:
        message_parts.append(f"Has {extra_count} extra rows that shouldn't be there")

    return {
        "error_type": "wrong_results",
        "message": ". ".join(message_parts) + ".",
        "missing_rows": missing_samples,
        "extra_rows": extra_samples,
        "row_count_actual": actual_count,
        "row_count_expected": expected_count,
    }


def _format_previous_attempts(attempts: list[dict]) -> str:
    """Format previous attempts for the prompt."""
    if not attempts:
        return ""

    lines = ["PREVIOUS ATTEMPTS (FAILED - DO NOT REPEAT):"]
    for attempt in attempts[-3:]:  # Show last 3 attempts
        lines.append(f"\n--- Attempt {attempt['iteration']} ---")
        lines.append(f"Query:\n```sparql\n{attempt['query'][:500]}{'...' if len(attempt['query']) > 500 else ''}\n```")
        feedback = attempt.get("feedback", {})
        lines.append(f"Result: {feedback.get('error_type', 'unknown')}")
        lines.append(f"Feedback: {feedback.get('message', 'No feedback')}")

    return "\n".join(lines)


def _format_feedback_section(feedback: dict | None) -> str:
    """Format the latest feedback for the prompt."""
    if not feedback:
        return ""

    lines = ["LATEST FEEDBACK:"]
    lines.append(f"- Error Type: {feedback.get('error_type', 'unknown')}")
    lines.append(f"- {feedback.get('message', 'No message')}")

    if feedback.get("row_count_expected"):
        lines.append(
            f"- Expected {feedback['row_count_expected']} rows, "
            f"got {feedback.get('row_count_actual', 0)} rows"
        )

    if feedback.get("missing_rows"):
        lines.append("- Sample missing rows:")
        for row in feedback["missing_rows"][:2]:
            lines.append(f"    {row}")

    if feedback.get("extra_rows"):
        lines.append("- Sample extra rows (shouldn't be there):")
        for row in feedback["extra_rows"][:2]:
            lines.append(f"    {row}")

    return "\n".join(lines)


async def correct_query_iteratively(
    generated_query: str,
    groundtruth_query: str,
    groundtruth_results: dict,
    llm_instance: LLMInstance,
    execute_fn,  # async function: (query: str) -> (success: bool, results: dict | None, error_type: str | None, error_msg: str | None)
    user_query: str = "",
    max_iterations: int = 5,
    request_id: str = "",
) -> dict[str, Any]:
    """
    Iteratively correct a query until results match groundtruth.

    Args:
        generated_query: The generated SPARQL query to correct
        groundtruth_query: The correct SPARQL query (reference)
        groundtruth_results: Expected results from executing groundtruth
        llm_instance: The LLM instance to use for correction
        execute_fn: Async function to execute SPARQL queries
        user_query: Optional natural language query for context
        max_iterations: Maximum number of correction attempts
        request_id: Optional request ID for logging

    Returns:
        Dictionary with:
            - corrected_query: The final corrected query
            - success: True if results match groundtruth
            - iterations: Number of iterations used
            - attempts: List of all attempts with feedback
            - token_usage: Total token usage across all iterations
    """
    log_prefix = f"[{request_id}]" if request_id else ""
    logger.info(f"{log_prefix} Starting iterative query correction (max {max_iterations} iterations)")

    attempts: list[dict] = []
    total_tokens = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    user_context = f"USER QUESTION: {user_query}\n" if user_query else ""

    # Iterations 1 to max_iterations-1: Try to correct the query
    for iteration in range(1, max_iterations):
        logger.info(f"{log_prefix} Correction iteration {iteration}/{max_iterations}")

        # Build the prompt
        if iteration == 1:
            iteration_context = "This is your first attempt."
            iteration_instruction = "Analyze the differences carefully before making changes."
            previous_section = ""
            feedback_section = ""
        else:
            iteration_context = f"This is attempt {iteration}. Previous attempts failed."
            iteration_instruction = "Do NOT repeat previous queries. Try a DIFFERENT approach based on the feedback."
            previous_section = _format_previous_attempts(attempts)
            last_feedback = attempts[-1].get("feedback") if attempts else None
            feedback_section = _format_feedback_section(last_feedback)

        prompt = CORRECTION_PROMPT_WITH_FEEDBACK.format(
            iteration_context=iteration_context,
            generated_query=generated_query,
            groundtruth_query=groundtruth_query,
            user_context=user_context,
            previous_attempts_section=previous_section,
            feedback_section=feedback_section,
            iteration_specific_instruction=iteration_instruction,
        )

        # Call LLM for correction
        try:
            llm_response: LLMResponse = await asyncio.to_thread(
                llm_instance.invoke_with_usage,
                prompt,
                CORRECTION_SYSTEM_MESSAGE,
            )

            total_tokens["input_tokens"] += llm_response.input_tokens
            total_tokens["output_tokens"] += llm_response.output_tokens
            total_tokens["total_tokens"] += llm_response.total_tokens

            parsed = robust_json_parse(llm_response.content, request_id)

            if not parsed:
                logger.warning(f"{log_prefix} Failed to parse LLM response, skipping iteration")
                attempts.append({
                    "iteration": iteration,
                    "query": "",
                    "feedback": {"error_type": "parse_error", "message": "Could not parse LLM response"},
                })
                continue

            corrected_query = _clean_query_response(parsed.get("corrected_query", ""))
            reasoning = parsed.get("reasoning", "")
            variable_mapping = parsed.get("variable_mapping", {})

            if not corrected_query:
                logger.warning(f"{log_prefix} Empty corrected query, skipping iteration")
                attempts.append({
                    "iteration": iteration,
                    "query": "",
                    "feedback": {"error_type": "empty_query", "message": "LLM returned empty query"},
                })
                continue

        except Exception as e:
            logger.error(f"{log_prefix} LLM error in iteration {iteration}: {e}")
            attempts.append({
                "iteration": iteration,
                "query": "",
                "feedback": {"error_type": "llm_error", "message": str(e)},
            })
            continue

        # Execute the corrected query
        try:
            success, results, error_type, error_msg = await execute_fn(corrected_query)
        except Exception as e:
            logger.error(f"{log_prefix} Execution error in iteration {iteration}: {e}")
            success, results, error_type, error_msg = False, None, "execution_error", str(e)

        # Generate feedback
        if success and results:
            feedback = generate_correction_feedback(results, groundtruth_results)
        else:
            feedback = generate_correction_feedback(None, groundtruth_results, error_type, error_msg)

        # Store attempt
        attempts.append({
            "iteration": iteration,
            "query": corrected_query,
            "reasoning": reasoning,
            "variable_mapping": variable_mapping,
            "feedback": feedback,
            "results_count": feedback.get("row_count_actual", 0),
        })

        # Check for success
        if feedback.get("error_type") == "success":
            logger.info(f"{log_prefix} Correction successful after {iteration} iterations")
            return {
                "corrected_query": corrected_query,
                "success": True,
                "iterations": iteration,
                "attempts": attempts,
                "token_usage": total_tokens,
            }

        logger.info(
            f"{log_prefix} Iteration {iteration} failed: {feedback.get('error_type')} - "
            f"{feedback.get('message', '')[:100]}"
        )

    # ==========================================================================
    # FINAL ITERATION: Variable normalization only
    # Take groundtruth and only normalize variable names to match generated query
    # ==========================================================================
    logger.info(
        f"{log_prefix} Final iteration ({max_iterations}): "
        f"Normalizing groundtruth variable names only"
    )

    prompt = FINAL_ITERATION_NORMALIZATION_PROMPT.format(
        generated_query=generated_query,
        groundtruth_query=groundtruth_query,
        user_context=user_context,
    )

    try:
        llm_response = await asyncio.to_thread(
            llm_instance.invoke_with_usage,
            prompt,
            CORRECTION_SYSTEM_MESSAGE,
        )

        total_tokens["input_tokens"] += llm_response.input_tokens
        total_tokens["output_tokens"] += llm_response.output_tokens
        total_tokens["total_tokens"] += llm_response.total_tokens

        parsed = robust_json_parse(llm_response.content, request_id)

        if parsed:
            normalized_query = _clean_query_response(parsed.get("corrected_query", ""))
            variable_mapping = parsed.get("variable_mapping", {})

            if normalized_query:
                attempts.append({
                    "iteration": max_iterations,
                    "query": normalized_query,
                    "reasoning": "Final iteration: variable normalization only",
                    "variable_mapping": variable_mapping,
                    "feedback": {"error_type": "fallback_normalized", "message": "Groundtruth with normalized variable names"},
                })

                logger.info(
                    f"{log_prefix} Returning normalized groundtruth with variable mapping: {variable_mapping}"
                )

                return {
                    "corrected_query": normalized_query,
                    "success": False,  # Still a fallback, not a successful correction
                    "iterations": max_iterations,
                    "attempts": attempts,
                    "token_usage": total_tokens,
                }

    except Exception as e:
        logger.error(f"{log_prefix} Final normalization failed: {e}")

    # Absolute fallback - return groundtruth as-is
    logger.warning(
        f"{log_prefix} All iterations failed including final normalization. "
        f"Returning groundtruth as-is."
    )

    return {
        "corrected_query": groundtruth_query,
        "success": False,
        "iterations": max_iterations,
        "attempts": attempts,
        "token_usage": total_tokens,
    }
