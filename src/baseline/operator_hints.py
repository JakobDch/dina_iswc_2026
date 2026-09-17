"""
Operator-specific hints for SPARQL query debugging.

This module provides functionality to match relevant operator hints based on
patterns detected in SPARQL queries and format them for inclusion in LLM prompts.
"""

import json
import re
import logging
from pathlib import Path
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

# Cache for loaded hints
_OPERATOR_HINTS_CACHE: Optional[List[Dict]] = None


def load_operator_hints() -> List[Dict]:
    """
    Load operator hints from JSON file.

    Returns:
        List of operator hint dictionaries, each containing:
        - id: Unique identifier for the hint
        - trigger_patterns: List of regex patterns to match against queries
        - structural_hint: Description of the problem
        - instruction: How to fix the problem
    """
    global _OPERATOR_HINTS_CACHE

    # Return cached hints if already loaded
    if _OPERATOR_HINTS_CACHE is not None:
        return _OPERATOR_HINTS_CACHE

    # Determine path to hints JSON file
    hints_file = Path(__file__).parent / "operator_hints.json"

    try:
        with open(hints_file, "r", encoding="utf-8") as f:
            hints = json.load(f)

        logger.info(f"Loaded {len(hints)} operator hints from {hints_file}")
        _OPERATOR_HINTS_CACHE = hints
        return hints

    except FileNotFoundError:
        logger.warning(f"Operator hints file not found: {hints_file}")
        return []

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse operator hints JSON: {e}")
        return []

    except Exception as e:
        logger.error(f"Unexpected error loading operator hints: {e}")
        return []


def match_relevant_hints(query: str) -> List[Dict]:
    """
    Match relevant operator hints based on patterns in the SPARQL query.

    Args:
        query: SPARQL query string to analyze

    Returns:
        List of matched hint dictionaries
    """
    if not query or not isinstance(query, str):
        return []

    all_hints = load_operator_hints()
    if not all_hints:
        return []

    matched_hints = []

    for hint in all_hints:
        hint_id = hint.get("id", "UNKNOWN")
        trigger_patterns = hint.get("trigger_patterns", [])

        # Check if any trigger pattern matches the query
        for pattern in trigger_patterns:
            try:
                # Use IGNORECASE and DOTALL for more flexible matching
                if re.search(pattern, query, re.IGNORECASE | re.DOTALL):
                    matched_hints.append(hint)
                    logger.debug(f"Matched hint '{hint_id}' via pattern: {pattern}")
                    break  # Don't check other patterns for this hint

            except re.error as e:
                logger.warning(f"Invalid regex pattern in hint '{hint_id}': {pattern} - {e}")
                continue

    logger.info(f"Matched {len(matched_hints)} operator hints for query")
    return matched_hints


def format_hints_for_prompt(hints: List[Dict]) -> str:
    """
    Format matched hints as a structured text block for LLM prompts.

    Args:
        hints: List of matched hint dictionaries

    Returns:
        Formatted string for inclusion in LLM prompt
    """
    if not hints:
        return ""

    lines = []
    lines.append("Based on the patterns detected in your query, watch out for these common issues:\n")

    for hint in hints:
        hint_id = hint.get("id", "UNKNOWN")
        structural_hint = hint.get("structural_hint", "No description available")
        instruction = hint.get("instruction", "No fix available")

        lines.append(f"[!] {hint_id}")
        lines.append(f"    Issue: {structural_hint}")
        lines.append(f"    Fix: {instruction}")
        lines.append("")  # Blank line between hints

    return "\n".join(lines)


def get_hints_for_query(query: str) -> str:
    """
    Convenience function to match and format hints in one call.

    Args:
        query: SPARQL query string to analyze

    Returns:
        Formatted hints string ready for prompt inclusion, or empty string if no matches
    """
    matched_hints = match_relevant_hints(query)
    return format_hints_for_prompt(matched_hints)
