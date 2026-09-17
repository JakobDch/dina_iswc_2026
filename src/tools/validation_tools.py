"""
Validation tools for SPARQL query results.

Includes:
- Result comparison and agreement calculation
- Confidence scoring
- Query selection
- SPARQL-to-NL back-translation for similarity validation
"""

import asyncio
from collections import Counter

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

from src.baseline.metrics import canonicalize_results
from src.config import get_settings


@tool
def compare_query_results(results_list: list[dict]) -> dict:
    """
    Compare multiple SPARQL query results to calculate variance.

    Args:
        results_list: List of SPARQL result JSON objects

    Returns:
        Dict with agreement_rate and result_groups
    """
    if not results_list:
        return {
            "agreement_rate": 0.0,
            "result_groups": [],
            "num_unique_results": 0,
        }

    # Canonicalize each result
    canonical_results = []
    for result in results_list:
        canonical = canonicalize_results(result)
        if canonical is not None:
            # Convert Counter to hashable tuple for comparison
            canonical_tuple = tuple(sorted(canonical.items()))
            canonical_results.append(canonical_tuple)

    if not canonical_results:
        return {
            "agreement_rate": 0.0,
            "result_groups": [],
            "num_unique_results": 0,
        }

    # Count occurrences of each unique result
    result_counts = Counter(canonical_results)
    num_unique = len(result_counts)
    total_results = len(canonical_results)

    # Agreement rate: proportion of results matching the most common one
    most_common_count = result_counts.most_common(1)[0][1]
    agreement_rate = most_common_count / total_results

    # Group results
    result_groups = [
        {"count": count, "proportion": count / total_results}
        for _, count in result_counts.most_common()
    ]

    return {
        "agreement_rate": round(agreement_rate, 4),
        "result_groups": result_groups,
        "num_unique_results": num_unique,
    }


@tool
def calculate_confidence(
    results_list: list[dict],
    syntax_valid_list: list[bool],
) -> float:
    """
    Calculate confidence score based on result agreement and syntax validity.

    Args:
        results_list: List of SPARQL result JSON objects
        syntax_valid_list: List of syntax validity booleans

    Returns:
        Confidence score between 0 and 1
    """
    if not results_list:
        return 0.0

    # Factor 1: Syntax validity rate
    valid_count = sum(1 for v in syntax_valid_list if v)
    syntax_rate = valid_count / len(syntax_valid_list)

    # Factor 2: Result agreement
    comparison = compare_query_results.invoke({"results_list": results_list})
    agreement_rate = comparison.get("agreement_rate", 0.0)

    # Factor 3: Penalty for having results (empty results might indicate errors)
    has_results = any(
        result.get("results", {}).get("bindings", [])
        for result in results_list
        if isinstance(result, dict)
    )
    result_penalty = 1.0 if has_results else 0.5

    # Combined confidence
    confidence = syntax_rate * 0.3 + agreement_rate * 0.5 + result_penalty * 0.2

    return round(confidence, 4)


@tool
def select_best_query(
    queries: list[str],
    results_list: list[dict],
    confidence_scores: list[float] | None = None,
) -> dict:
    """
    Select the best query from a list based on results and confidence.

    Args:
        queries: List of SPARQL queries
        results_list: List of execution results
        confidence_scores: Optional per-query confidence scores

    Returns:
        Dict with selected_index, query, and reasoning
    """
    if not queries:
        return {
            "selected_index": -1,
            "query": None,
            "reasoning": "No queries provided",
        }

    if len(queries) == 1:
        return {
            "selected_index": 0,
            "query": queries[0],
            "reasoning": "Only one query provided",
        }

    # Score each query
    scores = []
    for i, (query, result) in enumerate(zip(queries, results_list)):
        score = 0.0

        # Syntax validity
        if isinstance(result, dict) and "error_details" not in result:
            score += 0.3

        # Has results
        if isinstance(result, dict):
            bindings = result.get("results", {}).get("bindings", [])
            if bindings:
                score += 0.3
                # Prefer queries with reasonable result sizes
                if 1 <= len(bindings) <= 100:
                    score += 0.2

        # Use provided confidence if available
        if confidence_scores and i < len(confidence_scores):
            score += confidence_scores[i] * 0.2

        scores.append(score)

    # Select highest scoring query
    best_index = max(range(len(scores)), key=lambda i: scores[i])

    return {
        "selected_index": best_index,
        "query": queries[best_index],
        "score": scores[best_index],
        "reasoning": f"Selected query {best_index + 1} with score {scores[best_index]:.2f}",
    }


# --- SPARQL-to-NL Back-Translation ---

BACK_TRANSLATION_PROMPT = """You are an expert at explaining SPARQL queries in natural language.

Given a SPARQL query, provide a clear, concise natural language description of what the query is asking for.
Focus on:
1. What data is being retrieved (the SELECT variables)
2. What conditions/filters are applied
3. How entities are related

Keep the explanation simple and focused on the intent, not the technical syntax.
Output ONLY the natural language description, nothing else."""


SIMILARITY_ASSESSMENT_PROMPT = """You are comparing two natural language descriptions to assess their semantic similarity.

Original Query: {original}

Back-translated Query: {backtranslated}

Assess how well the back-translated description captures the intent of the original query.

Output a JSON object with:
- similarity_score: A float between 0 and 1 (1 = perfect match, 0 = completely different)
- matching_aspects: List of aspects that match
- missing_aspects: List of aspects from original that are missing in back-translation
- extra_aspects: List of aspects in back-translation not in original
- assessment: Brief explanation of the similarity

Output ONLY valid JSON."""


class SPARQLBackTranslator:
    """Translates SPARQL queries back to natural language for validation."""

    def __init__(self, llm_model: str = "deepseek-chat"):
        """Initialize the back translator."""
        settings = get_settings()

        if "claude" in llm_model.lower():
            self.llm = ChatAnthropic(
                model=llm_model,
                api_key=settings.anthropic_api_key,
            )
        else:
            self.llm = ChatOpenAI(
                model=llm_model,
                api_key=settings.openai_api_key,
            )

    async def translate_to_nl(self, sparql_query: str) -> str:
        """
        Translate a SPARQL query to natural language.

        Args:
            sparql_query: The SPARQL query to translate

        Returns:
            Natural language description of the query
        """
        messages = [
            SystemMessage(content=BACK_TRANSLATION_PROMPT),
            HumanMessage(content=f"SPARQL Query:\n{sparql_query}"),
        ]

        try:
            response = await self.llm.ainvoke(messages)
            return response.content.strip()
        except Exception as e:
            return f"Error translating query: {e}"

    async def assess_similarity(
        self,
        original_query: str,
        backtranslated_description: str,
    ) -> dict:
        """
        Assess similarity between original and back-translated queries.

        Args:
            original_query: Original natural language query
            backtranslated_description: Back-translated description from SPARQL

        Returns:
            Dict with similarity assessment
        """
        prompt = SIMILARITY_ASSESSMENT_PROMPT.format(
            original=original_query,
            backtranslated=backtranslated_description,
        )

        messages = [
            HumanMessage(content=prompt),
        ]

        try:
            response = await self.llm.ainvoke(messages)
            import json
            return json.loads(response.content)
        except Exception as e:
            return {
                "similarity_score": 0.0,
                "matching_aspects": [],
                "missing_aspects": [],
                "extra_aspects": [],
                "assessment": f"Error assessing similarity: {e}",
            }


# Singleton instance
_back_translator: SPARQLBackTranslator | None = None


def get_back_translator(llm_model: str = "deepseek-chat") -> SPARQLBackTranslator:
    """Get or create the back translator singleton."""
    global _back_translator
    if _back_translator is None:
        _back_translator = SPARQLBackTranslator(llm_model)
    return _back_translator


@tool
def translate_sparql_to_nl(sparql_query: str) -> str:
    """
    Translate a SPARQL query to natural language description.

    Use this to verify that a generated SPARQL query correctly captures
    the intent of the original natural language question.

    Args:
        sparql_query: The SPARQL query to translate

    Returns:
        Natural language description of what the query does

    Example:
        translate_sparql_to_nl("SELECT ?name WHERE { ?s eduo:name ?name }")
        -> "Find all names in the dataset"
    """
    translator = get_back_translator()
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(translator.translate_to_nl(sparql_query))
    finally:
        loop.close()


@tool
def validate_query_intent(
    original_nl_query: str,
    sparql_query: str,
) -> dict:
    """
    Validate that a SPARQL query correctly captures the original query intent.

    This tool:
    1. Translates the SPARQL query back to natural language
    2. Compares it with the original query
    3. Returns a similarity assessment

    Args:
        original_nl_query: The original natural language query from the user
        sparql_query: The generated SPARQL query to validate

    Returns:
        Dict with similarity_score (0-1), matching/missing aspects, and assessment

    Example:
        validate_query_intent(
            "Find all graduate students",
            "SELECT ?s WHERE { ?s a eduo:DoctoralCandidate }"
        )
    """
    translator = get_back_translator()
    loop = asyncio.new_event_loop()
    try:
        # Step 1: Back-translate the SPARQL
        nl_description = loop.run_until_complete(
            translator.translate_to_nl(sparql_query)
        )

        # Step 2: Assess similarity
        assessment = loop.run_until_complete(
            translator.assess_similarity(original_nl_query, nl_description)
        )

        return {
            "back_translated": nl_description,
            **assessment,
        }
    finally:
        loop.close()


@tool
def calculate_intent_confidence(
    original_nl_query: str,
    queries: list[str],
) -> list[dict]:
    """
    Calculate intent confidence for multiple SPARQL query variants.

    Useful for selecting the best query based on how well it captures
    the original intent.

    Args:
        original_nl_query: The original natural language query
        queries: List of generated SPARQL queries

    Returns:
        List of dicts with query index, similarity_score, and assessment
    """
    results = []
    for i, query in enumerate(queries):
        validation = validate_query_intent.invoke({
            "original_nl_query": original_nl_query,
            "sparql_query": query,
        })
        results.append({
            "query_index": i,
            "similarity_score": validation.get("similarity_score", 0.0),
            "back_translated": validation.get("back_translated", ""),
            "assessment": validation.get("assessment", ""),
        })

    return results


VALIDATION_TOOLS = [
    compare_query_results,
    calculate_confidence,
    select_best_query,
    translate_sparql_to_nl,
    validate_query_intent,
    calculate_intent_confidence,
]

__all__ = [
    "compare_query_results",
    "calculate_confidence",
    "select_best_query",
    "translate_sparql_to_nl",
    "validate_query_intent",
    "calculate_intent_confidence",
    "SPARQLBackTranslator",
    "get_back_translator",
    "VALIDATION_TOOLS",
]
