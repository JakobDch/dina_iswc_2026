"""
Formatting functions for SPARQL pipeline prompts and outputs.

Provides functions to format decomposition results, instance search data,
and other pipeline outputs for LLM consumption and logging.
"""

import logging
import re
from typing import Optional, List, Dict, Any

from rdflib import Graph, RDF

logger = logging.getLogger(__name__)


def transform_graph_with_literals(graph: Graph) -> Graph:
    """
    Transform an RDF graph by replacing instances with their types.

    Creates a semantic structure view showing Class + Property + Literal
    relationships instead of instance-level details.
    """
    type_mapping = {}

    for (sub, _, obj) in graph.triples((None, RDF.type, None)):
        type_mapping[sub] = obj

    transformed_graph = Graph()

    for ns in graph.namespaces():
        transformed_graph.namespace_manager.bind(ns[0], ns[1])

    for (sub, pred, obj) in graph.triples((None, None, None)):
        if pred == RDF.type:
            continue

        if sub in type_mapping:
            sub = type_mapping[sub]
        else:
            continue

        if obj in type_mapping:
            obj = type_mapping[obj]

        transformed_graph.add((sub, pred, obj))

    return transformed_graph


def _format_triple_change(change_detail: str) -> str:
    """Format a triple change for better readability by shortening URIs."""
    uri_regex = re.compile(r"<[^>]*[/#]([^/>]+)>")

    def shorten_uris(text: str) -> str:
        return uri_regex.sub(r":\1", text)

    if "' -> '" in change_detail:
        old, new = change_detail.split("' -> '")
        old_short = shorten_uris(old.strip("'"))
        new_short = shorten_uris(new.strip("'"))
        return f"  - FROM: {old_short}\n    TO:   {new_short}"

    return f"  - {shorten_uris(change_detail)}"


def _format_evaluation_result(result: dict) -> str:
    """Format evaluation result with structure and readability."""
    score_percent = "N/A"
    if result.get('max_edit_cost', 0) > 0:
        score_percent = f"{(1 - (result.get('actual_edit_cost', 0) / result['max_edit_cost'])) * 100:.1f}%"

    log_output = [
        "\n" + "=" * 80,
        " SPARQL QUERY EVALUATION RESULT ".center(80, "="),
        "=" * 80,
        f"\n{'Edit Score:':<20} {result.get('actual_edit_cost', 'N/A')} / {result.get('max_edit_cost', 'N/A')} ({score_percent})",
        f"{'Query Correct:':<20} {'✓' if result.get('actual_edit_cost') == 0 else '✗'}"
    ]

    if result.get('cost_breakdown'):
        log_output.extend([
            "\n" + "MAX EDIT COST BREAKDOWN".center(80, "-"),
            f"{'Component':<30} {'Cost':<10} {'Details'}",
            "-" * 80
        ])
        for comp in result['cost_breakdown']:
            log_output.append(f"{comp.get('component', ''):<30} {comp.get('cost', 0):<10} {comp.get('detail', '')}")

    if result.get('corrected_query'):
        log_output.extend([
            "\n" + "CORRECTED QUERY".center(80, "-"),
            result['corrected_query'],
            "-" * 80
        ])

    log_output.append("=" * 80 + "\n")
    return "\n".join(log_output)


def format_hierarchical_decomposition_for_llm(
    decomposition_results: Dict[str, Any],
    user_query: str,
    request_id: str,
    generated_sparql_query: str = ""
) -> str:
    """
    Format hierarchical decomposition results for LLM analysis.

    Constructs a structured prompt showing pattern ablation results,
    filter effects, join overlap verification, and sample data.
    """
    from ..operator_hints import get_hints_for_query

    has_union = decomposition_results.get("has_union", False)

    if has_union:
        num_union_blocks = decomposition_results.get("num_union_blocks", 0)
        union_block_results = decomposition_results.get("union_block_results", [])

        prompt = f"""**QUERY STRUCTURE:**
Query contains UNION with {num_union_blocks} separate blocks. Each block was analyzed independently.

"""

        for idx, block_result in enumerate(union_block_results):
            block_idx = idx + 1
            prompt += f"\n{'=' * 60}\n"
            prompt += f"**UNION BLOCK {block_idx}/{num_union_blocks}:**\n"
            prompt += f"{'=' * 60}\n\n"
            prompt += _format_decomposition_steps(block_result)

        prompt += f"\n{'=' * 60}\n\n"

        if generated_sparql_query:
            operator_hints_text = get_hints_for_query(generated_sparql_query)
            if operator_hints_text:
                prompt += "=" * 60 + "\n"
                prompt += "**OPERATOR-SPECIFIC HINTS:**\n"
                prompt += operator_hints_text
                prompt += "=" * 60 + "\n\n"

        return prompt

    prompt = ""
    prompt += _format_decomposition_steps(decomposition_results)

    if generated_sparql_query:
        operator_hints_text = get_hints_for_query(generated_sparql_query)
        if operator_hints_text:
            prompt += "\n" + "=" * 60 + "\n"
            prompt += "**OPERATOR-SPECIFIC HINTS:**\n"
            prompt += operator_hints_text
            prompt += "=" * 60 + "\n\n"

    return prompt


def _format_decomposition_steps(decomposition_results: Dict[str, Any]) -> str:
    """Format pattern-ablation diagnostic results."""
    prompt = ""

    has_union = decomposition_results.get("has_union", False)

    if has_union:
        prompt += "**STEP 0 - UNION QUERY DETECTED:**\n"
        num_blocks = decomposition_results.get("num_union_blocks", 0)
        prompt += f"Query was split into {num_blocks} separate UNION blocks.\n"
        prompt += "Analyzing each block independently:\n\n"

        union_block_results = decomposition_results.get("union_block_results", [])

        for block_result in union_block_results:
            block_idx = block_result.get("union_block_index", "?")
            prompt += f"{'=' * 60}\n"
            prompt += f"UNION BLOCK #{block_idx}\n"
            prompt += f"{'=' * 60}\n\n"

            block_formatted = _format_single_block_diagnostics(block_result)
            prompt += block_formatted
            prompt += "\n"

        return prompt

    return _format_single_block_diagnostics(decomposition_results)


def _format_single_block_diagnostics(decomposition_results: Dict[str, Any]) -> str:
    """Format diagnostics for a single query block."""
    prompt = ""

    ablation = decomposition_results.get("pattern_ablation", {})
    if ablation:
        prompt += "**STEP 1 - INCREMENTAL PATTERN ABLATION:**\n"
        prompt += "(Building query incrementally: P1, then P1+P2, then P1+P2+P3, etc. to find problematic patterns)\n\n"

        ablation_error = ablation.get("error")
        if ablation_error:
            prompt += f"  ⚠️ ERROR: {ablation_error}\n\n"
        else:
            incremental_tests = ablation.get("incremental_tests", [])

            for test in incremental_tests:
                step = test.get("step")
                count = test.get("count")
                status = test.get("status")
                patterns_tested = test.get("patterns_tested", [])

                if status == "killer":
                    prompt += f"  ❌ Step {step}: COUNT = {count} ← KILLER PATTERN!\n"
                elif status == "ok":
                    prompt += f"  ✓ Step {step}: COUNT = {count}\n"
                else:
                    error_msg = test.get('error_message', 'Unknown')[:500]
                    prompt += f"  ⚠️ Step {step}: ERROR - {error_msg}\n"

                prompt += f"     Tested patterns:\n"
                for i, pattern in enumerate(patterns_tested, 1):
                    pattern_display = pattern[:120] + "..." if len(pattern) > 120 else pattern
                    if i == len(patterns_tested):
                        prompt += f"       {i}. → {pattern_display}\n"
                    else:
                        prompt += f"       {i}.   {pattern_display}\n"
                prompt += "\n"

            prompt += "\n"

    filters = decomposition_results.get("filter_effects", {})
    if filters and filters.get("has_filters"):
        prompt += "**STEP 2 - FILTER EFFECT ISOLATION:**\n"
        base_count = filters.get("base_count_without_filters", 0)
        prompt += f"Base query (NO filters): COUNT = {base_count}\n\n"

        filter_tests = filters.get("filter_tests", [])
        for i, test in enumerate(filter_tests, 1):
            filter_str = test.get("filter", "")
            count = test.get("count_with_filter", 0)
            blocks = test.get("blocks_results", False)

            if blocks:
                prompt += f"  ❌ Filter {i}: COUNT = {count} ← BLOCKS ALL RESULTS!\n"
                prompt += f"     {filter_str}\n"

                range_info = test.get("range_info")
                if range_info:
                    var = range_info.get("variable")
                    min_val = range_info.get("min")
                    max_val = range_info.get("max")
                    prompt += f"     Observed range for {var}: {min_val} to {max_val}\n"
                    prompt += f"     → Filter is outside actual data range!\n"
            else:
                prompt += f"  ✓ Filter {i}: COUNT = {count}\n"
                prompt += f"     {filter_str[:80]}\n"

        prompt += "\n"
    elif filters:
        prompt += "**STEP 2 - FILTER EFFECT ISOLATION:** No filters in query\n\n"

    joins = decomposition_results.get("join_overlap", {})
    if joins and joins.get("join_variables"):
        prompt += "**STEP 3 - JOIN KEY OVERLAP VERIFICATION:**\n"
        prompt += "(Join variables are variables used in multiple triple patterns to connect data.\n"
        prompt += "This checks if these variables have any matching values across the patterns they appear in.)\n\n"

        join_tests = joins.get("join_tests", [])
        for test in join_tests:
            var = test.get("variable")
            overlap = test.get("overlap_count", 0)
            has_overlap = test.get("has_overlap", False)

            if not has_overlap:
                prompt += f"  ❌ {var}: ZERO OVERLAP - No matching values found!\n"
                warning = test.get("warning", "")
                if warning:
                    prompt += f"     → {warning}\n"
            else:
                prompt += f"  ✓ {var}: {overlap} matching values\n"

        prompt += "\n"
    elif joins:
        prompt += "**STEP 3 - JOIN KEY OVERLAP:** No join variables detected\n\n"

    samples = decomposition_results.get("mini_samples", {})
    sample_list = samples.get("samples", [])

    if sample_list:
        prompt += "**STEP 4 - SAMPLE DATA (showing actual values in database):**\n\n"

        for i, sample in enumerate(sample_list[:3], 1):
            resource = sample.get("resource", "")
            properties = sample.get("properties", {})

            prompt += f"Sample {i}:\n"
            prompt += f"  Resource: {resource}\n"
            prompt += f"  Properties:\n"

            for prop, value in list(properties.items())[:10]:
                value_str = str(value)[:100]
                prompt += f"    - {prop}: {value_str}\n"

            if len(properties) > 10:
                prompt += f"    ... ({len(properties) - 10} more properties)\n"
            prompt += "\n"

    else:
        prompt += "**STEP 4 - SAMPLE DATA:** No samples available\n\n"

    return prompt


def format_instance_search_for_prompt(instance_search_results: Optional[Dict[str, Any]]) -> str:
    """Format instance search results for the empty correction prompt."""
    if not instance_search_results:
        return "No instance data available for debugging."

    if instance_search_results.get("error"):
        return f"Instance search error: {instance_search_results.get('error')}"

    found_terms = instance_search_results.get("found_terms", [])
    results_by_term = instance_search_results.get("results_by_term", {})

    if not found_terms:
        return "No matching instances found in the data."

    formatted = "**CONCRETE INSTANCE DATA FROM DATABASE:**\n\n"
    formatted += "The following terms were found in the actual RDF data:\n\n"

    for term in found_terms[:5]:
        term_data = results_by_term.get(term, {})
        example_matches = term_data.get("example_matches", [])[:3]

        if example_matches:
            formatted += f"Search term: '{term}' ({term_data.get('total_matches', 0)} total matches)\n"
            for i, match in enumerate(example_matches, 1):
                semantic_structure = match.get('semantic_structure', '')
                if semantic_structure:
                    formatted += f"  Example {i}: {semantic_structure}\n"
            formatted += "\n"

    formatted += "**STRING VALUES FOUND IN DATA:**\n"
    formatted += "(Check exact format - whitespace matters!)\n\n"

    string_literals = {}
    for term_data in results_by_term.values():
        for match in term_data.get("example_matches", []):
            semantic_structure = match.get('semantic_structure', '')
            string_matches = re.findall(r'(\w+:\w+)\s+"([^"]+)"', semantic_structure)
            for prop, value in string_matches:
                if prop not in string_literals:
                    string_literals[prop] = set()
                string_literals[prop].add(value)

    for prop, values in list(string_literals.items())[:10]:
        values_list = list(values)[:3]
        formatted += f"  {prop}: {values_list}\n"

    return formatted


def format_few_shot_examples_for_prompt(adaptive_few_shot_examples: Optional[List[Dict[str, Any]]]) -> str:
    """Format adaptive few-shot examples for the empty correction prompt."""
    if not adaptive_few_shot_examples or len(adaptive_few_shot_examples) == 0:
        return "No successful example queries available for reference."

    formatted = "**SUCCESSFUL QUERY PATTERNS (from similar queries):**\n\n"
    formatted += "These queries successfully worked with the same semantic models:\n\n"

    for i, example in enumerate(adaptive_few_shot_examples[:3], 1):
        user_nl = example.get("user_nl", "N/A")
        sparql_query = example.get("sparql_query", "N/A")
        similarity_score = example.get("similarity_score", 0.0)

        formatted += f"Example {i} (similarity: {similarity_score:.2f}):\n"
        formatted += f"  User Query: {user_nl[:150]}...\n"
        formatted += f"  SPARQL Query (first 400 chars):\n"

        if len(sparql_query) > 400:
            formatted += f"    {sparql_query[:400]}...\n"
        else:
            formatted += f"    {sparql_query}\n"
        formatted += "\n"

    formatted += "**KEY PATTERNS TO NOTE:**\n"
    formatted += "- Check how type constraints are used (a eno:Class)\n"
    formatted += "- Check property connection patterns\n"
    formatted += "- Check string literal formatting\n"
    formatted += "- Check FILTER usage patterns\n"

    return formatted
