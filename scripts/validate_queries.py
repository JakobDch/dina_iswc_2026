"""
Validate Use-Case Queries against OnTop Endpoints.

This script:
1. Executes each ground-truth SPARQL query against the appropriate endpoint
2. Records success/failure and result counts
3. Generates a validation report
"""

import json
import requests
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.queries.use_case_queries import (
    ALL_QUERIES,
    SET_A_QUERIES,
    SET_B_QUERIES,
    SET_C_QUERIES,
    SET_D_QUERIES,
    SET_E_QUERIES,
    UseCaseQuery,
    SPARQLComplexity,
    InstanceMatchLevel,
)

# Endpoint configuration
ENDPOINTS = {
    "EDU": "http://localhost:8080/sparql",
    "TRN": "http://localhost:8081/sparql",
    "NRG": "http://localhost:8082/sparql",
    "BSBM": "http://localhost:8083/sparql",
    "BGEE": "http://localhost:8084/sparql",
    "LCA": "http://localhost:8085/sparql",
}


@dataclass
class QueryResult:
    """Result of executing a query."""
    query_id: str
    query_set: str
    dataset: str
    nl_query: str
    sparql_query: str
    success: bool
    result_count: int
    error_message: str
    execution_time_ms: float
    sample_results: list
    matches_intent: str  # "yes", "no", "partial", "needs_review"
    notes: str


def execute_sparql(endpoint: str, query: str, timeout: int = 30) -> tuple[bool, list, str, float]:
    """Execute SPARQL query against endpoint."""
    import time

    headers = {
        "Accept": "application/sparql-results+json",
        "Content-Type": "application/x-www-form-urlencoded",
    }

    start = time.time()
    try:
        response = requests.post(
            endpoint,
            data={"query": query},
            headers=headers,
            timeout=timeout,
        )
        elapsed = (time.time() - start) * 1000

        if response.status_code == 200:
            try:
                data = response.json()
                bindings = data.get("results", {}).get("bindings", [])
                return True, bindings, "", elapsed
            except json.JSONDecodeError as e:
                return False, [], f"JSON decode error: {e}", elapsed
        else:
            return False, [], f"HTTP {response.status_code}: {response.text[:200]}", elapsed

    except requests.exceptions.Timeout:
        elapsed = (time.time() - start) * 1000
        return False, [], "Query timeout", elapsed
    except requests.exceptions.ConnectionError as e:
        elapsed = (time.time() - start) * 1000
        return False, [], f"Connection error: {e}", elapsed
    except Exception as e:
        elapsed = (time.time() - start) * 1000
        return False, [], f"Unexpected error: {e}", elapsed


def get_endpoint_for_query(query: UseCaseQuery) -> str | None:
    """Get the appropriate endpoint for a query."""
    # For single-dataset queries, use that dataset's endpoint
    if len(query.datasets) == 1:
        return ENDPOINTS.get(query.datasets[0])

    # For cross-dataset queries, return None (needs special handling)
    return None


def validate_query(query: UseCaseQuery) -> QueryResult:
    """Validate a single query."""
    endpoint = get_endpoint_for_query(query)

    # Handle cross-dataset queries
    if endpoint is None:
        if query.query_set == "D":
            return QueryResult(
                query_id=query.id,
                query_set=query.query_set,
                dataset=query.dataset,
                nl_query=query.query,
                sparql_query=query.expected_sparql[:200] + "...",
                success=False,
                result_count=0,
                error_message="Cross-dataset query - requires federation (SERVICE clause)",
                execution_time_ms=0,
                sample_results=[],
                matches_intent="needs_review",
                notes="Federation queries need special handling",
            )
        else:
            return QueryResult(
                query_id=query.id,
                query_set=query.query_set,
                dataset=query.dataset,
                nl_query=query.query,
                sparql_query=query.expected_sparql[:200] + "...",
                success=False,
                result_count=0,
                error_message="No endpoint for multi-dataset query",
                execution_time_ms=0,
                sample_results=[],
                matches_intent="needs_review",
                notes="",
            )

    # Execute query
    success, results, error, elapsed = execute_sparql(endpoint, query.expected_sparql)

    # Get sample results (first 3)
    sample = []
    for r in results[:3]:
        sample.append({k: v.get("value", "") for k, v in r.items()})

    # Determine if results match intent
    matches_intent = "needs_review"
    if success:
        if len(results) > 0:
            matches_intent = "yes"  # Has results, likely correct
        else:
            matches_intent = "partial"  # Query works but no results

    return QueryResult(
        query_id=query.id,
        query_set=query.query_set,
        dataset=query.dataset,
        nl_query=query.query,
        sparql_query=query.expected_sparql[:500] + ("..." if len(query.expected_sparql) > 500 else ""),
        success=success,
        result_count=len(results),
        error_message=error,
        execution_time_ms=round(elapsed, 2),
        sample_results=sample,
        matches_intent=matches_intent,
        notes="",
    )


def validate_all_queries() -> list[QueryResult]:
    """Validate all queries and return results."""
    results = []

    print("Validating queries...")
    print("=" * 60)

    for i, query in enumerate(ALL_QUERIES):
        print(f"[{i+1}/{len(ALL_QUERIES)}] {query.id}: {query.query[:50]}...")
        result = validate_query(query)
        results.append(result)

        status = "OK" if result.success else "FAIL"
        count = result.result_count if result.success else result.error_message[:30]
        print(f"         {status} - {count}")

    return results


def generate_report(results: list[QueryResult]) -> str:
    """Generate a markdown report."""
    lines = []
    lines.append("# Query Validation Report")
    lines.append(f"\nGenerated: {datetime.now().isoformat()}")
    lines.append(f"\nTotal queries: {len(results)}")

    # Summary statistics
    successful = sum(1 for r in results if r.success)
    failed = len(results) - successful
    with_results = sum(1 for r in results if r.success and r.result_count > 0)
    empty_results = sum(1 for r in results if r.success and r.result_count == 0)

    lines.append("\n## Summary")
    lines.append(f"- Successful executions: {successful} ({100*successful/len(results):.1f}%)")
    lines.append(f"- Failed executions: {failed} ({100*failed/len(results):.1f}%)")
    lines.append(f"- Queries with results: {with_results}")
    lines.append(f"- Queries with empty results: {empty_results}")

    # Per-set statistics
    lines.append("\n## Results by Query Set")
    for query_set in ["A", "B", "C", "D", "E"]:
        set_results = [r for r in results if r.query_set == query_set]
        if not set_results:
            continue

        set_success = sum(1 for r in set_results if r.success)
        set_with_results = sum(1 for r in set_results if r.success and r.result_count > 0)

        set_names = {
            "A": "Single-Mapping Baseline",
            "B": "Multi-Mapping Challenge",
            "C": "Instance Value Matching",
            "D": "Cross-Dataset",
            "E": "Semantic Variance",
        }

        lines.append(f"\n### Set {query_set}: {set_names.get(query_set, '')}")
        lines.append(f"- Total: {len(set_results)}")
        lines.append(f"- Successful: {set_success} ({100*set_success/len(set_results):.1f}%)")
        lines.append(f"- With results: {set_with_results}")

    # Detailed results
    lines.append("\n## Detailed Results")

    for query_set in ["A", "B", "C", "D", "E"]:
        set_results = [r for r in results if r.query_set == query_set]
        if not set_results:
            continue

        lines.append(f"\n### Set {query_set}")
        lines.append("")
        lines.append("| ID | Status | Results | Time (ms) | NL Query | Notes |")
        lines.append("|-----|--------|---------|-----------|----------|-------|")

        for r in set_results:
            status = "OK" if r.success else "FAIL"
            notes = r.error_message[:30] if r.error_message else ""
            nl_short = r.nl_query[:40] + "..." if len(r.nl_query) > 40 else r.nl_query
            lines.append(f"| {r.query_id} | {status} | {r.result_count} | {r.execution_time_ms} | {nl_short} | {notes} |")

    # Failed queries detail
    failed_results = [r for r in results if not r.success]
    if failed_results:
        lines.append("\n## Failed Queries Detail")
        for r in failed_results:
            lines.append(f"\n### {r.query_id}")
            lines.append(f"**NL Query:** {r.nl_query}")
            lines.append(f"**Error:** {r.error_message}")
            lines.append(f"**SPARQL:**")
            lines.append("```sparql")
            lines.append(r.sparql_query)
            lines.append("```")

    # Sample results for successful queries
    lines.append("\n## Sample Results (First 3 per Query)")
    for r in results:
        if r.success and r.result_count > 0 and r.sample_results:
            lines.append(f"\n### {r.query_id}: {r.nl_query[:60]}...")
            lines.append(f"Total results: {r.result_count}")
            lines.append("```json")
            lines.append(json.dumps(r.sample_results, indent=2))
            lines.append("```")

    return "\n".join(lines)


def main():
    """Main entry point."""
    print("Query Validation Script")
    print("=" * 60)

    # Check endpoints
    print("\nChecking endpoints...")
    for name, url in ENDPOINTS.items():
        try:
            response = requests.get(url.replace("/sparql", ""), timeout=5)
            status = "OK" if response.status_code in [200, 404] else f"HTTP {response.status_code}"
        except Exception as e:
            status = f"FAIL: {e}"
        print(f"  {name}: {status}")

    print("\n")

    # Validate queries
    results = validate_all_queries()

    # Generate report
    report = generate_report(results)

    # Save report
    output_dir = Path(__file__).parent.parent / "data" / "queries" / "ground_truth"
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / "validation_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nReport saved to: {report_path}")

    # Save detailed results as JSON
    json_path = output_dir / "validation_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"JSON results saved to: {json_path}")

    # Print summary
    successful = sum(1 for r in results if r.success)
    print(f"\n{'='*60}")
    print(f"SUMMARY: {successful}/{len(results)} queries executed successfully")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()