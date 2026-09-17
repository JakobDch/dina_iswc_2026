"""Validate all SPARQL queries against their respective endpoints."""

import httpx
import sys
import re
sys.path.insert(0, ".")
from use_case_queries_tiered import ALL_TIERED_QUERIES

ENDPOINTS = {
    "EDU": "http://localhost:8080/sparql",
    "TRN": "http://localhost:8081/sparql",
    "NRG": "http://localhost:8082/sparql",
    "BSBM": "http://localhost:8083/sparql",
}

# Prefix to dataset mapping for cross-dataset queries
PREFIX_DATASET_MAP = {
    "eduo:": "EDU",
    "trn:": "TRN",
    "eno:": "NRG",
    "bsbm:": "BSBM",
}

def detect_dataset_from_query(sparql: str) -> str | None:
    """Detect which dataset a SPARQL query targets based on prefixes used."""
    for prefix, dataset in PREFIX_DATASET_MAP.items():
        if prefix in sparql:
            return dataset
    return None

def test_query(endpoint: str, sparql: str, timeout: float = 30.0) -> tuple[str, int, str]:
    """Test a SPARQL query and return (status, count, error_msg)."""
    try:
        response = httpx.post(
            endpoint,
            data={"query": sparql},
            headers={"Accept": "application/sparql-results+json"},
            timeout=timeout,
        )
        if response.status_code != 200:
            return "FAIL", 0, f"HTTP {response.status_code}"

        data = response.json()
        bindings = data.get("results", {}).get("bindings", [])
        count = len(bindings)

        if count == 0:
            return "EMPTY", 0, ""
        return "PASS", count, ""
    except httpx.TimeoutException:
        return "TIMEOUT", 0, "Query timeout"
    except Exception as e:
        return "FAIL", 0, str(e)[:50]

def main():
    results = {"PASS": [], "EMPTY": [], "FAIL": [], "TIMEOUT": []}

    for gt in ALL_TIERED_QUERIES:
        is_cross_dataset = len(gt.datasets) > 1

        # Test each value set
        for i, vs in enumerate(gt.value_sets):
            if not vs.sparql_query:
                continue

            # For cross-dataset queries, detect the target dataset from the query
            if is_cross_dataset:
                detected_dataset = detect_dataset_from_query(vs.sparql_query)
                endpoint = ENDPOINTS.get(detected_dataset) if detected_dataset else None
                if not endpoint:
                    # Fallback to primary dataset
                    endpoint = ENDPOINTS.get(gt.dataset)
            else:
                endpoint = ENDPOINTS.get(gt.dataset)

            if not endpoint:
                print(f"[SKIP] {gt.query_id}: Unknown dataset {gt.dataset}")
                continue

            status, count, error = test_query(endpoint, vs.sparql_query)
            vs_id = f"{gt.query_id}.VS{i+1}"

            results[status].append((vs_id, gt.description, count, error))

            if status == "PASS":
                print(f"[PASS] {vs_id}: {count} results")
            elif status == "EMPTY":
                print(f"[EMPTY] {vs_id}: {gt.description}")
            else:
                print(f"[{status}] {vs_id}: {error}")

    print("\n" + "="*60)
    print(f"SUMMARY: {len(results['PASS'])} PASS, {len(results['EMPTY'])} EMPTY, {len(results['FAIL'])} FAIL, {len(results['TIMEOUT'])} TIMEOUT")

    if results['FAIL']:
        print("\nFAILED queries:")
        for vs_id, desc, _, err in results['FAIL']:
            print(f"  - {vs_id}: {desc} ({err})")

    if results['TIMEOUT']:
        print("\nTIMEOUT queries:")
        for vs_id, desc, _, _ in results['TIMEOUT']:
            print(f"  - {vs_id}: {desc}")

    if results['EMPTY']:
        print("\nEMPTY queries:")
        for vs_id, desc, _, _ in results['EMPTY']:
            print(f"  - {vs_id}: {desc}")

if __name__ == "__main__":
    main()
