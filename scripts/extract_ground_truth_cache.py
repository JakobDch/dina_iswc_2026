#!/usr/bin/env python3
"""
Extract and cache Ground Truth data for offline reevaluation.

This script extracts GT data from SPARQL endpoints and caches it to JSON,
enabling reevaluation without running containers.

Usage:
    python scripts/extract_ground_truth_cache.py [--skip-large]

Options:
    --skip-large  Skip LARGE dataset queries (faster extraction)

Output:
    data/cache/ground_truth_full.json - Complete GT cache with tuples and values
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tqdm import tqdm

from data.queries.experimental_corpus import ALL_EXPERIMENTAL_QUERIES
from data.queries.use_case_queries_tiered_tuples import (
    AdaptiveGroundTruth,
    RelevanceLevel,
)
from src.config import ONTOP_SINGLE_ENDPOINTS, SLOT_ENDPOINTS, CACHE_DIR

# Endpoints for extraction
# Try slot 0 first (8100+), fall back to original containers (8080-8085)
# This allows extraction to work with either container setup
def get_extraction_endpoints() -> dict[str, str]:
    """Get endpoints for GT extraction, preferring slot 0 containers."""
    import httpx

    # Try slot 0 endpoints first (including LARGE endpoints)
    slot0_endpoints = {
        # Small datasets
        "EDU": SLOT_ENDPOINTS[0].get("edu-small", ""),
        "TRN": SLOT_ENDPOINTS[0].get("trn-small", ""),
        "NRG": SLOT_ENDPOINTS[0].get("nrg-small", ""),
        "BSBM": SLOT_ENDPOINTS[0].get("bsbm", ""),
        "LCA": SLOT_ENDPOINTS[0].get("lca", ""),
        # LARGE endpoints for SET_LARGE queries
        "EDU_LARGE": SLOT_ENDPOINTS[0].get("edu-large", ""),
        "TRN_LARGE": SLOT_ENDPOINTS[0].get("trn-large", ""),
        "NPD_LARGE": SLOT_ENDPOINTS[0].get("nrg-large", ""),
    }

    # Test if slot 0 EDU is reachable
    test_endpoint = slot0_endpoints.get("EDU", "")
    if test_endpoint:
        try:
            response = httpx.get(test_endpoint.replace("/sparql", ""), timeout=2.0)
            if response.status_code < 500:
                print("Using slot 0 containers (8100+)")
                return slot0_endpoints
        except Exception:
            pass

    # Fall back to original endpoints (including LARGE)
    print("Using original containers (8080-8092)")
    endpoints = {k.upper(): v for k, v in ONTOP_SINGLE_ENDPOINTS.items() if "-large" not in k}
    # Add LARGE endpoints with underscore keys matching dataset names (e.g. EDU_LARGE)
    for k, v in ONTOP_SINGLE_ENDPOINTS.items():
        if "-large" in k:
            key = k.replace("-large", "_LARGE").replace("-", "").upper()
            endpoints[key] = v
    return endpoints


ENDPOINTS = get_extraction_endpoints()

# Output file
CACHE_FILE = CACHE_DIR / "ground_truth_full.json"


def _extract_variant_columns(gt: AdaptiveGroundTruth, query_index: int) -> list[dict]:
    """Extract column metadata for a specific query variant."""
    cols = gt.get_columns_for_query(query_index)
    return [
        {
            "var_name": col.var_name,
            "level": col.level.name,
            "description": col.description,
            "is_measurement": col.is_measurement,
            "semantic_concept": col.semantic_concept,
        }
        for col in cols
    ]


def extract_single_dataset_gt(
    gt: AdaptiveGroundTruth,
    endpoint: str,
    timeout: float = 60.0,
) -> dict:
    """Extract GT data for a single-dataset query (all SPARQL variants)."""
    query_variants = []

    for qi in range(len(gt.sparql_queries)):
        try:
            gt_tuples, gt_columns, null_columns = gt.execute_query(
                endpoint, timeout, query_index=qi
            )
            preferred_values = gt.get_values_by_level(
                RelevanceLevel.PREFERRED, endpoint, timeout, query_index=qi
            )
            acceptable_values = gt.get_values_by_level(
                RelevanceLevel.ACCEPTABLE, endpoint, timeout, query_index=qi
            )

            query_variants.append({
                "query_index": qi,
                "sparql": gt.sparql_queries[qi],
                "success": True,
                "endpoint_used": endpoint,
                "gt_tuples": [list(t) for t in sorted(gt_tuples)],
                "gt_columns": gt_columns,
                "null_columns": null_columns,
                "values_by_level": {
                    "PREFERRED": sorted(preferred_values),
                    "ACCEPTABLE": sorted(acceptable_values),
                },
                "row_count": len(gt_tuples),
                "columns": _extract_variant_columns(gt, qi),
            })
        except Exception as e:
            query_variants.append({
                "query_index": qi,
                "sparql": gt.sparql_queries[qi] if qi < len(gt.sparql_queries) else "",
                "success": False,
                "error": str(e),
            })

    # Use first successful variant as top-level result (backward compat)
    main = next((v for v in query_variants if v.get("success")), query_variants[0] if query_variants else {})
    result = {
        "success": main.get("success", False),
        "endpoint_used": endpoint,
        "gt_tuples": main.get("gt_tuples", []),
        "gt_columns": main.get("gt_columns", []),
        "null_columns": main.get("null_columns", []),
        "values_by_level": main.get("values_by_level", {}),
        "row_count": main.get("row_count", 0),
        "query_variants": query_variants,
    }
    if not main.get("success"):
        result["error"] = main.get("error", "No variants extracted")
    return result


def _extract_cross_variant(
    gt: AdaptiveGroundTruth,
    query_index: int,
    timeout: float = 60.0,
) -> dict:
    """Extract GT data for one SPARQL variant of a CROSS query across all datasets."""
    import httpx

    sparql = gt.sparql_queries[query_index]
    all_tuples: set[tuple[str, ...]] = set()
    all_columns: list[str] = []
    preferred_values: set[str] = set()
    acceptable_values: set[str] = set()
    endpoints_used: list[str] = []
    errors: list[str] = []

    for dataset in gt.datasets:
        endpoint = ENDPOINTS.get(dataset.upper())
        if not endpoint:
            errors.append(f"No endpoint for dataset {dataset}")
            continue

        endpoints_used.append(endpoint)

        try:
            response = httpx.post(
                endpoint,
                data={"query": sparql},
                headers={"Accept": "application/sparql-results+json"},
                timeout=timeout,
            )
            response.raise_for_status()
            result = response.json()

            bindings = result.get("results", {}).get("bindings", [])
            vars_list = result.get("head", {}).get("vars", [])

            if not all_columns and vars_list:
                all_columns = vars_list

            for binding in bindings:
                row_values = []
                for var in all_columns:
                    val = binding.get(var, {})
                    if isinstance(val, dict):
                        row_values.append(val.get("value", "").strip().lower())
                    else:
                        row_values.append(str(val).strip().lower() if val else "")
                if row_values and any(v for v in row_values):
                    all_tuples.add(tuple(row_values))

        except Exception as e:
            errors.append(f"{dataset}: {e}")

    # Compute values_by_level directly from accumulated tuples
    cols = gt.get_columns_for_query(query_index)
    preferred_indices = [i for i, c in enumerate(cols) if c.level == RelevanceLevel.PREFERRED]
    acceptable_indices = [i for i, c in enumerate(cols) if c.level == RelevanceLevel.ACCEPTABLE]
    for tup in all_tuples:
        for idx in preferred_indices:
            if idx < len(tup) and tup[idx]:
                preferred_values.add(tup[idx])
        for idx in acceptable_indices:
            if idx < len(tup) and tup[idx]:
                acceptable_values.add(tup[idx])

    return {
        "query_index": query_index,
        "sparql": sparql,
        "success": len(all_tuples) > 0,
        "is_cross_query": True,
        "datasets": gt.datasets,
        "endpoints_used": endpoints_used,
        "gt_tuples": [list(t) for t in sorted(all_tuples)],
        "gt_columns": all_columns,
        "null_columns": [],
        "values_by_level": {
            "PREFERRED": sorted(preferred_values),
            "ACCEPTABLE": sorted(acceptable_values),
        },
        "row_count": len(all_tuples),
        "columns": _extract_variant_columns(gt, query_index),
        "errors": errors if errors else None,
    }


def extract_cross_dataset_gt(
    gt: AdaptiveGroundTruth,
    timeout: float = 60.0,
) -> dict:
    """Extract GT data for a CROSS (multi-dataset) query - all SPARQL variants."""
    if not gt.sparql_queries:
        return {"success": False, "error": "No SPARQL queries defined"}

    query_variants = []
    for qi in range(len(gt.sparql_queries)):
        variant = _extract_cross_variant(gt, qi, timeout)
        query_variants.append(variant)

    # Use first successful variant as top-level (backward compat)
    main = next((v for v in query_variants if v.get("success")), query_variants[0])
    result = {
        "success": main.get("success", False),
        "is_cross_query": True,
        "datasets": gt.datasets,
        "endpoints_used": main.get("endpoints_used", []),
        "gt_tuples": main.get("gt_tuples", []),
        "gt_columns": main.get("gt_columns", []),
        "null_columns": [],
        "values_by_level": main.get("values_by_level", {}),
        "row_count": main.get("row_count", 0),
        "query_variants": query_variants,
        "errors": main.get("errors"),
    }
    if not main.get("success"):
        result["error"] = main.get("error", "No variants extracted")
    return result


def extract_all_ground_truth(skip_large: bool = False, only_large: bool = False, timeout: float = 60.0) -> dict:
    """Extract GT data for all queries.

    Args:
        skip_large: If True, skip LARGE dataset queries.
        only_large: If True, extract ONLY LARGE dataset queries.
    """
    queries = ALL_EXPERIMENTAL_QUERIES
    if only_large:
        queries = [q for q in ALL_EXPERIMENTAL_QUERIES if q.query_set == "LARGE"]
        print(f"Extracting ONLY LARGE queries. Extracting {len(queries)} queries...")
    elif skip_large:
        queries = [q for q in ALL_EXPERIMENTAL_QUERIES if q.query_set != "LARGE"]
        print(f"Skipping LARGE queries. Extracting {len(queries)} queries...")
    else:
        print(f"Extracting Ground Truth for {len(queries)} queries...")
    print(f"Using endpoints: {ENDPOINTS}")

    cache = {
        "extracted_at": datetime.now().isoformat(),
        "endpoints": ENDPOINTS,
        "queries": {},
    }

    success_count = 0
    error_count = 0

    for gt in tqdm(queries, desc="Extracting GT"):
        query_id = gt.query_id
        is_cross = len(gt.datasets) > 1

        if is_cross:
            result = extract_cross_dataset_gt(gt, timeout=timeout)
        else:
            endpoint = ENDPOINTS.get(gt.dataset.upper())
            if not endpoint:
                result = {"success": False, "error": f"No endpoint for {gt.dataset}"}
            else:
                result = extract_single_dataset_gt(gt, endpoint, timeout=timeout)

        # Add metadata
        result["query_id"] = query_id
        result["query_set"] = gt.query_set
        result["dataset"] = gt.dataset
        result["datasets"] = gt.datasets if gt.datasets else [gt.dataset]
        result["query_text"] = gt.query
        result["description"] = gt.description
        result["sparql"] = gt.sparql_queries[0] if gt.sparql_queries else ""
        result["columns"] = _extract_variant_columns(gt, 0)

        cache["queries"][query_id] = result

        if result.get("success"):
            success_count += 1
        else:
            error_count += 1
            print(f"  Error for {query_id}: {result.get('error', 'Unknown')}")

    cache["summary"] = {
        "total_queries": len(queries),
        "skipped_large": skip_large,
        "successful": success_count,
        "errors": error_count,
    }

    return cache


def main():
    parser = argparse.ArgumentParser(
        description="Extract Ground Truth data for offline reevaluation."
    )
    parser.add_argument(
        "--skip-large",
        action="store_true",
        help="Skip LARGE dataset queries (faster extraction)",
    )
    parser.add_argument(
        "--only-large",
        action="store_true",
        help="Extract ONLY LARGE dataset queries",
    )
    parser.add_argument(
        "--merge-into",
        type=str,
        help="Merge results into existing cache file instead of overwriting",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="SPARQL query timeout in seconds (default: 60)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("GROUND TRUTH EXTRACTION")
    print("=" * 60)

    # Ensure cache directory exists
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Extract all GT data
    cache = extract_all_ground_truth(
        skip_large=args.skip_large,
        only_large=args.only_large,
        timeout=args.timeout,
    )

    # Merge into existing cache if requested
    if args.merge_into:
        merge_path = Path(args.merge_into)
        if merge_path.exists():
            with open(merge_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
            existing["queries"].update(cache["queries"])
            existing["summary"]["total_queries"] = len(existing["queries"])
            existing["summary"]["successful"] = sum(
                1 for q in existing["queries"].values() if q.get("success")
            )
            existing["summary"]["errors"] = sum(
                1 for q in existing["queries"].values() if not q.get("success")
            )
            cache = existing
            print(f"Merged into existing cache ({len(cache['queries'])} total queries)")

    # Save to file
    output_file = Path(args.merge_into) if args.merge_into else CACHE_FILE
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 60}")
    print(f"EXTRACTION COMPLETE")
    print(f"{'=' * 60}")
    print(f"Total queries: {cache['summary']['total_queries']}")
    print(f"Successful:    {cache['summary']['successful']}")
    print(f"Errors:        {cache['summary']['errors']}")
    if args.skip_large:
        print(f"Note: LARGE queries skipped")
    if args.only_large:
        print(f"Note: Only LARGE queries extracted")
    print(f"\nCache saved to: {output_file}")
    print(f"File size: {output_file.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()