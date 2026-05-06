#!/usr/bin/env python3
"""
Pre-index schema embeddings for all datasets.

Run this script during Docker build or before first use to create
persistent FAISS indices for fast semantic search.

Usage:
    python scripts/preindex_embeddings.py
    python scripts/preindex_embeddings.py --datasets EDU TRN
    python scripts/preindex_embeddings.py --force  # Rebuild all indices
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import argparse
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import AVAILABLE_DATASETS, EMBEDDINGS_CACHE_DIR
from src.tools.retrieval_tools import get_schema_index

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def preindex_dataset(dataset: str, force: bool = False) -> int:
    """
    Pre-index a single dataset.

    Args:
        dataset: Dataset name (EDU, NRG, etc.)
        force: Force re-indexing even if cache exists

    Returns:
        Number of elements indexed
    """
    logger.info(f"Pre-indexing {dataset}...")

    index = get_schema_index()
    count = index.index_dataset(dataset, force_reindex=force)

    if count > 0:
        logger.info(f"Successfully indexed {count} elements for {dataset}")
    else:
        logger.warning(f"No elements indexed for {dataset}")

    return count


def main():
    parser = argparse.ArgumentParser(
        description="Pre-index schema embeddings for semantic search"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=AVAILABLE_DATASETS,
        choices=AVAILABLE_DATASETS,
        help=f"Datasets to index (default: all)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force rebuild of all indices, ignoring cache",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Only check which indices exist",
    )

    args = parser.parse_args()

    # Ensure cache directory exists
    EMBEDDINGS_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if args.check:
        print("\nEmbedding Cache Status:")
        print("-" * 50)
        for dataset in AVAILABLE_DATASETS:
            cache_dir = EMBEDDINGS_CACHE_DIR / dataset
            faiss_path = cache_dir / "schema_index.faiss"
            elements_path = cache_dir / "schema_elements.json"

            if faiss_path.exists() and elements_path.exists():
                import json
                with open(elements_path) as f:
                    elements = json.load(f)
                print(f"  {dataset}: CACHED ({len(elements)} elements)")
            else:
                print(f"  {dataset}: NOT CACHED")
        print()
        return

    print("\n" + "=" * 60)
    print("PRE-INDEXING SCHEMA EMBEDDINGS")
    print("=" * 60)
    print(f"Datasets: {', '.join(args.datasets)}")
    print(f"Force rebuild: {args.force}")
    print(f"Cache directory: {EMBEDDINGS_CACHE_DIR}")
    print("=" * 60 + "\n")

    total_elements = 0
    results = {}

    for dataset in args.datasets:
        try:
            count = preindex_dataset(dataset, force=args.force)
            results[dataset] = count
            total_elements += count
        except Exception as e:
            logger.error(f"Failed to index {dataset}: {e}")
            results[dataset] = -1

    print("\n" + "=" * 60)
    print("INDEXING COMPLETE")
    print("=" * 60)
    for dataset, count in results.items():
        status = f"{count} elements" if count >= 0 else "FAILED"
        print(f"  {dataset}: {status}")
    print(f"\nTotal: {total_elements} elements indexed")
    print(f"Cache location: {EMBEDDINGS_CACHE_DIR}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
