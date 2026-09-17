#!/usr/bin/env python3
"""Initialize slot-based mapping directories by copying base mappings.

This script creates isolated mapping directories for each concurrency slot,
allowing the Mapping Optimizer to modify mappings without affecting other
concurrent queries.

Usage:
    python scripts/initialize_slot_mappings.py --slots 4
    python scripts/initialize_slot_mappings.py --slots 2 --clean
"""

import argparse
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
MAPPINGS_SRC = PROJECT_ROOT / "mappings"
MAPPINGS_SLOTS = PROJECT_ROOT / "mappings-slots"

# Datasets that need slot isolation (used by OnTop containers)
DATASETS = [
    "edu-small",
    "trn-small",
    "nrg-small",
    "bsbm",
    "bgee",
    "lca",
    "edu-large",
    "trn-large",
    "nrg-large",
]


def initialize_slots(num_slots: int, clean: bool = False) -> None:
    """Create slot directories with copies of mapping files.

    Args:
        num_slots: Number of slots to create (typically matches concurrency)
        clean: If True, remove existing slot directories before creating
    """
    if clean and MAPPINGS_SLOTS.exists():
        print(f"Cleaning existing slot directories: {MAPPINGS_SLOTS}")
        shutil.rmtree(MAPPINGS_SLOTS)

    MAPPINGS_SLOTS.mkdir(parents=True, exist_ok=True)

    for slot_id in range(num_slots):
        slot_dir = MAPPINGS_SLOTS / f"slot{slot_id}"
        slot_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nCreating slot {slot_id}:")

        for dataset in DATASETS:
            src_dir = MAPPINGS_SRC / dataset
            if not src_dir.exists():
                print(f"  Skipping {dataset} (source not found)")
                continue

            dst_dir = slot_dir / dataset
            if dst_dir.exists():
                if not clean:
                    print(f"  Skipping {dataset} (already exists)")
                    continue
                shutil.rmtree(dst_dir)

            shutil.copytree(src_dir, dst_dir)
            print(f"  Created {dst_dir.relative_to(PROJECT_ROOT)}")

    print(f"\nInitialized {num_slots} slots in {MAPPINGS_SLOTS.relative_to(PROJECT_ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Initialize slot-based mapping directories for concurrent execution"
    )
    parser.add_argument(
        "--slots",
        type=int,
        default=3,
        help="Number of slots to create (default: 3)",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove existing slot directories before creating",
    )
    args = parser.parse_args()

    if args.slots < 1:
        parser.error("Number of slots must be at least 1")

    initialize_slots(args.slots, args.clean)


if __name__ == "__main__":
    main()
