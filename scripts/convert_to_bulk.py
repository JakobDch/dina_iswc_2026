#!/usr/bin/env python3
"""Convert single INSERT statements to bulk INSERTs for faster MySQL import."""

import re
import sys
from pathlib import Path
from collections import defaultdict

def convert_to_bulk_inserts(input_file: Path, output_file: Path, batch_size: int = 1000):
    """Convert single INSERTs to bulk INSERTs, pass through other statements."""

    print(f"Reading {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    print(f"Processing {len(lines)} lines...")

    # Pattern to match INSERT statements
    insert_pattern = re.compile(
        r'^insert\s+ignore\s+into\s+(\w+)\s*\(([^)]+)\)\s*values\s*\((.+)\)\s*;?\s*$',
        re.IGNORECASE
    )

    output_lines = []
    current_table = None
    current_columns = None
    current_values = []

    def flush_batch():
        """Write accumulated values as bulk INSERT."""
        nonlocal current_table, current_columns, current_values
        if current_values:
            # Write bulk INSERT
            values_str = "),\n(".join(current_values)
            output_lines.append(f"INSERT IGNORE INTO {current_table} ({current_columns}) VALUES\n({values_str});\n")
            current_values = []

    insert_count = 0
    other_count = 0

    for i, line in enumerate(lines):
        if i % 100000 == 0:
            print(f"  Processing line {i}/{len(lines)}...")

        line = line.strip()
        if not line:
            continue

        match = insert_pattern.match(line)

        if match:
            table_name = match.group(1)
            columns = match.group(2)
            values = match.group(3)

            # If switching tables or batch full, flush
            if table_name != current_table or columns != current_columns:
                flush_batch()
                current_table = table_name
                current_columns = columns

            current_values.append(values)
            insert_count += 1

            # Flush when batch is full
            if len(current_values) >= batch_size:
                flush_batch()
        else:
            # Non-INSERT statement - flush any pending and pass through
            flush_batch()
            current_table = None
            current_columns = None
            output_lines.append(line + "\n")
            other_count += 1

    # Final flush
    flush_batch()

    print(f"Writing {output_file}...")
    print(f"  Converted {insert_count} INSERTs to bulk format")
    print(f"  Passed through {other_count} other statements")

    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(output_lines)

    print("Done!")

if __name__ == "__main__":
    input_path = Path("data/sql/01_edu_schema.sql")
    output_path = Path("data/sql/01_edu_schema_bulk.sql")

    if len(sys.argv) > 1:
        input_path = Path(sys.argv[1])
    if len(sys.argv) > 2:
        output_path = Path(sys.argv[2])

    convert_to_bulk_inserts(input_path, output_path)