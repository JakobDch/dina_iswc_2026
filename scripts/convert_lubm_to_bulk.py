#!/usr/bin/env python3
"""Convert EDU INSERT+UPDATE pattern to efficient bulk INSERTs."""

import re
from pathlib import Path
from collections import defaultdict

def convert_edu_to_bulk(input_file: Path, output_file: Path, batch_size: int = 1000):
    """
    EDU files have this inefficient pattern:
        INSERT IGNORE INTO table (nr) VALUES (123);
        UPDATE table set col1= "val1" where nr = 123;
        UPDATE table set col2= "val2" where nr = 123;

    Convert to efficient bulk INSERTs:
        INSERT INTO table (nr, col1, col2) VALUES (123, 'val1', 'val2'), ...;

    IMPORTANT: Only uses columns that exist in the CREATE TABLE definition.
    ALTER TABLE CHANGE statements and subsequent UPDATEs are kept in postamble.
    """

    print(f"Reading {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()

    lines = content.split('\n')
    print(f"Processing {len(lines)} lines...")

    # First pass: Parse CREATE TABLE statements to get valid column names
    print("First pass: Parsing schema to get valid columns...")
    create_table_pattern = re.compile(
        r'CREATE\s+TABLE\s+(\w+)\s*\(',
        re.IGNORECASE
    )
    column_pattern = re.compile(r'^\s*(\w+)\s+(?:int|varchar|char|text|date|datetime|decimal|float|double|blob|enum)', re.IGNORECASE)

    valid_columns = {}  # {table_lower: set of column names (lowercase)}
    current_create_table = None

    for line in lines:
        line_stripped = line.strip()

        # Check for CREATE TABLE
        create_match = create_table_pattern.match(line_stripped)
        if create_match:
            current_create_table = create_match.group(1).lower()
            valid_columns[current_create_table] = set()
            continue

        # Inside CREATE TABLE, parse column definitions
        if current_create_table:
            if line_stripped.startswith(')'):
                current_create_table = None
                continue
            col_match = column_pattern.match(line_stripped)
            if col_match:
                col_name = col_match.group(1).lower()
                valid_columns[current_create_table].add(col_name)

    print(f"  Found {len(valid_columns)} tables with columns:")
    for table, cols in sorted(valid_columns.items()):
        print(f"    {table}: {sorted(cols)}")

    # Second pass: Parse data
    print("Second pass: Parsing INSERT and UPDATE statements...")

    # Patterns
    # Pattern 1: Single-column INSERT header (for tables with UPDATE pattern)
    insert_header_pattern = re.compile(
        r'^insert\s+ignore\s+into\s+(\w+)\s*\((\w+)\)\s*values\s*$',
        re.IGNORECASE
    )
    values_pattern = re.compile(r'^\((\d+)\)\s*;?\s*$')

    # Pattern 2: Multi-column INSERT header (for join tables)
    # Example: INSERT IGNORE INTO undergraduateStudentTakeCourse (undergraduateStudentID, undergraduateCourseID) VALUES
    insert_multi_col_pattern = re.compile(
        r'^insert\s+ignore\s+into\s+(\w+)\s*\(([^)]+)\)\s*values\s*$',
        re.IGNORECASE
    )
    # Values for multi-column: (1, 491), or (1, 491);
    multi_values_pattern = re.compile(r'^\(([^)]+)\)\s*[,;]?\s*$')

    update_pattern = re.compile(
        r'^UPDATE\s+(\w+)\s+set\s+(\w+)\s*=\s*(.+?)\s+where\s+(\w+)\s*=\s*(\d+)\s*;?\s*$',
        re.IGNORECASE
    )

    # ALTER TABLE ... CHANGE pattern (renames columns - marks end of bulk data)
    alter_change_pattern = re.compile(
        r'^ALTER\s+TABLE\s+\w+\s+CHANGE',
        re.IGNORECASE
    )

    pending_insert = None  # (table, pk_col) waiting for values
    pending_multi_insert = None  # (table, [col1, col2, ...]) waiting for values

    # Track records per table: {table: {pk_value: {col: value}}}
    tables = defaultdict(lambda: defaultdict(dict))
    table_pk_col = {}  # {table: pk_column_name}

    # Track join table records: {table: [(val1, val2, ...), ...]}
    join_tables = defaultdict(list)
    join_table_cols = {}  # {table: [col1, col2, ...]}

    # Non-data statements (CREATE, ALTER, etc.)
    preamble = []
    postamble = []
    in_data_section = False
    data_finished = False
    past_column_rename = False  # True after ALTER TABLE ... CHANGE

    for i, line in enumerate(lines):
        if i % 100000 == 0:
            print(f"  Parsing line {i}/{len(lines)}...")

        line_stripped = line.strip()
        if not line_stripped:
            continue

        # Check for ALTER TABLE ... CHANGE (column rename - everything after goes to postamble)
        if alter_change_pattern.match(line_stripped):
            past_column_rename = True
            postamble.append(line)
            continue

        # After column rename, everything goes to postamble
        if past_column_rename:
            postamble.append(line)
            continue

        # Check for multi-column INSERT header (join tables)
        insert_multi_match = insert_multi_col_pattern.match(line_stripped)
        if insert_multi_match:
            table = insert_multi_match.group(1)
            cols_str = insert_multi_match.group(2)
            cols = [c.strip() for c in cols_str.split(',')]
            # Only handle if more than one column (join tables)
            if len(cols) > 1:
                in_data_section = True
                pending_multi_insert = (table, cols)
                join_table_cols[table] = cols
                continue

        # Check for values line following multi-column INSERT header
        if pending_multi_insert:
            multi_match = multi_values_pattern.match(line_stripped)
            if multi_match:
                table, cols = pending_multi_insert
                vals_str = multi_match.group(1)
                vals = [v.strip() for v in vals_str.split(',')]
                join_tables[table].append(tuple(vals))
                # Keep pending for next values line
                continue
            else:
                pending_multi_insert = None  # Reset if not a values line

        # Check for single-column INSERT header (INSERT INTO table (col) VALUES)
        insert_header_match = insert_header_pattern.match(line_stripped)
        if insert_header_match:
            in_data_section = True
            pending_insert = (insert_header_match.group(1), insert_header_match.group(2))
            continue

        # Check for values line following single-column INSERT header
        if pending_insert:
            values_match = values_pattern.match(line_stripped)
            if values_match:
                table, pk_col = pending_insert
                pk_val = int(values_match.group(1))
                table_pk_col[table] = pk_col
                tables[table][pk_val][pk_col] = pk_val
                pending_insert = None
                continue
            else:
                pending_insert = None  # Reset if not a values line

        # Check for UPDATE
        update_match = update_pattern.match(line_stripped)
        if update_match:
            in_data_section = True
            table = update_match.group(1)
            col = update_match.group(2)
            val = update_match.group(3).strip()
            pk_val = int(update_match.group(5))

            # Only include column if it exists in the schema
            table_lower = table.lower()
            col_lower = col.lower()
            if table_lower in valid_columns and col_lower in valid_columns[table_lower]:
                # Clean up the value
                if val.startswith('"') and val.endswith('"'):
                    val = val[1:-1]  # Remove quotes, we'll re-add them

                tables[table][pk_val][col] = val
            continue

        # Other statements
        if not in_data_section:
            preamble.append(line)
        elif line_stripped.lower().startswith('alter table') and 'enable keys' in line_stripped.lower():
            data_finished = True
            postamble.append(line)
        elif data_finished:
            postamble.append(line)

    print(f"  Found {len(tables)} tables with data")
    for table, records in tables.items():
        print(f"    {table}: {len(records)} records")
    print(f"  Found {len(join_tables)} join tables with data")
    for table, records in join_tables.items():
        print(f"    {table}: {len(records)} records")

    # Generate bulk INSERTs
    print("Generating bulk INSERTs...")
    output_lines = preamble.copy()
    output_lines.append("")
    output_lines.append("-- Bulk INSERT data")
    output_lines.append("SET FOREIGN_KEY_CHECKS = 0;")
    output_lines.append("")

    for table, records in tables.items():
        if not records:
            continue

        # Get all columns used in this table (only valid ones)
        table_lower = table.lower()
        all_cols = set()
        for pk_val, cols in records.items():
            for col in cols.keys():
                if table_lower in valid_columns and col.lower() in valid_columns[table_lower]:
                    all_cols.add(col)

        # Sort columns, pk first
        pk_col = table_pk_col.get(table, 'nr')
        sorted_cols = [pk_col] + sorted(c for c in all_cols if c != pk_col)

        output_lines.append(f"-- {table}: {len(records)} records")

        # Generate batched INSERTs
        record_list = list(records.items())
        for batch_start in range(0, len(record_list), batch_size):
            batch = record_list[batch_start:batch_start + batch_size]

            values_list = []
            for pk_val, cols in batch:
                row_values = []
                for col in sorted_cols:
                    val = cols.get(col)
                    if val is None:
                        row_values.append("NULL")
                    elif isinstance(val, int) or (isinstance(val, str) and val.isdigit()):
                        row_values.append(str(val))
                    else:
                        # Escape single quotes
                        escaped = str(val).replace("'", "''").replace("\\", "\\\\")
                        row_values.append(f"'{escaped}'")
                values_list.append(f"({', '.join(row_values)})")

            cols_str = ", ".join(sorted_cols)
            values_str = ",\n".join(values_list)
            output_lines.append(f"INSERT INTO {table} ({cols_str}) VALUES")
            output_lines.append(f"{values_str};")
            output_lines.append("")

    # Generate bulk INSERTs for join tables
    for table, records in join_tables.items():
        if not records:
            continue

        cols = join_table_cols.get(table, [])
        if not cols:
            continue

        output_lines.append(f"-- {table}: {len(records)} records")

        # Generate batched INSERTs
        for batch_start in range(0, len(records), batch_size):
            batch = records[batch_start:batch_start + batch_size]

            values_list = []
            for vals in batch:
                row_values = []
                for val in vals:
                    if val.isdigit():
                        row_values.append(val)
                    else:
                        escaped = val.replace("'", "''").replace("\\", "\\\\")
                        row_values.append(f"'{escaped}'")
                values_list.append(f"({', '.join(row_values)})")

            cols_str = ", ".join(cols)
            values_str = ",\n".join(values_list)
            output_lines.append(f"INSERT INTO {table} ({cols_str}) VALUES")
            output_lines.append(f"{values_str};")
            output_lines.append("")

    output_lines.append("SET FOREIGN_KEY_CHECKS = 1;")
    output_lines.append("")
    output_lines.extend(postamble)

    print(f"Writing {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(output_lines))

    print("Done!")

if __name__ == "__main__":
    input_path = Path("data/sql/01_edu_schema.sql.loaded")
    output_path = Path("data/sql/01_edu_schema.sql")

    import sys
    if len(sys.argv) > 1:
        input_path = Path(sys.argv[1])
    if len(sys.argv) > 2:
        output_path = Path(sys.argv[2])

    convert_edu_to_bulk(input_path, output_path)
