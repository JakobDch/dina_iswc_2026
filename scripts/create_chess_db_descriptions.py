"""Generate database_description CSVs for CHESS from SQLite schema.

CHESS uses these CSVs to build a context vector database for semantic
column search during schema selection. Each table gets one CSV file
with columns: original_column_name, column_name, column_description,
data_format, value_description.
"""

import argparse
import csv
import sqlite3
from pathlib import Path

CHESS_DB_ROOT = Path(__file__).resolve().parent.parent / "tools" / "chess" / "data" / "dev" / "dev_databases"

DB_IDS = ["edu", "trn", "nrg", "bsbm", "lca"]


def get_table_info(conn: sqlite3.Connection, table_name: str) -> list[dict]:
    """Get column info for a table using PRAGMA table_info."""
    cursor = conn.execute(f"PRAGMA table_info(`{table_name}`)")
    columns = []
    for row in cursor.fetchall():
        # cid, name, type, notnull, dflt_value, pk
        columns.append({
            "name": row[1],
            "type": row[2],
            "notnull": row[3],
            "pk": row[5],
        })
    return columns


def get_foreign_keys(conn: sqlite3.Connection, table_name: str) -> dict[str, str]:
    """Get foreign key info: column -> 'references table(column)'."""
    cursor = conn.execute(f"PRAGMA foreign_key_list(`{table_name}`)")
    fks = {}
    for row in cursor.fetchall():
        # id, seq, table, from, to, on_update, on_delete, match
        fks[row[3]] = f"references {row[2]}({row[4]})"
    return fks


def get_sample_values(conn: sqlite3.Connection, table_name: str, column_name: str, limit: int = 5) -> list[str]:
    """Get a few sample values for a column."""
    try:
        cursor = conn.execute(
            f"SELECT DISTINCT `{column_name}` FROM `{table_name}` WHERE `{column_name}` IS NOT NULL LIMIT {limit}"
        )
        return [str(row[0]) for row in cursor.fetchall()]
    except Exception:
        return []


def humanize_column_name(name: str) -> str:
    """Convert column_name to a more readable form."""
    # Replace underscores with spaces, capitalize
    return name.replace("_", " ").strip()


def generate_description_csvs(db_id: str) -> None:
    """Generate database_description/ CSVs for one database."""
    db_dir = CHESS_DB_ROOT / db_id
    db_path = db_dir / f"{db_id}.sqlite"
    if not db_path.exists():
        print(f"  SKIP: {db_path} not found")
        return

    desc_dir = db_dir / "database_description"
    desc_dir.mkdir(exist_ok=True)

    conn = sqlite3.connect(str(db_path))
    try:
        # Get all table names
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()

        total_cols = 0
        for (table_name,) in tables:
            if table_name.startswith("sqlite_"):
                continue

            columns = get_table_info(conn, table_name)
            fks = get_foreign_keys(conn, table_name)

            csv_path = desc_dir / f"{table_name}.csv"
            with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.DictWriter(f, fieldnames=[
                    "original_column_name", "column_name", "column_description",
                    "data_format", "value_description"
                ])
                writer.writeheader()

                for col in columns:
                    col_name = col["name"]
                    col_type = col["type"] or ""

                    # Build description
                    desc_parts = []
                    if col["pk"]:
                        desc_parts.append("Primary key.")
                    if col_name in fks:
                        desc_parts.append(f"Foreign key, {fks[col_name]}.")
                    if col["notnull"]:
                        desc_parts.append("Not null.")

                    # Get sample values for description
                    samples = get_sample_values(conn, table_name, col_name)
                    value_desc = ""
                    if samples:
                        value_desc = f"Example values: {', '.join(samples[:5])}"

                    writer.writerow({
                        "original_column_name": col_name,
                        "column_name": humanize_column_name(col_name),
                        "column_description": " ".join(desc_parts) if desc_parts else "",
                        "data_format": col_type,
                        "value_description": value_desc,
                    })
                    total_cols += 1

        print(f"  {db_id}: {len(tables)} tables, {total_cols} columns -> {desc_dir}")
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(description="Generate CHESS database_description CSVs")
    parser.add_argument(
        "--merged",
        action="store_true",
        help="Generate descriptions for the merged database instead of individual ones",
    )
    args = parser.parse_args()

    print("Generating CHESS database_description CSVs...")
    if args.merged:
        generate_description_csvs("merged")
    else:
        for db_id in DB_IDS:
            generate_description_csvs(db_id)
    print("Done.")


if __name__ == "__main__":
    main()
