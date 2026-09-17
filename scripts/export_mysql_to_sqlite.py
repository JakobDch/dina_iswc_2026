"""Export MySQL SQL dumps to SQLite databases for CHESS Text2SQL integration.

Converts the project's MySQL SQL dump files into SQLite databases in the CHESS
expected directory structure:
    tools/chess/data/dev/dev_databases/{db_id}/{db_id}.sqlite

Usage:
    python scripts/export_mysql_to_sqlite.py [--datasets edu trn nrg bsbm lca]
"""

import argparse
import re
import sqlite3
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ============================================================================
# Dataset -> SQL file mapping
# ============================================================================
DATASET_CONFIG = {
    "edu": {
        "sql_files": [PROJECT_ROOT / "data" / "sql" / "experiment" / "03_lubm_small.sql"],
        "description": "Education domain (LUBM - 100 universities)",
    },
    "trn": {
        "sql_files": [PROJECT_ROOT / "data" / "sql" / "experiment" / "01_gtfs_small.sql"],
        "description": "Transport domain (GTFS Madrid Metro)",
    },
    "nrg": {
        "sql_files": [PROJECT_ROOT / "data" / "sql" / "experiment" / "05_npd_small.sql"],
        "description": "Energy domain (Norwegian Petroleum Directorate)",
    },
    "bsbm": {
        "sql_files": [PROJECT_ROOT / "data" / "sql" / "07_bsbm_benchmark.sql"],
        "description": "E-commerce (Berlin SPARQL Benchmark)",
    },
    "lca": {
        "sql_files": [
            PROJECT_ROOT / "data" / "sql" / "08_lca_schema.sql",
            PROJECT_ROOT / "data" / "sql" / "08b_lca_data.sql",
        ],
        "description": "Life Cycle Assessment (HDPE production)",
    },
}

CHESS_DB_ROOT = PROJECT_ROOT / "tools" / "chess" / "data" / "dev" / "dev_databases"

# Lines that start with these patterns are MySQL-specific and should be skipped
SKIP_RE = re.compile(
    r"^\s*("
    r"CREATE\s+DATABASE|DROP\s+DATABASE|USE\s+|SET\s+|"
    r"LOCK\s+TABLES|UNLOCK\s+TABLES|/\*!|"
    r"ALTER\s+TABLE\s+.*\b(DISABLE|ENABLE)\s+KEYS|"
    r"ALTER\s+TABLE\s+.*\bADD\s+CONSTRAINT"
    r")",
    re.IGNORECASE,
)


def _protect_backtick_names(sql: str) -> tuple[str, dict[str, str]]:
    """Replace backtick-quoted identifiers with placeholders to protect from type conversion."""
    placeholders = {}
    counter = [0]

    def replacer(m: re.Match) -> str:
        key = f"__BT{counter[0]}__"
        placeholders[key] = m.group(0)
        counter[0] += 1
        return key

    protected = re.sub(r"`[^`]+`", replacer, sql)
    return protected, placeholders


def _restore_backtick_names(sql: str, placeholders: dict[str, str]) -> str:
    """Restore backtick-quoted identifiers from placeholders."""
    for key, value in placeholders.items():
        sql = sql.replace(key, value)
    return sql


def convert_create_table(sql: str) -> str:
    """Convert a complete CREATE TABLE statement from MySQL to SQLite."""
    # Protect backtick-quoted identifiers (e.g., `date`, `key`) from type conversion
    sql, placeholders = _protect_backtick_names(sql)

    # Remove ENGINE=... clause at the end
    sql = re.sub(r"\)\s*ENGINE\s*=\s*\w+[^;]*;", ");", sql, flags=re.IGNORECASE)

    # Remove character set / collate
    sql = re.sub(r"\s+character\s+set\s+\w+(\s+collate\s+\w+)?", "", sql, flags=re.IGNORECASE)
    sql = re.sub(r"\s+DEFAULT\s+CHARSET\s*=\s*\w+", "", sql, flags=re.IGNORECASE)
    sql = re.sub(r"\s+COLLATE\s*=?\s*\w+", "", sql, flags=re.IGNORECASE)

    # Handle INT AUTO_INCREMENT PRIMARY KEY -> INTEGER PRIMARY KEY AUTOINCREMENT
    sql = re.sub(
        r"\bINT\s+AUTO_INCREMENT\s+PRIMARY\s+KEY\b",
        "INTEGER PRIMARY KEY AUTOINCREMENT",
        sql,
        flags=re.IGNORECASE,
    )
    sql = re.sub(r"\bINT\s+AUTO_INCREMENT\b", "INTEGER", sql, flags=re.IGNORECASE)

    # Remove 'unsigned' keyword
    sql = re.sub(r"\bunsigned\b", "", sql, flags=re.IGNORECASE)

    # Convert data types
    sql = re.sub(r"\bint\(\d+\)", "INTEGER", sql, flags=re.IGNORECASE)
    sql = re.sub(r"\bINT\b", "INTEGER", sql, flags=re.IGNORECASE)
    sql = re.sub(r"\bvarchar\(\d+\)", "TEXT", sql, flags=re.IGNORECASE)
    sql = re.sub(r"\bchar\(\d+\)", "TEXT", sql, flags=re.IGNORECASE)
    sql = re.sub(r"\bDECIMAL\(\d+\s*,\s*\d+\)", "REAL", sql, flags=re.IGNORECASE)
    sql = re.sub(r"\bDOUBLE\b", "REAL", sql, flags=re.IGNORECASE)
    sql = re.sub(r"\bFLOAT\b", "REAL", sql, flags=re.IGNORECASE)
    sql = re.sub(r"\bBOOLEAN\b", "INTEGER", sql, flags=re.IGNORECASE)
    sql = re.sub(r"\bDATETIME\b", "TEXT", sql, flags=re.IGNORECASE)
    sql = re.sub(r"\bDATE\b", "TEXT", sql, flags=re.IGNORECASE)
    sql = re.sub(r"\bTINYINT\(\d+\)", "INTEGER", sql, flags=re.IGNORECASE)
    sql = re.sub(r"\bSMALLINT\b", "INTEGER", sql, flags=re.IGNORECASE)
    sql = re.sub(r"\bBIGINT\b", "INTEGER", sql, flags=re.IGNORECASE)
    sql = re.sub(r"\bMEDIUMTEXT\b", "TEXT", sql, flags=re.IGNORECASE)
    sql = re.sub(r"\bLONGTEXT\b", "TEXT", sql, flags=re.IGNORECASE)
    sql = re.sub(r"\btext\b", "TEXT", sql, flags=re.IGNORECASE)

    # Remove remaining AUTO_INCREMENT
    sql = re.sub(r"\s+AUTO_INCREMENT", "", sql, flags=re.IGNORECASE)

    # Remove DEFAULT NULL
    sql = re.sub(r"\s+default\s+NULL", "", sql, flags=re.IGNORECASE)

    # Remove MySQL INDEX definitions (but NOT PRIMARY KEY, FOREIGN KEY, UNIQUE constraints)
    # IMPORTANT: UNIQUE KEY must be removed BEFORE standalone KEY to avoid partial matches
    sql = re.sub(r",?\s*UNIQUE\s+KEY\s+\w+\s*\([^)]+\)", "", sql, flags=re.IGNORECASE)
    # Forms: INDEX name (col), INDEX USING BTREE (col)
    sql = re.sub(r",?\s*INDEX\s+(?:USING\s+\w+\s*)?\w*\s*\([^)]+\)", "", sql, flags=re.IGNORECASE)
    # Standalone KEY (not PRIMARY/FOREIGN KEY): KEY idx_name (col1, col2)
    sql = re.sub(r",?\s*(?<!\w)KEY\s+\w+\s*\([^)]+\)", "", sql, flags=re.IGNORECASE)

    # Remove ON DELETE SET NULL / CASCADE (often causes issues with missing ref tables)
    sql = re.sub(r"\s+ON\s+DELETE\s+\w+(?:\s+\w+)?", "", sql, flags=re.IGNORECASE)
    sql = re.sub(r"\s+ON\s+UPDATE\s+\w+(?:\s+\w+)?", "", sql, flags=re.IGNORECASE)

    # Remove INTEGER(N) size specifiers (SQLite doesn't support them)
    sql = re.sub(r"\bINTEGER\s*\(\d+\)", "INTEGER", sql, flags=re.IGNORECASE)

    # If AUTOINCREMENT was used, remove separate PRIMARY KEY (col) constraint
    if "AUTOINCREMENT" in sql.upper():
        sql = re.sub(r",?\s*PRIMARY\s+KEY\s*\(\w+\)", "", sql, flags=re.IGNORECASE)

    # Clean up trailing commas before closing parenthesis
    sql = re.sub(r",\s*\)", "\n)", sql)

    # Clean up multiple spaces
    sql = re.sub(r"  +", " ", sql)

    # Restore protected backtick identifiers
    sql = _restore_backtick_names(sql, placeholders)

    return sql


def split_into_statements(file_path: Path) -> list[str]:
    """Split a MySQL SQL file into individual complete statements.

    Handles multi-line statements correctly by tracking whether we're
    inside a statement (waiting for the closing semicolon).
    """
    statements = []
    buffer = []
    in_statement = False

    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n\r")
            stripped = line.strip()

            # Skip empty lines and comments when not in a statement
            if not in_statement:
                if not stripped or stripped.startswith("--"):
                    continue
                if SKIP_RE.match(stripped):
                    continue

            # We're either starting a new statement or continuing one
            buffer.append(line)

            # Check if statement is complete (ends with ;)
            if stripped.endswith(";"):
                full_stmt = "\n".join(buffer).strip()
                if full_stmt:
                    statements.append(full_stmt)
                buffer = []
                in_statement = False
            else:
                in_statement = True

    # Handle any remaining buffer (incomplete statement)
    if buffer:
        full_stmt = "\n".join(buffer).strip()
        if full_stmt:
            statements.append(full_stmt)

    return statements


def convert_statement(stmt: str) -> str | None:
    """Convert a MySQL SQL statement to SQLite-compatible SQL.

    Returns None if the statement should be skipped.
    """
    upper = stmt.strip().upper()

    # Skip MySQL-specific commands
    if SKIP_RE.match(stmt.strip()):
        return None

    # CREATE TABLE
    if upper.startswith("CREATE TABLE"):
        return convert_create_table(stmt)

    # DROP TABLE
    if upper.startswith("DROP TABLE"):
        return stmt

    # INSERT INTO
    if upper.startswith("INSERT"):
        return stmt.replace("\\'", "''")

    # Other DDL (ALTER TABLE, etc.)
    if stmt.strip().endswith(";"):
        return stmt

    return None


def create_sqlite_database(db_id: str, sql_files: list[Path], output_dir: Path) -> Path:
    """Create a SQLite database from MySQL SQL dump files."""
    db_dir = output_dir / db_id
    db_dir.mkdir(parents=True, exist_ok=True)
    db_path = db_dir / f"{db_id}.sqlite"

    if db_path.exists():
        db_path.unlink()

    print(f"  Creating: {db_path}")

    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=OFF")
    conn.execute("PRAGMA foreign_keys=OFF")  # OFF during import for speed

    success_count = 0
    error_count = 0

    for sql_file in sql_files:
        if not sql_file.exists():
            print(f"  WARNING: SQL file not found: {sql_file}")
            continue

        size_mb = sql_file.stat().st_size / 1024 / 1024
        print(f"  Parsing: {sql_file.name} ({size_mb:.1f} MB)")

        raw_statements = split_into_statements(sql_file)
        print(f"    {len(raw_statements)} raw statements found")

        for stmt in raw_statements:
            converted = convert_statement(stmt)
            if converted is None:
                continue
            try:
                conn.execute(converted)
                success_count += 1
            except Exception as e:
                error_count += 1
                if error_count <= 5:
                    short = converted[:150].replace("\n", " ")
                    print(f"    ERROR: {e}")
                    print(f"      SQL: {short}...")
                elif error_count == 6:
                    print(f"    ... (suppressing further errors)")

    conn.commit()

    # Verify
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [r[0] for r in cursor.fetchall() if r[0] != "sqlite_sequence"]
    total_rows = 0
    for table in tables:
        cnt = conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
        total_rows += cnt
        print(f"    Table '{table}': {cnt:,} rows")

    conn.close()

    size_mb = db_path.stat().st_size / 1024 / 1024
    print(f"  Result: {len(tables)} tables, {total_rows:,} rows, {size_mb:.1f} MB")
    print(f"          {success_count:,} OK, {error_count:,} errors")
    print()

    return db_path


def main():
    parser = argparse.ArgumentParser(description="Export MySQL SQL dumps to SQLite for CHESS")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DATASET_CONFIG.keys()),
        choices=list(DATASET_CONFIG.keys()),
        help="Datasets to export (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=CHESS_DB_ROOT,
        help=f"Output directory (default: {CHESS_DB_ROOT})",
    )
    parser.add_argument(
        "--merged",
        action="store_true",
        help="Create a single merged database containing all datasets",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("MySQL -> SQLite Export for CHESS Text2SQL")
    print("=" * 70)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.merged:
        # Merged mode: combine ALL SQL files into one database
        all_sql_files = []
        for config in DATASET_CONFIG.values():
            all_sql_files.extend(config["sql_files"])
        print(f"\n--- MERGED: All {len(DATASET_CONFIG)} datasets combined ---")
        create_sqlite_database(
            db_id="merged",
            sql_files=all_sql_files,
            output_dir=args.output_dir,
        )
    else:
        # Individual mode: one database per dataset
        for dataset_id in args.datasets:
            config = DATASET_CONFIG[dataset_id]
            print(f"\n--- {dataset_id.upper()}: {config['description']} ---")
            create_sqlite_database(
                db_id=dataset_id,
                sql_files=config["sql_files"],
                output_dir=args.output_dir,
            )

    print("=" * 70)
    print("Done! Databases at:", args.output_dir)
    print("=" * 70)


if __name__ == "__main__":
    main()
