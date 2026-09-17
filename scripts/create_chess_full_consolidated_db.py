"""Create a 'fully consolidated' CHESS database.

Starts from merged.sqlite (which has all original-named tables fully populated)
and additionally creates the alt-renamed tables from RENAME_MAP, populated with
the SAME full data as their corresponding original tables (with column names
renamed accordingly).

Result: a SQLite where both `field` and `petroleum_deposits` (etc.) hold the
complete row set. Used to evaluate HET-CHESS-SQL — including the queries that
reference alt-named tables — against full data, isolating the data effect.
"""

import sqlite3
import sys
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

# Reuse RENAME_MAP from the existing heterogenization script
from create_chess_heterogeneous_db import RENAME_MAP

CHESS_DB_DIR = PROJECT_ROOT / "tools" / "chess" / "data" / "dev" / "dev_databases"
SOURCE_DB = CHESS_DB_DIR / "merged" / "merged.sqlite"
TARGET_DIR = CHESS_DB_DIR / "merged_full"
TARGET_DB = TARGET_DIR / "merged_full.sqlite"


def main():
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Copying {SOURCE_DB} -> {TARGET_DB}")
    shutil.copyfile(SOURCE_DB, TARGET_DB)

    con = sqlite3.connect(str(TARGET_DB))
    cur = con.cursor()

    # Existing tables in source (the original names)
    existing = {r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}

    created = 0
    skipped = []
    for orig_name, info in RENAME_MAP.items():
        new_name = info["new_name"]
        col_map = info["columns"]  # original_col -> new_col

        if orig_name not in existing:
            skipped.append((orig_name, new_name, "original-table-missing-in-source"))
            continue

        # Source columns in original table
        src_cols_info = cur.execute(f'PRAGMA table_info("{orig_name}")').fetchall()
        src_cols = [r[1] for r in src_cols_info]
        src_types = {r[1]: r[2] for r in src_cols_info}

        # Build CREATE TABLE for new_name from src col types but renamed
        new_col_defs = []
        select_cols = []
        for c in src_cols:
            renamed = col_map.get(c, c)  # if no rename for this col, keep original
            new_col_defs.append(f'"{renamed}" {src_types.get(c) or "TEXT"}')
            select_cols.append(f'"{c}"')

        if new_name in existing:
            cur.execute(f'DROP TABLE "{new_name}"')
        cur.execute(f'CREATE TABLE "{new_name}" ({", ".join(new_col_defs)})')

        # Copy all rows from original to alt
        cur.execute(
            f'INSERT INTO "{new_name}" SELECT {", ".join(select_cols)} FROM "{orig_name}"'
        )
        cnt = cur.execute(f'SELECT COUNT(*) FROM "{new_name}"').fetchone()[0]
        print(f"  {orig_name} -> {new_name}: {cnt} rows")
        created += 1

    con.commit()
    con.close()

    print()
    print(f"Created {created} alt tables in {TARGET_DB}")
    if skipped:
        print(f"Skipped {len(skipped)} (no source table):")
        for o, n, why in skipped:
            print(f"  {o} -> {n}: {why}")


if __name__ == "__main__":
    main()