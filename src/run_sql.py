"""Run a .sql file and print the result of every SELECT.

Usage (from the project root):
    python -m src.run_sql sql/03_data_quality.sql
    python -m src.run_sql sql/04_eda_queries.sql --db data/processed/m5_sample.db

The comment line right above a query is used as its title.
"""
import argparse
import sqlite3
from pathlib import Path

import pandas as pd

from src import config


def split_statements(text):
    """Split SQL text into (title, statement) pairs, using the comment above each statement as title."""
    statements, buf, title = [], [], None
    for line in text.splitlines():
        stripped = line.strip()
        if not buf and stripped.startswith("--"):
            comment = stripped.lstrip("-").strip()
            if title is None and comment and not set(comment) <= {"="}:
                title = comment
            continue
        if not buf and not stripped:
            continue
        buf.append(line)
        candidate = "\n".join(buf)
        if sqlite3.complete_statement(candidate):
            statements.append((title, candidate))
            buf, title = [], None
    return statements


def main():
    parser = argparse.ArgumentParser(description="Run a SQL file against the M5 database")
    parser.add_argument("sql_file", type=Path)
    parser.add_argument("--db", type=Path, default=config.DB_PATH)
    parser.add_argument("--max-rows", type=int, default=40)
    args = parser.parse_args()

    if not args.db.exists():
        raise SystemExit(f"Database not found: {args.db}\nRun first: python -m src.build_database")

    pd.set_option("display.width", 160)
    con = sqlite3.connect(args.db)
    for title, sql in split_statements(args.sql_file.read_text(encoding="utf-8")):
        first_word = sql.lstrip().split(None, 1)[0].upper()
        if first_word in ("SELECT", "WITH"):
            df = pd.read_sql_query(sql, con)
            print(f"\n=== {title or 'query'} ===")
            print(df.head(args.max_rows).to_string(index=False))
            if len(df) > args.max_rows:
                print(f"... ({len(df):,} rows, showing {args.max_rows})")
        else:
            con.execute(sql)
    con.commit()
    con.close()


if __name__ == "__main__":
    main()
