"""Helpers for interactive tables (notebooks) and for exporting tables to the web.

show_table(df)  -> sortable, searchable, filterable table in Jupyter (uses `itables`).
                   Falls back to a normal table if itables is not installed.
save_table(df)  -> writes reports/tables/<name>.csv and dashboard/data/<name>.json,
                   so the notebook and the web read exactly the same numbers.
"""
import pandas as pd

from src import config


def show_table(df, **kwargs):
    try:
        from itables import show
        show(df, **kwargs)
    except Exception:
        try:
            from IPython.display import display
            display(df)
        except ImportError:
            print(df)


def save_table(df, name, web=True):
    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = config.TABLES_DIR / f"{name}.csv"
    df.to_csv(csv_path, index=False)
    paths = [csv_path]
    if web:
        config.DASHBOARD_DATA_DIR.mkdir(parents=True, exist_ok=True)
        json_path = config.DASHBOARD_DATA_DIR / f"{name}.json"
        df.to_json(json_path, orient="records", force_ascii=False, double_precision=4)
        paths.append(json_path)
    return paths
