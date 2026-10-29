"""Project-wide settings.

Change STATES to load a different subset of the M5 data.
M5 has 3 states: CA (4 stores), TX (3 stores), WI (3 stores).
CA alone is ~12,000 item-store series x 1,941 days = ~24 million rows,
which is large enough to be realistic but still fine on a laptop.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = Path(os.environ.get("M5_RAW_DIR", ROOT / "data" / "raw"))
PROCESSED_DIR = ROOT / "data" / "processed"
DB_PATH = Path(os.environ.get("M5_DB", PROCESSED_DIR / "m5.db"))
SQL_DIR = ROOT / "sql"
FIG_DIR = ROOT / "reports" / "figures"

# Which states to load into the database
STATES = ["CA"]

# Raw file names (the evaluation file is preferred; it has 1,941 days)
SALES_FILES = ["sales_train_evaluation.csv", "sales_train_validation.csv"]
CALENDAR_FILE = "calendar.csv"
PRICES_FILE = "sell_prices.csv"

# How many wide rows (series) to melt and insert at a time
CHUNK_ROWS = 500
