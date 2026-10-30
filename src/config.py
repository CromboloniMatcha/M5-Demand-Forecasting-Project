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
TABLES_DIR = ROOT / "reports" / "tables"

# ---------------------------------------------------------------- Week 1: loading
# Which states to load into the database
STATES = ["CA"]

# Raw file names (the evaluation file is preferred; it has 1,941 days)
SALES_FILES = ["sales_train_evaluation.csv", "sales_train_validation.csv"]
CALENDAR_FILE = "calendar.csv"
PRICES_FILE = "sell_prices.csv"

# How many wide rows (series) to melt and insert at a time
CHUNK_ROWS = 500

# ---------------------------------------------------------------- Week 2: evaluation
HORIZON = 28            # forecast horizon in days (same as the M5 competition)
N_ORIGINS = 3           # rolling-origin backtests; the last test window is the last 28 days
MIN_HISTORY_DAYS = 56   # a series needs this many active days before the cutoff to be evaluated

# ---------------------------------------------------------------- Week 2: segmentation
SEG_WINDOW_DAYS = 364   # look-back window (52 weeks) used to classify series
MIN_SEG_DAYS = 90       # fewer active days than this in the window -> label "NEW"
ABC_CUTS = (0.80, 0.95)  # cumulative revenue share: A up to 80%, B up to 95%, C the rest
XYZ_CUTS = (0.5, 1.0)    # coefficient of variation of weekly demand: X < 0.5, Y < 1.0, Z otherwise
SBC_ADI_CUT = 1.32       # Syntetos-Boylan-Croston demand classification cut-offs
SBC_CV2_CUT = 0.49

# ---------------------------------------------------------------- Week 2: models
CROSTON_ALPHA = 0.1
ETS_SAMPLE_SIZE = 300       # series sampled for ETS (slow to fit on every series)
STATS_MAX_TRAIN_DAYS = 730  # ETS / SARIMA only look at the most recent 2 years

# ---------------------------------------------------------------- Notebook 03: LightGBM
DASHBOARD_DATA_DIR = ROOT / "dashboard" / "data"   # JSON copies of result tables for the web
MODELS_DIR = ROOT / "models"

LGB_TRAIN_ORIGINS = 52     # weekly forecast origins used for training (about 1 year of history)
LGB_ORIGIN_STEP = 7
LGB_H_PER_ORIGIN = 4       # horizons sampled per series and origin (keeps the training set small)
LGB_MAX_ROUNDS = 1500
LGB_EARLY_STOP = 50
LGB_SHAP_ROWS = 10_000     # rows used for SHAP values
LGB_PARAMS = {
    "objective": "tweedie",            # standard choice for zero-inflated sales counts
    "tweedie_variance_power": 1.1,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "seed": 42,
    "verbose": -1,
    "num_threads": 0,                  # 0 = use all cores
}
