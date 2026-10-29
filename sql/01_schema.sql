-- ============================================================
-- 01_schema.sql  |  Star schema for the M5 demand forecasting project
-- Engine: SQLite (runs anywhere, no server needed)
-- A "series" = one item sold in one store (the unit we forecast).
-- ============================================================

DROP VIEW  IF EXISTS v_sales_enriched;
DROP TABLE IF EXISTS series_profile;
DROP TABLE IF EXISTS agg_daily_store_cat;
DROP TABLE IF EXISTS fact_prices;
DROP TABLE IF EXISTS fact_sales;
DROP TABLE IF EXISTS dim_series;
DROP TABLE IF EXISTS dim_item;
DROP TABLE IF EXISTS dim_store;
DROP TABLE IF EXISTS dim_calendar;

CREATE TABLE dim_calendar (
    d_num        INTEGER PRIMARY KEY,   -- 1..1941 (day index, "d_1" in the raw file)
    date         TEXT NOT NULL,         -- ISO date, YYYY-MM-DD
    wm_yr_wk     INTEGER NOT NULL,      -- Walmart week id, joins to prices
    weekday      TEXT,
    wday         INTEGER,               -- 1 = Saturday ... 7 = Friday (M5 convention)
    month        INTEGER,
    year         INTEGER,
    event_name_1 TEXT,
    event_type_1 TEXT,
    event_name_2 TEXT,
    event_type_2 TEXT,
    snap_ca      INTEGER,               -- 1 if SNAP food-stamp purchases allowed that day
    snap_tx      INTEGER,
    snap_wi      INTEGER
);

CREATE TABLE dim_store (
    store_id TEXT PRIMARY KEY,          -- e.g. CA_1
    state_id TEXT NOT NULL              -- CA / TX / WI
);

CREATE TABLE dim_item (
    item_id TEXT PRIMARY KEY,           -- e.g. FOODS_3_090
    dept_id TEXT NOT NULL,              -- e.g. FOODS_3
    cat_id  TEXT NOT NULL               -- FOODS / HOBBIES / HOUSEHOLD
);

CREATE TABLE dim_series (
    series_id INTEGER PRIMARY KEY,      -- surrogate key (keeps the big fact tables small)
    item_id   TEXT NOT NULL REFERENCES dim_item(item_id),
    store_id  TEXT NOT NULL REFERENCES dim_store(store_id),
    UNIQUE (item_id, store_id)
);

-- Daily units sold per series (long format). Indexes are added after loading.
CREATE TABLE fact_sales (
    series_id INTEGER NOT NULL,
    d_num     INTEGER NOT NULL,
    sales     INTEGER NOT NULL
);

-- Weekly selling price per series. A missing row means the item was NOT on sale
-- yet in that store, so earlier zero sales are not real "zero demand".
CREATE TABLE fact_prices (
    series_id  INTEGER NOT NULL,
    wm_yr_wk   INTEGER NOT NULL,
    sell_price REAL NOT NULL
);
