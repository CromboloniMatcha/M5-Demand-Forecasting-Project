-- ============================================================
-- 02_indexes_views.sql  |  Run after the data is loaded
-- Builds indexes, one enriched view, and two pre-aggregated tables
-- that make EDA and later feature engineering fast.
-- ============================================================

CREATE UNIQUE INDEX idx_fact_sales_pk  ON fact_sales (series_id, d_num);
CREATE UNIQUE INDEX idx_fact_prices_pk ON fact_prices (series_id, wm_yr_wk);

-- Row-level view: sales + product + store + calendar + price in one place.
-- "active" rows are those where a price exists (the item was really on sale).
CREATE VIEW v_sales_enriched AS
SELECT
    s.series_id,
    sr.item_id,
    sr.store_id,
    st.state_id,
    i.dept_id,
    i.cat_id,
    s.d_num,
    c.date,
    c.year,
    c.month,
    c.wday,
    c.weekday,
    c.wm_yr_wk,
    c.event_name_1,
    c.event_type_1,
    CASE st.state_id WHEN 'CA' THEN c.snap_ca
                     WHEN 'TX' THEN c.snap_tx
                     ELSE c.snap_wi END          AS snap,
    p.sell_price,
    s.sales,
    s.sales * p.sell_price                       AS revenue,
    CASE WHEN p.sell_price IS NULL THEN 0 ELSE 1 END AS is_active
FROM fact_sales s
JOIN dim_series   sr ON sr.series_id = s.series_id
JOIN dim_store    st ON st.store_id  = sr.store_id
JOIN dim_item     i  ON i.item_id    = sr.item_id
JOIN dim_calendar c  ON c.d_num      = s.d_num
LEFT JOIN fact_prices p
       ON p.series_id = s.series_id AND p.wm_yr_wk = c.wm_yr_wk;

-- Daily totals by store and category (small table, used for dashboards and EDA).
CREATE TABLE agg_daily_store_cat AS
SELECT
    sr.store_id,
    i.cat_id,
    s.d_num,
    SUM(s.sales)                                          AS units,
    SUM(s.sales * p.sell_price)                           AS revenue,
    COUNT(*)                                              AS n_series,
    SUM(CASE WHEN p.sell_price IS NULL THEN 0 ELSE 1 END) AS n_active
FROM fact_sales s
JOIN dim_series   sr ON sr.series_id = s.series_id
JOIN dim_item     i  ON i.item_id    = sr.item_id
JOIN dim_calendar c  ON c.d_num      = s.d_num
LEFT JOIN fact_prices p
       ON p.series_id = s.series_id AND p.wm_yr_wk = c.wm_yr_wk
GROUP BY sr.store_id, i.cat_id, s.d_num;

CREATE INDEX idx_agg_daily_d ON agg_daily_store_cat (d_num);

-- One row per series: launch timing, volume, intermittency. This is the base for
-- ABC-XYZ segmentation (week 2). Standard deviation is computed later in pandas
-- from sum_units_active and sumsq_units_active.
CREATE TABLE series_profile AS
SELECT
    s.series_id,
    MIN(CASE WHEN p.sell_price IS NOT NULL THEN s.d_num END)          AS first_price_d,
    MIN(CASE WHEN s.sales > 0 THEN s.d_num END)                        AS first_sale_d,
    SUM(CASE WHEN p.sell_price IS NOT NULL THEN 1 ELSE 0 END)          AS active_days,
    SUM(CASE WHEN p.sell_price IS NULL THEN 1 ELSE 0 END)              AS days_without_price,
    SUM(CASE WHEN p.sell_price IS NOT NULL AND s.sales = 0 THEN 1 ELSE 0 END) AS zero_days,
    SUM(s.sales)                                                       AS total_units,
    SUM(CASE WHEN p.sell_price IS NOT NULL THEN s.sales ELSE 0 END)    AS sum_units_active,
    SUM(CASE WHEN p.sell_price IS NOT NULL THEN s.sales * s.sales ELSE 0 END) AS sumsq_units_active,
    SUM(CASE WHEN p.sell_price IS NULL THEN s.sales ELSE 0 END)        AS units_without_price,
    MAX(s.sales)                                                       AS max_units,
    COALESCE(SUM(s.sales * p.sell_price), 0)                           AS total_revenue
FROM fact_sales s
JOIN dim_calendar c ON c.d_num = s.d_num
LEFT JOIN fact_prices p
       ON p.series_id = s.series_id AND p.wm_yr_wk = c.wm_yr_wk
GROUP BY s.series_id;

CREATE UNIQUE INDEX idx_series_profile_pk ON series_profile (series_id);

ANALYZE;
