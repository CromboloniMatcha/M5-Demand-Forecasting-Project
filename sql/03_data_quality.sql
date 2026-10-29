-- ============================================================
-- 03_data_quality.sql  |  Sanity checks. Run: python -m src.run_sql sql/03_data_quality.sql
-- Every check should come out as expected before you start modelling.
-- ============================================================

-- Q1: Row counts per table
SELECT 'dim_calendar' AS table_name, COUNT(*) AS n_rows FROM dim_calendar
UNION ALL SELECT 'dim_store',           COUNT(*) FROM dim_store
UNION ALL SELECT 'dim_item',            COUNT(*) FROM dim_item
UNION ALL SELECT 'dim_series',          COUNT(*) FROM dim_series
UNION ALL SELECT 'fact_sales',          COUNT(*) FROM fact_sales
UNION ALL SELECT 'fact_prices',         COUNT(*) FROM fact_prices
UNION ALL SELECT 'agg_daily_store_cat', COUNT(*) FROM agg_daily_store_cat
UNION ALL SELECT 'series_profile',      COUNT(*) FROM series_profile;

-- Q2: Is the sales table complete? (series x days should equal row count)
SELECT
    (SELECT COUNT(*) FROM dim_series)                              AS n_series,
    (SELECT COUNT(DISTINCT d_num) FROM fact_sales)                 AS n_days,
    (SELECT COUNT(*) FROM dim_series) * (SELECT COUNT(DISTINCT d_num) FROM fact_sales) AS expected_rows,
    (SELECT COUNT(*) FROM fact_sales)                              AS actual_rows,
    (SELECT COUNT(*) FROM dim_series) * (SELECT COUNT(DISTINCT d_num) FROM fact_sales)
        - (SELECT COUNT(*) FROM fact_sales)                        AS missing_rows;

-- Q3: Calendar coverage and continuity (days_in_range should equal n_days)
SELECT
    MIN(date)                   AS first_date,
    MAX(date)                   AS last_date,
    COUNT(*)                    AS n_days,
    MAX(d_num) - MIN(d_num) + 1 AS days_in_range
FROM dim_calendar;

-- Q4: Invalid sales values (expect 0 negatives)
SELECT
    SUM(CASE WHEN sales < 0 THEN 1 ELSE 0 END) AS negative_sales_rows,
    MIN(sales)                                 AS min_sales,
    MAX(sales)                                 AS max_sales
FROM fact_sales;

-- Q5: Price sanity (expect no zero or negative prices)
SELECT
    COUNT(*)                                       AS n_price_rows,
    SUM(CASE WHEN sell_price <= 0 THEN 1 ELSE 0 END) AS non_positive_prices,
    ROUND(MIN(sell_price), 2)                      AS min_price,
    ROUND(AVG(sell_price), 2)                      AS avg_price,
    ROUND(MAX(sell_price), 2)                      AS max_price
FROM fact_prices;

-- Q6: Items not yet on sale. Days without a price are NOT real zero demand.
-- units_without_price should be ~0, otherwise sales exist where no price exists.
SELECT
    SUM(days_without_price)                                         AS days_without_price,
    SUM(active_days)                                                AS active_days,
    ROUND(100.0 * SUM(days_without_price) / (SUM(days_without_price) + SUM(active_days)), 1) AS pct_days_without_price,
    SUM(units_without_price)                                        AS units_sold_without_price
FROM series_profile;

-- Q7: Series that never sold anything
SELECT COUNT(*) AS series_with_zero_total_units
FROM series_profile
WHERE total_units = 0;

-- Q8: Most extreme single-day sales (check for outliers / data errors)
SELECT sr.store_id, sr.item_id, p.max_units
FROM series_profile p
JOIN dim_series sr ON sr.series_id = p.series_id
ORDER BY p.max_units DESC
LIMIT 5;

-- Q9: How often are calendar events and SNAP days present?
SELECT
    ROUND(100.0 * SUM(CASE WHEN event_name_1 IS NOT NULL THEN 1 ELSE 0 END) / COUNT(*), 1) AS pct_days_with_event,
    ROUND(100.0 * SUM(snap_ca) / COUNT(*), 1)                                           AS pct_snap_ca,
    ROUND(100.0 * SUM(snap_tx) / COUNT(*), 1)                                           AS pct_snap_tx,
    ROUND(100.0 * SUM(snap_wi) / COUNT(*), 1)                                           AS pct_snap_wi
FROM dim_calendar;
