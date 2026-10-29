-- ============================================================
-- 04_eda_queries.sql  |  Business questions answered in SQL
-- Run: python -m src.run_sql sql/04_eda_queries.sql
-- Uses CTEs and window functions (LAG, ROW_NUMBER, running SUM).
-- ============================================================

-- Q1: Headline KPIs
SELECT
    COUNT(DISTINCT store_id)                     AS stores,
    ROUND(SUM(units), 0)                         AS total_units,
    ROUND(SUM(revenue), 0)                       AS total_revenue,
    ROUND(SUM(revenue) / SUM(units), 2)          AS avg_price_per_unit
FROM agg_daily_store_cat;

-- Q2: Monthly revenue and month-over-month growth (partial months with under 28 days are excluded)
WITH monthly AS (
    SELECT c.year, c.month, SUM(a.revenue) AS revenue, SUM(a.units) AS units
    FROM agg_daily_store_cat a
    JOIN dim_calendar c ON c.d_num = a.d_num
    GROUP BY c.year, c.month
    HAVING COUNT(DISTINCT a.d_num) >= 28
)
SELECT
    year,
    month,
    ROUND(revenue, 0) AS revenue,
    units,
    ROUND(100.0 * (revenue - LAG(revenue) OVER (ORDER BY year, month))
          / LAG(revenue) OVER (ORDER BY year, month), 1) AS mom_growth_pct
FROM monthly
ORDER BY year, month;

-- Q3: Revenue share by category and store
SELECT
    store_id,
    cat_id,
    ROUND(SUM(revenue), 0) AS revenue,
    ROUND(100.0 * SUM(revenue) / SUM(SUM(revenue)) OVER (), 1) AS pct_of_total
FROM agg_daily_store_cat
GROUP BY store_id, cat_id
ORDER BY revenue DESC;

-- Q4: Day-of-week demand index (100 = average day)
WITH by_day AS (
    SELECT c.wday, c.weekday, SUM(a.units) AS units, COUNT(DISTINCT a.d_num) AS n_days
    FROM agg_daily_store_cat a
    JOIN dim_calendar c ON c.d_num = a.d_num
    GROUP BY c.wday, c.weekday
)
SELECT
    weekday,
    ROUND(1.0 * units / n_days, 0) AS avg_units_per_day,
    ROUND(100.0 * (1.0 * units / n_days) / (SELECT 1.0 * SUM(units) / SUM(n_days) FROM by_day), 1) AS demand_index
FROM by_day
ORDER BY wday;

-- Q5: SNAP effect - average units per active series per day, SNAP vs non-SNAP days
SELECT
    cat_id,
    CASE WHEN snap = 1 THEN 'SNAP day' ELSE 'non-SNAP day' END AS day_type,
    ROUND(1.0 * SUM(units) / SUM(n_active), 4) AS units_per_active_series_day
FROM (
    SELECT a.cat_id, a.units, a.n_active,
           CASE st.state_id WHEN 'CA' THEN c.snap_ca
                            WHEN 'TX' THEN c.snap_tx
                            ELSE c.snap_wi END AS snap
    FROM agg_daily_store_cat a
    JOIN dim_store    st ON st.store_id = a.store_id
    JOIN dim_calendar c  ON c.d_num     = a.d_num
)
GROUP BY cat_id, day_type
ORDER BY cat_id, day_type;

-- Q6: Calendar events - average units per active series per day by event type
SELECT
    COALESCE(c.event_type_1, 'No event') AS event_type,
    COUNT(DISTINCT a.d_num)              AS n_days,
    ROUND(1.0 * SUM(a.units) / SUM(a.n_active), 4) AS units_per_active_series_day
FROM agg_daily_store_cat a
JOIN dim_calendar c ON c.d_num = a.d_num
GROUP BY COALESCE(c.event_type_1, 'No event')
ORDER BY units_per_active_series_day DESC;

-- Q7: Top 5 series by revenue in each store
WITH ranked AS (
    SELECT
        sr.store_id,
        sr.item_id,
        p.total_revenue,
        ROW_NUMBER() OVER (PARTITION BY sr.store_id ORDER BY p.total_revenue DESC) AS rank_in_store
    FROM series_profile p
    JOIN dim_series sr ON sr.series_id = p.series_id
)
SELECT store_id, rank_in_store, item_id, ROUND(total_revenue, 0) AS total_revenue
FROM ranked
WHERE rank_in_store <= 5
ORDER BY store_id, rank_in_store;

-- Q8: ABC preview (Pareto) - A = items making the first 80% of revenue, B = next 15%, C = last 5%
WITH ranked AS (
    SELECT
        series_id,
        total_revenue,
        SUM(total_revenue) OVER (ORDER BY total_revenue DESC, series_id
                                 ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS cum_revenue,
        SUM(total_revenue) OVER ()  AS grand_total,
        COUNT(*) OVER ()            AS n_series
    FROM series_profile
)
SELECT
    CASE WHEN cum_revenue <= 0.80 * grand_total THEN 'A'
         WHEN cum_revenue <= 0.95 * grand_total THEN 'B'
         ELSE 'C' END                                   AS abc_class,
    COUNT(*)                                            AS n_series,
    ROUND(100.0 * COUNT(*) / MAX(n_series), 1)          AS pct_of_series,
    ROUND(100.0 * SUM(total_revenue) / MAX(grand_total), 1) AS pct_of_revenue
FROM ranked
GROUP BY abc_class
ORDER BY abc_class;

-- Q9: Intermittent demand - share of active days with zero sales, per series
SELECT
    CASE WHEN 1.0 * zero_days / active_days < 0.25 THEN '1) 0-25% zero days'
         WHEN 1.0 * zero_days / active_days < 0.50 THEN '2) 25-50% zero days'
         WHEN 1.0 * zero_days / active_days < 0.75 THEN '3) 50-75% zero days'
         ELSE '4) 75-100% zero days' END AS zero_share_bucket,
    COUNT(*) AS n_series,
    ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM series_profile WHERE active_days > 0), 1) AS pct_of_series
FROM series_profile
WHERE active_days > 0
GROUP BY zero_share_bucket
ORDER BY zero_share_bucket;

-- Q10: How often do prices change? (number of weekly price changes per series)
WITH chg AS (
    SELECT
        series_id,
        CASE WHEN sell_price <> LAG(sell_price) OVER (PARTITION BY series_id ORDER BY wm_yr_wk)
             THEN 1 ELSE 0 END AS changed
    FROM fact_prices
),
per_series AS (
    SELECT series_id, SUM(changed) AS n_changes FROM chg GROUP BY series_id
)
SELECT
    CASE WHEN n_changes = 0 THEN '0 changes'
         WHEN n_changes <= 2 THEN '1-2 changes'
         WHEN n_changes <= 5 THEN '3-5 changes'
         ELSE '6+ changes' END AS price_changes,
    COUNT(*) AS n_series,
    ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM per_series), 1) AS pct_of_series
FROM per_series
GROUP BY price_changes
ORDER BY price_changes;
