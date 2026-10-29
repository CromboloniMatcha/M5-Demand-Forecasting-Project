# Data dictionary

A **series** is one item sold in one store. All dates are ISO text (`YYYY-MM-DD`).

## Tables

| Table | Grain | Key columns |
|---|---|---|
| `dim_calendar` | one row per day | `d_num` (1..N), `date`, `wm_yr_wk`, `wday` (1 = Saturday), `event_name_1/2`, `event_type_1/2`, `snap_ca/tx/wi` |
| `dim_store` | one row per store | `store_id` (e.g. CA_1), `state_id` |
| `dim_item` | one row per item | `item_id`, `dept_id`, `cat_id` (FOODS / HOBBIES / HOUSEHOLD) |
| `dim_series` | one row per item-store | `series_id` (surrogate), `item_id`, `store_id` |
| `fact_sales` | series x day | `series_id`, `d_num`, `sales` (units) |
| `fact_prices` | series x week | `series_id`, `wm_yr_wk`, `sell_price` |

## Derived objects

| Object | What it is |
|---|---|
| `v_sales_enriched` | Sales joined to item, store, calendar and price. `is_active = 1` when a price exists. `snap` is the SNAP flag of the store's own state. |
| `agg_daily_store_cat` | Daily units and revenue per store and category, plus `n_series` and `n_active` (series on sale that day). |
| `series_profile` | One row per series: first price day, first sale day, `active_days`, `zero_days`, `total_units`, `total_revenue`, `max_units`, and sums for computing mean and standard deviation. |

## Notes

- **Not-yet-launched items:** days with no price are not zero demand. Use `active_days` and `is_active`.
- **SNAP:** government food-assistance purchase days, which differ by state. Mostly relevant for FOODS.
- **`wm_yr_wk`:** Walmart's week id, used to join weekly prices to daily sales.
