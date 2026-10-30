# Notebook 04 checklist: inventory simulation

If you changed `STATES` in `config.py`, set it again.

## Steps
- [ ] Re-run LightGBM once so it saves the forecasts of every origin: `python -m src.run_lgbm`
- [ ] Run the simulation: `python -m src.run_inventory`
- [ ] Open `notebooks/04_inventory_simulation.ipynb` and run it top to bottom
- [ ] Optional: try another setting, for example `python -m src.run_inventory --lead-time 14 --service-level 0.98`
- [ ] Fill in the "Key findings" cell with your own numbers

## How to read the results
- Lead with the **service vs inventory trade-off** and the **equal-service comparison**. They need no cost guesses.
- Show costs only together with the **assumptions** and the **sensitivity grid**.
- The **oracle** is the best any forecast could do. "Share of oracle gap closed" is an honest way to say how much of the possible gain is captured.

## Assumptions to state in the README and in interviews
- M5 has no stock, cost or lead-time data, so lead time (7 days), review period (7 days), unit cost (70% of price), holding rate (20% per year) and stockout penalty (0) are assumptions.
- Periodic-review order-up-to policy, lost sales (no backorders), series treated independently.
- Forecasts refreshed once per 28-day origin and used for the whole window.
- Safety stock uses each forecast's own past errors, measured only on windows before the simulated one.
- Two simulated 28-day windows (backtest origins 2 and 3). Origin 1 is only used to measure errors.
- Weekday patterns do not matter here: with weekly ordering the 14-day cover always contains each weekday twice.

## Deliverables
1. A leakage-free inventory simulation comparing forecasts under the same policy
2. Trade-off curves, an equal-service comparison and a cost comparison against an oracle
3. Savings by segment and a sensitivity check on the cost assumptions
4. A replenishment plan (safety stock, order-up-to, reorder point) for every series
5. Tables as CSV and JSON for the web
