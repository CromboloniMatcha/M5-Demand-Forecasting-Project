# Notebook 2 checklist: segmentation and baselines

## Steps
- [ ] `pip install -r requirements.txt` (installs statsmodels)
- [ ] `python -m src.stat_models` should print ETS and SARIMA forecasts (checks statsmodels works)
- [ ] `python -m src.run_week2` (fast: segmentation + 5 baselines, a few minutes the first time)
- [ ] `python -m src.run_week2 --with-stats` (adds ETS on a sample and ETS/SARIMA on aggregates)
- [ ] Open `notebooks/02_baselines.ipynb` and run it top to bottom
- [ ] Fill in the "Key findings" cell with numbers from your own tables
- [ ] Commit `reports/tables/*.csv` and `reports/figures/*.png` to GitHub

## Deliverables at the end of notebook 2
1. ABC, XYZ and demand-pattern segment for every series (`series_segments.csv`)
2. A backtest with 3 rolling origins x 28 days for 5 baselines (+ ETS, SARIMA)
3. Accuracy tables by segment, and the best method per segment
4. The target number the week 3 LightGBM model has to beat

## Method notes (good to mention in the README)
- Segments use only data before the first backtest cutoff, so they never see a test window.
- Days before an item's first price are treated as "not on sale", not as zero demand.
- RMSSE is bottom-level only; the official M5 WRMSSE also averages 12 aggregation levels.
- ETS is run on a stratified sample (fitting every series one by one is too slow). Baselines are scored on the same sample for a fair comparison.
- SARIMA uses fixed orders (1,0,1)(1,1,1,7), not automatic order selection, and runs on aggregated series only.
