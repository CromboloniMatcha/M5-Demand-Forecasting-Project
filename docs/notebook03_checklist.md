# Notebook 03 checklist: LightGBM model

If you changed `STATES` in `config.py`, set it again.

## Steps
- [ ] `pip install -r requirements.txt`
- [ ] Make sure `reports/tables/series_segments.csv` and `baseline_metrics.csv` exist (from `python -m src.run_week2`)
- [ ] First, a quicker run: `python -m src.run_lgbm --train-origins 26`
- [ ] Open `notebooks/03_lightgbm_model.ipynb` and run it top to bottom
- [ ] If it looks right, do the full run: `python -m src.run_lgbm` (about 1 year of training origins)
- [ ] Re-run the notebook, then fill in the "Key findings" cell with your own numbers

## How the evaluation is kept honest
- Same 3 backtest origins and same evaluated series as notebook 02, so numbers are directly comparable.
- For each origin: training targets end before the validation window, and validation ends before the test window.
- Features use only data before the forecast origin. Only the calendar and price of the target day are used from the future, because both are known in advance.
- The model chooses its number of trees by early stopping on the validation window, never on the test window.

## Deliverables
1. A global LightGBM model with a documented, leakage-free backtest
2. A comparison with every baseline, overall and per segment
3. Accuracy by day ahead, and at aggregate level
4. Feature importance and SHAP explanations
5. `series_scoreboard`, `model_comparison_overall` and `model_comparison_by_segment` as CSV and JSON for the web
