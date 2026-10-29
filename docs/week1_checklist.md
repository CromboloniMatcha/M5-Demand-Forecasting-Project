# Week 1 checklist: data, SQL, EDA

- [ ] Create a Kaggle account, accept the M5 competition rules, download the data
- [ ] Put `calendar.csv`, `sell_prices.csv`, `sales_train_evaluation.csv` in `data/raw/`
- [ ] Create the virtual environment and install `requirements.txt`
- [ ] Test the pipeline on synthetic data (see README, "Try it first")
- [ ] Build the real database: `python -m src.build_database`
- [ ] Run `sql/03_data_quality.sql` and confirm: no missing rows, no negative sales, 0 units sold without a price
- [ ] Run `sql/04_eda_queries.sql` and read every result
- [ ] Run `notebooks/01_eda.ipynb` top to bottom on the real data
- [ ] Write the "Key findings" cell with your own numbers
- [ ] Commit to GitHub (data files are git-ignored), with the figures from `reports/figures/`

## Deliverables at the end of week 1

1. A working database build from raw CSV
2. SQL files for schema, data quality and business questions
3. An EDA notebook with saved charts and written findings
4. A README that someone else can follow
