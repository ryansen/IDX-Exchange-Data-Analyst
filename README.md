# IDX Exchange Data Analyst

Python utilities for exporting, combining, and validating CRMLS listing and sold
datasets.

## Data handling

Raw MLS exports and generated analysis outputs are local-only. Do not commit
files in `csv/`, `reports/`, or any tabular data files such as `*.csv` to
GitHub. The repository is configured to ignore those files, and the included
pre-commit hook blocks CSVs and common secret file types if they are staged by
mistake.

To enable the hook in this clone:

```bash
git config core.hooksPath .githooks
```

## Local credentials

The export scripts read the IDX Exchange token key from an environment variable:

```bash
IDXEXCHANGE_TRESTLE_TOKEN_KEY=your-local-token-key
```

Use `.env.example` as a template for local setup. Keep real `.env` files and
token keys out of Git.

## Setup

```bash
pip install -r requirements.txt
```

## Weeks 4–5: data cleaning and preparation

Run the cleaning script after the Weeks 2–3 mortgage enrichment:

```bash
python python/weeks_4_5_data_cleaning.py
```

The default input is the combined Residential sold dataset for January 2024
through August 2026 with mortgage rates. The script saves a new `_Cleaned.csv`
beside it and quality reports in `reports/weeks_4_5/`. To use another file, edit
`INPUT_FILE`, `OUTPUT_FILE`, and `REPORT_DIR` near the top of the script.
The default encoding is UTF-8, matching the generated combined CSVs. For a raw
Windows export, use `run(encoding="cp1252")`.

For Jupyter, open `python/weeks_4_5_data_cleaning.ipynb` and choose **Run All**.
It expands the script's cleaning rules into 25 short code cells with data previews,
type confirmations and quality tables. Charts show numeric errors, date violations,
missingness before/after cleaning, geographic issues and numeric distributions.
A printed comparison summary reports counts, mean, median, standard deviation,
minimum and maximum for sale price, living area, days on market, bedrooms,
bathrooms and price per square foot. You can run it from the project folder or the `python/` folder.
The last cells export the cleaned data and reports without repeating the cleaning.
The variables `sold` and `reports` remain available for inspection. Notebooks and
their outputs remain local and ignored by Git because they can contain MLS data.

Every row is retained. Invalid numeric measurements and unusable coordinate
pairs become null with boolean flags; dates are checked without reordering them.
Missing values are left null. The script documents each transformation and why.
The summary includes row counts, date flag counts and geographic screening
results. Field reports confirm types, null counts and conversion failures;
separate audits retain invalid tokens and coordinates with source record numbers.
The geographic envelope is approximate and does not establish exact state
membership. Use `schema.json` when loading the CSV to restore dates, nullable
numbers, booleans and string identifiers.

Run the focused cleaning checks with:

```bash
python -m unittest discover -s python -p "test_weeks_4_5_data_cleaning.py"
```
