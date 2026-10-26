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
through August 2026 with mortgage rates. This one command writes both `_Cleaned.csv`
and the authoritative `_Cleaned_Final.csv`, with quality reports in
`reports/weeks_4_5/` and final decisions in `reports/weeks_4_5/final/`. To use another file, edit
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
The initial cleaning leaves missing values null; final preparation then applies
the investigated policy below. The script documents each transformation and why.
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

## Missing-data experiments and preparation

Notebook sections 26–37 explain MCAR, MAR and NMAR, show missingness by property
type, compare deletion and imputation, and document field-level decisions. The
default displays saved results after checking the input file hash, experimental
code and configuration. Set `RERUN_EXPERIMENTS = True` to repeat model fitting.
The existing common-statistics section still uses observed cleaned values.

To reproduce the experiments and create a separate analysis copy:

```bash
python python/weeks_4_5_missing_data_experiments.py
```

The experiments require scikit-learn (included in `requirements.txt`). They use
group-separated holdouts, train-only preprocessing, three split seeds, two
masking seeds and four mechanisms: MCAR, MAR, NMAR_high and NMAR_low. For each
of 11 fields, 20% of observed holdout values are hidden and four methods compared.
These simulations do not identify the true missingness mechanism.

Reports and the explanation are saved in `reports/weeks_4_5/missing_data/`.
The exploratory `_Cleaned_AnalysisReady.csv` preserves all rows and the original cleaned
file. Nine mostly/entirely empty fields are removed. Only supported missing
OriginalListPrice and BathroomsTotalInteger values are imputed after passing
the experimental screen; every estimate has an audit record and flag. Other
gaps remain explicit. A price-analysis eligibility flag excludes unknown prices
from price calculations without deleting those records from the master view.
Imputation is for exploratory predictors, not confidence intervals or tail estimates.

## Final missing-data decisions

The final deliverable is `_Cleaned_Final.csv`, produced by the main cleaning
script. The helper `python/weeks_4_5_missing_data_finalization.py` applies the
policy and records the follow-up investigations. Notebook sections 38–41 show
the authoritative final results; sections 26–37 retain the exploratory experiments.

- Retain all 480,486 records; remove the nine sparse fields from the final view.
- Impute only 58 supported bathroom counts; retain original reported counts in
  `BathroomsTotalInteger_observed`. Keep the 12 unknown BoatSlip counts null.
- Leave original list price missing. This revises the earlier exploratory choice:
  the model's high-price NMAR mean bias exceeded the final 5% tolerance and the
  actual missing-price cohort differs substantially from the observed cohort.
- Derive 77 missing acre values and 56 square-foot values from the other explicit
  unit using 43,560 sq ft per acre. Audit and flag conversions; do not assume the
  unit of `LotSizeArea`. Flag conflicting observed lot-size pairs without overwriting them.
- Leave all other unsupported gaps unknown. Never substitute zero for unknown HOA
  fees, false for missing booleans, or guessed dates/coordinates.

Missing-data explanations, counts, decision ledger, source coverage and audits are
in `reports/weeks_4_5/final/`. Reload final types with `final_schema.json`. Eligibility
flags support price, price/area, days-on-market, map and timeline analyses without
deleting rows from the master file. The missingness mechanism and HOA applicability
cannot be established from this export alone; those uncertainties have explicit
conservative handling in the final policy.

## Final handbook check and meeting summary

```bash
python python/weeks_4_5_final_check.py
```

This reloads the complete final CSV using its schema, verifies all four date
types and numeric fields, recalculates invalid-number/date/geographic flags,
and reconciles row counts and reports. It writes `requirements_checklist.csv`,
`data_types_confirmed.csv`, `requirements_check_result.json`, and a bullet-point
`meeting_summary.md` in `reports/weeks_4_5/final/`.

The notebook now has a reading guide and separate required-cleaning, exploratory
experiment and final-decision sections. Section 42 displays the final checklist
and meeting summary. It identifies the authoritative Cleaned_Final CSV clearly;
the earlier exploratory AnalysisReady file is retained only for comparison.

```bash
python -m unittest discover -s python -p "test_missing_data_experiments.py"
```
