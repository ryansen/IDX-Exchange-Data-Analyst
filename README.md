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
