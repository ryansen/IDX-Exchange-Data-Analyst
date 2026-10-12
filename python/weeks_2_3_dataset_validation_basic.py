"""
Weeks 2-3 deliverable: dataset structuring, validation, and EDA.

This script loads the raw monthly CRMLS sold files from January 2024 through
August 2026, documents the property types found, filters to Residential records,
creates missing-value and numeric-distribution reports, and saves the filtered
dataset as a new CSV.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd


START_MONTH = "202401"
END_MONTH = "202608"

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "csv"
REPORT_DIR = PROJECT_DIR / "reports" / "weeks_2_3"

FILTERED_OUTPUT = DATA_DIR / f"CRMLSSold_{START_MONTH}_{END_MONTH}_Residential_Filtered.csv"

CORE_FIELDS = {
    "ListingKey",
    "ListingId",
    "PropertyType",
    "PropertySubType",
    "MlsStatus",
    "CloseDate",
    "ClosePrice",
    "ListPrice",
    "OriginalListPrice",
    "ListingContractDate",
    "PurchaseContractDate",
    "DaysOnMarket",
    "LivingArea",
    "LotSizeAcres",
    "BedroomsTotal",
    "BathroomsTotalInteger",
    "YearBuilt",
    "CountyOrParish",
    "City",
    "StateOrProvince",
    "PostalCode",
    "Latitude",
    "Longitude",
}

MARKET_ANALYSIS_FIELDS = {
    "CloseDate",
    "ClosePrice",
    "ListPrice",
    "OriginalListPrice",
    "ListingContractDate",
    "PurchaseContractDate",
    "DaysOnMarket",
    "LivingArea",
    "LotSizeAcres",
    "LotSizeSquareFeet",
    "BedroomsTotal",
    "BathroomsTotalInteger",
    "YearBuilt",
    "PropertyType",
    "PropertySubType",
    "CountyOrParish",
    "City",
    "StateOrProvince",
    "PostalCode",
    "Latitude",
    "Longitude",
}

NUMERIC_FIELDS = [
    "ClosePrice",
    "ListPrice",
    "OriginalListPrice",
    "LivingArea",
    "LotSizeAcres",
    "BedroomsTotal",
    "BathroomsTotalInteger",
    "DaysOnMarket",
    "YearBuilt",
]

REQUIRED_NUMERIC_SUMMARY_FIELDS = ["ClosePrice", "LivingArea", "DaysOnMarket"]


def month_range(start_month: str, end_month: str) -> list[str]:
    year = int(start_month[:4])
    month = int(start_month[4:])
    end_year = int(end_month[:4])
    end_month_number = int(end_month[4:])

    months = []
    while (year, month) <= (end_year, end_month_number):
        months.append(f"{year}{month:02d}")
        month += 1
        if month == 13:
            month = 1
            year += 1
    return months


def find_monthly_file(month: str) -> Path:
    candidates = [
        DATA_DIR / f"CRMLSSold{month}.csv",
        DATA_DIR / f"CRMLSSold{month}_filled.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Missing sold file for {month}")


def load_sold_data() -> pd.DataFrame:
    frames = []
    for month in month_range(START_MONTH, END_MONTH):
        path = find_monthly_file(month)
        df = pd.read_csv(path, encoding="cp1252", encoding_errors="replace", low_memory=False)
        df.insert(0, "SourceMonth", month)
        df.insert(1, "SourceFile", path.name)
        frames.append(df)
        print(f"Loaded {path.name}: {len(df):,} rows")

    sold = pd.concat(frames, ignore_index=True, sort=False)
    print(f"Rows after concatenation: {len(sold):,}")
    return sold


def add_field_classification(null_report: pd.DataFrame) -> pd.DataFrame:
    def field_group(column_name: str) -> str:
        if column_name in MARKET_ANALYSIS_FIELDS:
            return "market_analysis"
        return "metadata"

    def action(row: pd.Series) -> str:
        column_name = row["column"]
        if column_name in CORE_FIELDS:
            return "retain_core_field"
        if row["missing_pct"] > 90:
            return "review_or_drop_high_missing"
        return "retain"

    null_report["field_group"] = null_report["column"].map(field_group)
    null_report["recommended_action"] = null_report.apply(action, axis=1)
    return null_report


def make_missing_value_report(df: pd.DataFrame) -> pd.DataFrame:
    report = pd.DataFrame(
        {
            "column": df.columns,
            "dtype": [str(dtype) for dtype in df.dtypes],
            "non_missing_count": df.notna().sum().values,
            "missing_count": df.isna().sum().values,
        }
    )
    report["missing_pct"] = (report["missing_count"] / len(df) * 100).round(2)
    report["above_90_pct_missing"] = report["missing_pct"] > 90
    report = add_field_classification(report)
    return report.sort_values(["missing_pct", "column"], ascending=[False, True])


def make_numeric_summary(df: pd.DataFrame, fields: list[str]) -> pd.DataFrame:
    rows = []
    for field in fields:
        if field not in df.columns:
            continue

        values = pd.to_numeric(df[field], errors="coerce").dropna()
        rows.append(
            {
                "field": field,
                "count": int(values.count()),
                "missing_count": int(df[field].isna().sum()),
                "min": values.min(),
                "p01": values.quantile(0.01),
                "p05": values.quantile(0.05),
                "p25": values.quantile(0.25),
                "median": values.median(),
                "mean": values.mean(),
                "p75": values.quantile(0.75),
                "p95": values.quantile(0.95),
                "p99": values.quantile(0.99),
                "max": values.max(),
            }
        )
    return pd.DataFrame(rows)


def make_outlier_report(df: pd.DataFrame, fields: list[str]) -> pd.DataFrame:
    rows = []
    for field in fields:
        if field not in df.columns:
            continue

        values = pd.to_numeric(df[field], errors="coerce")
        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = values[(values < lower_bound) | (values > upper_bound)]

        rows.append(
            {
                "field": field,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "outlier_count": int(outliers.count()),
                "outlier_pct": round(outliers.count() / values.notna().sum() * 100, 2)
                if values.notna().sum()
                else 0,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    sold = load_sold_data()

    print(f"\nDataset shape before filtering: {sold.shape[0]:,} rows x {sold.shape[1]:,} columns")
    print("\nColumn data types:")
    print(sold.dtypes)

    property_type_report = (
        sold["PropertyType"]
        .fillna("Missing")
        .value_counts(dropna=False)
        .rename_axis("PropertyType")
        .reset_index(name="count")
    )
    property_type_report["share_pct"] = (
        property_type_report["count"] / len(sold) * 100
    ).round(2)
    property_type_report.to_csv(REPORT_DIR / "property_type_summary.csv", index=False)

    print("\nUnique property types found:")
    print(property_type_report)

    # Filtering logic: keep only records where PropertyType is exactly Residential.
    residential = sold[sold["PropertyType"] == "Residential"].copy()
    print(f"\nRows before Residential filter: {len(sold):,}")
    print(f"Rows after Residential filter: {len(residential):,}")
    print(f"Rows removed by filter: {len(sold) - len(residential):,}")

    null_report = make_missing_value_report(residential)
    high_missing_report = null_report[null_report["above_90_pct_missing"]].copy()
    numeric_summary = make_numeric_summary(residential, NUMERIC_FIELDS)
    required_numeric_summary = make_numeric_summary(
        residential, REQUIRED_NUMERIC_SUMMARY_FIELDS
    )
    outlier_report = make_outlier_report(residential, NUMERIC_FIELDS)

    null_report.to_csv(REPORT_DIR / "null_count_summary.csv", index=False)
    high_missing_report.to_csv(REPORT_DIR / "columns_above_90pct_missing.csv", index=False)
    numeric_summary.to_csv(REPORT_DIR / "numeric_distribution_summary_all_fields.csv", index=False)
    required_numeric_summary.to_csv(
        REPORT_DIR / "numeric_distribution_summary_required_fields.csv",
        index=False,
    )
    outlier_report.to_csv(REPORT_DIR / "numeric_outlier_summary.csv", index=False)

    residential.to_csv(FILTERED_OUTPUT, index=False)

    print("\nHigh-missing columns above 90%:")
    print(high_missing_report[["column", "missing_count", "missing_pct", "recommended_action"]])

    print("\nRequired numeric distribution summary:")
    print(required_numeric_summary)

    print("\nSaved filtered Residential dataset to:", FILTERED_OUTPUT)
    print("Saved validation reports to:", REPORT_DIR)


if __name__ == "__main__":
    main()
