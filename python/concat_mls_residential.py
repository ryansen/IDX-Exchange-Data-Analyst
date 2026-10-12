"""
Week 1 deliverable: combine monthly CRMLS files.

This script loads monthly MLS files from January 2024 through the most recently
completed calendar month, concatenates them into listing and sold datasets,
filters both to Residential properties, and saves two new CSV files.
"""

from datetime import date, timedelta
from pathlib import Path

import pandas as pd


START_MONTH = "202401"
DATA_DIR = Path(__file__).resolve().parent.parent / "csv"
OUTPUT_DIR = DATA_DIR


def get_last_completed_month() -> str:
    first_day_this_month = date.today().replace(day=1)
    last_day_last_month = first_day_this_month - timedelta(days=1)
    return last_day_last_month.strftime("%Y%m")


def get_months(start_month: str, end_month: str) -> list[str]:
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


def read_monthly_files(prefix: str, months: list[str]) -> tuple[pd.DataFrame, list[str]]:
    monthly_data = []
    missing_months = []

    print(f"\n{prefix} files")
    print("-" * (len(prefix) + 6))

    for month in months:
        regular_file = DATA_DIR / f"{prefix}{month}.csv"
        filled_file = DATA_DIR / f"{prefix}{month}_filled.csv"
        file_path = regular_file if regular_file.exists() else filled_file

        if not file_path.exists():
            missing_months.append(month)
            continue

        df = pd.read_csv(
            file_path,
            encoding="cp1252",
            encoding_errors="replace",
            low_memory=False,
        )
        df.insert(0, "SourceMonth", month)
        df.insert(1, "SourceFile", file_path.name)
        monthly_data.append(df)

        # Row count before concatenation for this monthly source file.
        print(f"{month}: {len(df):,} rows before concat from {file_path.name}")

    if not monthly_data:
        raise FileNotFoundError(f"No files found for {prefix}")

    combined = pd.concat(monthly_data, ignore_index=True, sort=False)

    # Row count after concatenation, before Residential filtering.
    print(f"Rows after concat: {len(combined):,}")

    return combined, missing_months


def keep_residential_only(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    if "PropertyType" not in df.columns:
        raise KeyError(f"{dataset_name} data is missing the PropertyType column")

    # Row counts before and after the PropertyType == Residential filter.
    print(f"{dataset_name} rows before Residential filter: {len(df):,}")
    residential = df[df["PropertyType"] == "Residential"].copy()
    print(f"{dataset_name} rows after Residential filter: {len(residential):,}")
    return residential


def main() -> None:
    end_month = get_last_completed_month()
    months = get_months(START_MONTH, end_month)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Combining monthly files from {START_MONTH} through {end_month}.")
    print(f"Source folder: {DATA_DIR}")

    listings, missing_listing_months = read_monthly_files("CRMLSListing", months)
    sold, missing_sold_months = read_monthly_files("CRMLSSold", months)

    residential_listings = keep_residential_only(listings, "Listing")
    residential_sold = keep_residential_only(sold, "Sold")

    listing_output = OUTPUT_DIR / (
        f"CRMLSListing_Combined_Residential_{START_MONTH}_{end_month}.csv"
    )
    sold_output = OUTPUT_DIR / (
        f"CRMLSSold_Combined_Residential_{START_MONTH}_{end_month}.csv"
    )

    residential_listings.to_csv(listing_output, index=False)
    residential_sold.to_csv(sold_output, index=False)

    if missing_listing_months:
        print("Missing listing months skipped:", ", ".join(missing_listing_months))
    if missing_sold_months:
        print("Missing sold months skipped:", ", ".join(missing_sold_months))

    print("\nSaved combined Residential listing data to:", listing_output)
    print("Saved combined Residential sold data to:", sold_output)


if __name__ == "__main__":
    main()
