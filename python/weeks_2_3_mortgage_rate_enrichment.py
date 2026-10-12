"""
Weeks 2-3 deliverable: mortgage rate enrichment.

This script fetches the FRED MORTGAGE30US weekly 30-year fixed mortgage
rate series, averages it to month level, and merges it onto the combined
Residential sold and listing datasets.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


FRED_MORTGAGE_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=MORTGAGE30US"

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "csv"

SOLD_INPUT = DATA_DIR / "CRMLSSold_Combined_Residential_202401_202608.csv"
LISTINGS_INPUT = DATA_DIR / "CRMLSListing_Combined_Residential_202401_202608.csv"

SOLD_OUTPUT = DATA_DIR / "CRMLSSold_Combined_Residential_202401_202608_With_MortgageRates.csv"
LISTINGS_OUTPUT = (
    DATA_DIR / "CRMLSListing_Combined_Residential_202401_202608_With_MortgageRates.csv"
)


def fetch_monthly_mortgage_rates() -> pd.DataFrame:
    mortgage = pd.read_csv(FRED_MORTGAGE_URL, parse_dates=["observation_date"])
    mortgage.columns = ["date", "rate_30yr_fixed"]
    mortgage["rate_30yr_fixed"] = pd.to_numeric(
        mortgage["rate_30yr_fixed"], errors="coerce"
    )
    mortgage = mortgage.dropna(subset=["date", "rate_30yr_fixed"])
    mortgage["year_month"] = mortgage["date"].dt.to_period("M")

    return (
        mortgage.groupby("year_month", as_index=False)["rate_30yr_fixed"]
        .mean()
        .sort_values("year_month")
    )


def load_dataset(path: Path, required_date_column: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing input CSV: {path}")

    df = pd.read_csv(
        path,
        encoding="cp1252",
        encoding_errors="replace",
        low_memory=False,
    )
    if required_date_column not in df.columns:
        raise KeyError(f"{path.name} is missing {required_date_column}")

    df["year_month"] = pd.to_datetime(
        df[required_date_column], errors="coerce"
    ).dt.to_period("M")
    missing_dates = int(df["year_month"].isna().sum())
    if missing_dates:
        raise ValueError(
            f"{path.name} has {missing_dates:,} rows with invalid {required_date_column}"
        )

    return df


def merge_and_validate(
    df: pd.DataFrame,
    mortgage_monthly: pd.DataFrame,
    dataset_name: str,
) -> pd.DataFrame:
    enriched = df.merge(mortgage_monthly, on="year_month", how="left")
    missing_rates = int(enriched["rate_30yr_fixed"].isna().sum())

    if missing_rates:
        missing_months = (
            enriched.loc[enriched["rate_30yr_fixed"].isna(), "year_month"]
            .dropna()
            .drop_duplicates()
            .sort_values()
            .astype(str)
            .tolist()
        )
        raise ValueError(
            f"{dataset_name} has {missing_rates:,} rows without mortgage rates. "
            f"Missing months: {', '.join(missing_months) or 'invalid date keys'}"
        )

    print(f"{dataset_name} null rate_30yr_fixed values: {missing_rates}")
    return enriched


def main() -> None:
    print("Fetching FRED MORTGAGE30US weekly mortgage rates.")
    mortgage_monthly = fetch_monthly_mortgage_rates()
    print(
        "Monthly mortgage rates available from "
        f"{mortgage_monthly['year_month'].min()} through "
        f"{mortgage_monthly['year_month'].max()}."
    )

    sold = load_dataset(SOLD_INPUT, "CloseDate")
    listings = load_dataset(LISTINGS_INPUT, "ListingContractDate")

    sold_with_rates = merge_and_validate(sold, mortgage_monthly, "Sold")
    listings_with_rates = merge_and_validate(listings, mortgage_monthly, "Listings")

    sold_with_rates.to_csv(SOLD_OUTPUT, index=False)
    listings_with_rates.to_csv(LISTINGS_OUTPUT, index=False)

    print("\nSold preview:")
    print(
        sold_with_rates[
            ["CloseDate", "year_month", "ClosePrice", "rate_30yr_fixed"]
        ].head()
    )

    print("\nListings preview:")
    print(
        listings_with_rates[
            ["ListingContractDate", "year_month", "ListPrice", "rate_30yr_fixed"]
        ].head()
    )

    print("\nSaved enriched sold dataset to:", SOLD_OUTPUT)
    print("Saved enriched listings dataset to:", LISTINGS_OUTPUT)


if __name__ == "__main__":
    main()
