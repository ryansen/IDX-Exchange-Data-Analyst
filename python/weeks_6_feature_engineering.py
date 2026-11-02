"""Week 6: market metrics, CDE school districts and Tableau segment tables.

Run: python python/weeks_6_feature_engineering.py
Use the project environment after installing requirements.txt. Download school
polygons once with weeks_6_school_districts.py; subsequent runs work offline.

Every input record is retained. Nonpositive/missing denominators produce null
ratios, never infinity or zero substitutes. Known date-order violations produce
null durations. DaysOnMarket remains the original cleaned field. Districts use
the user-specified 2024–25 boundaries, not transaction-date-specific boundaries.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from weeks_6_school_districts import CACHE, SOURCE_PAGE, download_boundaries, load_polygons, match_school_districts

ROOT = Path(__file__).resolve().parent.parent
FINAL_DIR = ROOT / "reports/weeks_4_5/final"
FINAL = ROOT / "csv/CRMLSSold_Combined_Residential_202401_202608_With_MortgageRates_Cleaned_Final.csv"
RAW = ROOT / "csv/CRMLSSold_Combined_Residential_202401_202608_With_MortgageRates.csv"
OUTPUT = ROOT / "csv/CRMLSSold_Residential_202401_202608_Week6_MarketMetrics.csv"
REPORT_DIR = ROOT / "reports/week_6"
PRICE_METRICS = ["PriceRatio", "CloseToOriginalListRatio", "PricePerSqFt"]
TIME_METRICS = ["DaysOnMarket", "ListingToContractDays", "ContractToCloseDays"]
METRICS = PRICE_METRICS + TIME_METRICS
GROUPS = {"property_type": ["PropertyType"], "property_subtype": ["PropertyType", "PropertySubType"],
          "county": ["CountyOrParish"], "mls_area": ["CountyOrParish", "MLSAreaMajor"],
          "listing_office": ["ListOfficeName"], "buyer_office": ["BuyerOfficeName"], "monthly": ["YrMo"]}


def fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def restore_offices(data, raw_path=RAW):
    """Restore requested office dimensions by checked source record identity, not row order alone."""
    keys = ["SourceRecordNumber", "ListingKey", "SourceFile", "SourceMonth"]
    fields = ["ListOfficeName", "BuyerOfficeName"]
    offices = pd.read_csv(raw_path, usecols=keys[1:] + fields, dtype="string", keep_default_na=False)
    offices.insert(0, "SourceRecordNumber", pd.Series(range(1, len(offices) + 1), dtype="Int64"))
    if len(offices) != len(data) or not data.SourceRecordNumber.is_unique:
        raise ValueError("Raw office source and final dataset are not the same record population")
    for field in offices.columns[1:]:
        text = offices[field].str.strip()
        offices[field] = text.mask(text.str.lower().isin(["", "null", "none", "n/a", "na", "nan", "<na>", "-"]))
    restored = data.drop(columns=fields, errors="ignore").merge(offices, how="left", on=keys,
                                                               validate="one_to_one", indicator="office_source_match")
    if not restored.office_source_match.eq("both").all():
        raise ValueError("Office source identity mismatch: ListingKey/SourceFile/SourceMonth changed")
    return restored.drop(columns="office_source_match")


def safe_ratio(numerator, denominator):
    valid = numerator.gt(0) & denominator.gt(0) & np.isfinite(numerator) & np.isfinite(denominator)
    return (numerator / denominator.where(valid.fillna(False))).astype("Float64").replace([np.inf, -np.inf], pd.NA)


def engineer_metrics(data):
    """Explicit equations; preserve source values and exclude unknown/contradictory inputs."""
    data = data.copy()
    for field in ["CloseDate", "ListingContractDate", "PurchaseContractDate"]:
        data[field] = pd.to_datetime(data[field], format="mixed", errors="coerce", utc=True).dt.tz_localize(None)
    data["PriceRatio"] = safe_ratio(data.ClosePrice, data.OriginalListPrice)
    # The handbook defines both names with exactly the same denominator.
    data["CloseToOriginalListRatio"] = data.PriceRatio.copy()
    data["PricePerSqFt"] = safe_ratio(data.ClosePrice, data.LivingArea)
    data["Year"] = data.CloseDate.dt.year.astype("Int64")
    data["Month"] = data.CloseDate.dt.month.astype("Int64")
    data["YrMo"] = data.CloseDate.dt.strftime("%Y-%m").astype("string")
    data["CloseMonth"] = data.CloseDate.dt.to_period("M").dt.to_timestamp()
    listing, contract, close = [data[c].dt.normalize() for c in ["ListingContractDate", "PurchaseContractDate", "CloseDate"]]
    bad_order = listing.gt(contract) | contract.gt(close) | listing.gt(close)
    data["EngineeredTimelineInvalidFlag"] = bad_order.astype(bool)
    # Calendar-day differences; same-day events are zero, not negative fractions.
    data["ListingToContractDays"] = (contract - listing).dt.days.where(~bad_order).astype("Int64")
    data["ContractToCloseDays"] = (close - contract).dt.days.where(~bad_order).astype("Int64")
    data["PriceRatioUnavailableFlag"] = data.PriceRatio.isna()
    data["PricePerSqFtUnavailableFlag"] = data.PricePerSqFt.isna()
    data["OriginalListPriceNonpositiveFlag"] = data.OriginalListPrice.le(0).fillna(False).astype(bool)
    return data


def summarize_segments(data, dimensions):
    """Summaries include valid-value counts; ratios are per-record ratios, not ratios of averages."""
    aggregation = {"Records": ("SourceRecordNumber", "size"), "DistinctListings": ("ListingKey", "nunique"),
                   "ValidClosePriceCount": ("ClosePrice", "count"), "MedianClosePrice": ("ClosePrice", "median"),
                   "MeanClosePrice": ("ClosePrice", "mean")}
    for field in METRICS:
        aggregation.update({"Valid" + field + "Count": (field, "count"),
                            "Median" + field: (field, "median"), "Mean" + field: (field, "mean")})
    table = data.groupby(dimensions, dropna=False, observed=True).agg(**aggregation).reset_index()
    table["RecordSharePct"] = table.Records / len(data) * 100 if len(data) else 0.0
    return table.sort_values("Records", ascending=False).reset_index(drop=True)


def metric_dictionary():
    return pd.DataFrame([
        ("PriceRatio", "ClosePrice / OriginalListPrice", "Positive finite inputs only; null otherwise"),
        ("CloseToOriginalListRatio", "ClosePrice / OriginalListPrice", "Exact alias of PriceRatio, retained as requested"),
        ("PricePerSqFt", "ClosePrice / LivingArea", "USD per square foot; positive finite inputs only"),
        ("DaysOnMarket", "DaysOnMarket", "Unchanged cleaned raw field; not replaced with calendar duration"),
        ("Year", "CloseDate.dt.year", "Nullable integer"), ("Month", "CloseDate.dt.month", "Nullable integer 1–12"),
        ("YrMo", "CloseDate formatted YYYY-MM", "Text label for Tableau; sorts chronologically"),
        ("CloseMonth", "First calendar day of CloseDate month", "Datetime for Tableau monthly axes"),
        ("ListingToContractDays", "PurchaseContractDate - ListingContractDate", "Calendar days; null for missing pair or known contradictory timeline"),
        ("ContractToCloseDays", "CloseDate - PurchaseContractDate", "Calendar days; null for missing pair or known contradictory timeline"),
    ], columns=["field", "formula", "handling_and_purpose"])


def write_report(summary, segments, sample):
    table = segments["county"].head(8)[["CountyOrParish", "Records", "MedianClosePrice", "MedianPricePerSqFt", "MedianDaysOnMarket"]]
    lines = ["# Week 6 — Feature Engineering and Market Metrics", "",
        f"Input: {FINAL.name}. Output: {OUTPUT.name}. All {summary['rows_after']:,} Residential sold records retained.", "",
        "## Engineered metrics", "",
        "PriceRatio and CloseToOriginalListRatio both equal ClosePrice / OriginalListPrice, as specified. "
        "They are aliases, not different negotiation measures. PricePerSqFt = ClosePrice / LivingArea. "
        "DaysOnMarket is preserved. Year, Month, YrMo (YYYY-MM) and CloseMonth come from CloseDate. "
        "ListingToContractDays and ContractToCloseDays use calendar-day differences. Same-day events are zero. "
        "Known contradictory dates null both durations; an incomplete timeline can still yield a valid available date-pair duration.", "",
        "No ratio uses zero/negative/missing denominators. Unknown original prices remain unknown; no fallback to current ListPrice. "
        "No engineered outliers are trimmed or winsorized. Means may be affected by the source data's extreme observations.", "",
        "## Required sample output", "",
        "sample_engineered_metrics.csv contains the first ten records with all requested metrics present; this is an illustrative complete-value sample, "
        "not a representative sample. sample_formula_checks.csv supplies the input values and differences for independent verification.", "",
        "## County segment preview", "", "```", table.round(3).to_string(index=False), "```", "",
        "Segment outputs cover PropertyType, PropertySubType, CountyOrParish, county + MLSAreaMajor, ListOfficeName, BuyerOfficeName and YrMo. "
        "Each includes record count, distinct listing count, valid count/median/mean per metric, and share of all input records. "
        "Unknown segment values are retained as missing-value groups. Record shares are not de-duplicated transaction market shares. "
        "Office names are restored from the raw export by SourceRecordNumber + ListingKey + SourceFile + SourceMonth with one-to-one validation. "
        "Names are source labels, not normalized brokerage/office IDs.", "",
        "## School district enrichment", "",
        f"[User-specified California Department of Education dataset]({SOURCE_PAGE}). The official GeoService supplies the same dataset's "
        "polygons as WGS84 GeoJSON. Boundaries are downloaded locally; no property locations are sent to the service. "
        "Elementary, High and Unified districts have separate names/codes. Elementary/high overlaps are expected and do not duplicate the main dataset. "
        "school_district_membership.csv is a separate record-to-district bridge. Same-type or mixed unified/split-district ambiguity and exact boundary touches are flagged. "
        "No nearest-district assignment is made for unmatched records. Existing geographic quality exclusions remain excluded.", "",
        "The boundary vintage is 2024–25 across the entire January 2024–August 2026 sales period. This is a fixed reference layer, "
        "not proof of district boundaries at every sale date or assignment to a particular school. Source download metadata and checksum are saved locally.", "",
        "## Quality and coverage", "", "```", json.dumps(summary, indent=2), "```", "",
        "Use schema.json to restore numeric, date and boolean types. All outputs are local/confidential. "
        "The Weeks 4–5 final dataset is preserved; this workflow covers sold data only.", ""]
    (REPORT_DIR / "week_6_report.md").write_text("\n".join(lines), encoding="utf-8")


def run():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    schema = json.loads((FINAL_DIR / "final_schema.json").read_text())
    print("Loading final cleaned sold data", flush=True)
    data = pd.read_csv(FINAL, dtype=schema["dtype"], parse_dates=schema["parse_dates"])
    initial_rows = len(data)
    dom_before = data.DaysOnMarket.copy()
    print("Restoring and validating office dimensions", flush=True)
    data = restore_offices(data)
    print("Creating Week 6 metrics", flush=True)
    data = engineer_metrics(data)
    print("Joining school district polygons locally", flush=True)
    boundary_path = download_boundaries()
    geometries, districts, repaired = load_polygons(boundary_path)
    enrichment, membership = match_school_districts(data, geometries, districts)
    data = data.merge(enrichment, on="SourceRecordNumber", how="left", validate="one_to_one")
    assert len(data) == initial_rows and data.SourceRecordNumber.is_unique
    pd.testing.assert_series_equal(dom_before, data.DaysOnMarket)
    pd.testing.assert_series_equal(data.PriceRatio, data.CloseToOriginalListRatio, check_names=False)
    assert not data[PRICE_METRICS].isin([np.inf, -np.inf]).any().any()
    assert not data[["ListingToContractDays", "ContractToCloseDays"]].lt(0).any().any()

    segments = {name: summarize_segments(data, dimensions) for name, dimensions in GROUPS.items()}
    for name, table in segments.items():
        assert int(table.Records.sum()) == initial_rows
        table.to_csv(REPORT_DIR / f"segments_{name}.csv", index=False)
    membership.to_csv(REPORT_DIR / "school_district_membership.csv", index=False)
    complete = data[METRICS + ["Year", "Month", "YrMo"]].notna().all(axis=1)
    sample_fields = ["SourceRecordNumber", "CloseDate", "OriginalListPrice", "ClosePrice", "LivingArea",
                     "PriceRatio", "CloseToOriginalListRatio", "PricePerSqFt", "DaysOnMarket", "Year", "Month", "YrMo",
                     "ListingToContractDays", "ContractToCloseDays"]
    sample = data.loc[complete, sample_fields].head(10)
    sample.to_csv(REPORT_DIR / "sample_engineered_metrics.csv", index=False)
    checks = data.loc[sample.index, ["SourceRecordNumber", "CloseDate", "ListingContractDate", "PurchaseContractDate",
                                    "ClosePrice", "OriginalListPrice", "LivingArea"]].copy()
    checks = checks.join(data.loc[sample.index, PRICE_METRICS + ["ListingToContractDays", "ContractToCloseDays"]])
    checks.to_csv(REPORT_DIR / "sample_formula_checks.csv", index=False)
    metric_dictionary().to_csv(REPORT_DIR / "metric_dictionary.csv", index=False)
    coverage = pd.DataFrame({"metric": METRICS, "valid_count": data[METRICS].notna().sum().values,
                             "missing_count": data[METRICS].isna().sum().values})
    coverage.to_csv(REPORT_DIR / "metric_coverage.csv", index=False)
    unmatched = data.SchoolDistrictMatchStatus.ne("matched")
    data.loc[unmatched, ["SourceRecordNumber", "ListingKey", "Latitude", "Longitude", "SchoolDistrictMatchStatus",
                         "SchoolDistrictMatchCount", "SchoolDistrictNames"]].to_csv(REPORT_DIR / "school_match_review.csv", index=False)
    summary = {"rows_before": initial_rows, "rows_after": len(data), "columns_after": len(data.columns),
               "metric_coverage": coverage.to_dict("records"),
               "district_match_status": data.SchoolDistrictMatchStatus.value_counts().to_dict(),
               "boundary_polygons": len(districts), "polygons_repaired": repaired,
               "district_memberships": len(membership), "boundary_vintage": "2024-25",
               "missing_listing_office": int(data.ListOfficeName.isna().sum()),
               "missing_buyer_office": int(data.BuyerOfficeName.isna().sum()),
               "final_input_sha256": fingerprint(FINAL), "office_source_sha256": fingerprint(RAW)}
    (REPORT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    output_schema = {"parse_dates": [c for c in data if pd.api.types.is_datetime64_any_dtype(data[c])],
                     "dtype": {c: str(t) for c, t in data.dtypes.items() if not pd.api.types.is_datetime64_any_dtype(t)}}
    (REPORT_DIR / "schema.json").write_text(json.dumps(output_schema, indent=2), encoding="utf-8")
    print("Saving Tableau-ready CSV", flush=True)
    data.to_csv(OUTPUT, index=False, date_format="%Y-%m-%dT%H:%M:%S.%f")
    write_report(summary, segments, sample)
    print(sample.round(4).to_string(index=False), flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    print("Saved:", OUTPUT, flush=True)
    return data, segments, summary


if __name__ == "__main__":
    run()
