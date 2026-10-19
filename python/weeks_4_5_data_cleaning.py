# %% Setup
"""Weeks 4–5: clean the sold data, flag problems and save quality reports.

All rows are kept. Invalid measurements become blank instead of being guessed.
The matching notebook presents these rules step by step, with previews and charts.
"""

import json
from pathlib import Path

import pandas as pd

# Works from the script, the project folder or a notebook in python/.
PROJECT_DIR = Path(__file__).resolve().parent.parent if "__file__" in globals() else Path.cwd()
if not (PROJECT_DIR / "csv").is_dir():
    PROJECT_DIR = PROJECT_DIR.parent

# Change these paths if you want to use a different file.
INPUT_FILE = PROJECT_DIR / "csv/CRMLSSold_Combined_Residential_202401_202608_With_MortgageRates.csv"
OUTPUT_FILE = INPUT_FILE.with_name(INPUT_FILE.stem + "_Cleaned.csv")
REPORT_DIR = PROJECT_DIR / "reports/weeks_4_5"

DATE_FIELDS = ["CloseDate", "PurchaseContractDate", "ListingContractDate", "ContractStatusChangeDate"]
NUMERIC_FIELDS = [
    "ClosePrice", "ListPrice", "OriginalListPrice", "LivingArea", "DaysOnMarket",
    "BedroomsTotal", "BathroomsTotalInteger", "BathroomsFull", "BathroomsHalf",
    "LotSizeAcres", "LotSizeSquareFeet", "LotSizeArea", "YearBuilt", "BuildingAreaTotal",
    "AboveGradeFinishedArea", "BelowGradeFinishedArea", "FireplacesTotal", "ParkingTotal",
    "GarageSpaces", "CoveredSpaces", "Stories", "MainLevelBedrooms", "AssociationFee",
    "TaxAnnualAmount", "TaxYear", "Latitude", "Longitude", "rate_30yr_fixed",
]
INTEGER_FIELDS = [
    "DaysOnMarket", "BedroomsTotal", "BathroomsTotalInteger", "BathroomsFull",
    "BathroomsHalf", "YearBuilt", "FireplacesTotal", "MainLevelBedrooms", "TaxYear",
]
TEXT_FIELDS = [
    "SourceMonth", "SourceFile", "ListingKey", "ListingId", "PropertyType", "PropertySubType",
    "MlsStatus", "CountyOrParish", "City", "StateOrProvince", "PostalCode", "MLSAreaMajor",
    "SubdivisionName", "UnparsedAddress", "OriginatingSystemName", "OriginatingSystemSubName",
    "year_month", "AssociationFeeFrequency",
]
BOOLEAN_FIELDS = [
    "latfilled", "lonfilled", "ViewYN", "WaterfrontYN", "BasementYN", "PoolPrivateYN",
    "AttachedGarageYN", "FireplaceYN", "NewConstructionYN",
]


# %% Clean fields
def clean_data(raw):
    """Return the cleaned data and the tables used to review data quality."""
    required = DATE_FIELDS[:3] + ["ClosePrice", "LivingArea", "DaysOnMarket", "BedroomsTotal",
                                 "BathroomsTotalInteger", "Latitude", "Longitude", "StateOrProvince"]
    missing = set(required) - set(raw.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    # 1. Keep useful analysis fields. Drop agent, office and other unused metadata.
    keep = TEXT_FIELDS + DATE_FIELDS + NUMERIC_FIELDS + BOOLEAN_FIELDS
    sold = raw[[column for column in raw if column in keep]].copy().reset_index(drop=True)
    dropped = pd.DataFrame({"column": [column for column in raw if column not in keep]})
    dropped["reason"] = "Not needed for the selected market analysis"

    # 2. Strip whitespace and leave missing values blank. Keep IDs and ZIP codes as text.
    for column in sold:
        text = sold[column].astype("string").str.strip()
        sold[column] = text.mask(text.str.lower().isin(["", "null", "none", "n/a", "na", "nan", "<na>", "-"]))
    for column in DATE_FIELDS:
        if column not in sold:
            sold[column] = pd.Series(pd.NA, index=sold.index, dtype="string")
    original = sold.copy()
    sold.insert(0, "SourceRecordNumber", pd.Series(range(1, len(sold) + 1), dtype="Int64"))
    conversions, invalid_values = [], []

    # 3. Convert dates, numbers and booleans. Save invalid original values for review.
    for column in DATE_FIELDS + NUMERIC_FIELDS + BOOLEAN_FIELDS:
        if column not in sold:
            continue
        text = original[column]
        invalid = pd.Series(False, index=sold.index)
        if column in DATE_FIELDS:
            # Normalize timezones to UTC. Bad dates become NaT (a missing date).
            values = pd.to_datetime(text, format="mixed", errors="coerce", utc=True).dt.tz_localize(None)
            flag = column + "_parse_error_flag"
        elif column in NUMERIC_FIELDS:
            # Accept $1,234.50, but reject malformed grouping such as 12,34.
            numbers = text.str.replace(r"^\$\s*", "", regex=True)
            grouped = numbers.str.fullmatch(r"[+-]?\d{1,3}(,\d{3})+(\.\d+)?").fillna(False)
            numbers = numbers.mask(grouped, numbers.str.replace(",", "", regex=False))
            values = pd.to_numeric(numbers, errors="coerce").astype("Float64")
            invalid = values.isin([float("inf"), float("-inf")])
            if column in ["ClosePrice", "LivingArea"]:
                invalid |= values.le(0).fillna(False)
            if column == "DaysOnMarket" or column.startswith(("Bedrooms", "Bathrooms")):
                invalid |= values.lt(0).fillna(False)
            if column in INTEGER_FIELDS:
                invalid |= (values.mod(1).ne(0) | values.le(-(2**63)) | values.ge(2**63)).fillna(False)
            flag = column + "_invalid_flag"
        else:
            values = text.str.lower().map({"true": True, "false": False, "yes": True, "no": False,
                                          "1": True, "0": False, "1.0": True, "0.0": False}).astype("boolean")
            flag = column + "_parse_error_flag"

        parse_error = text.notna() & values.isna()
        sold[flag] = (parse_error | invalid).astype(bool)
        sold[column] = values.mask(invalid)
        if column in INTEGER_FIELDS:
            sold[column] = sold[column].astype("Int64")
        conversions.append({"field": column, "parse_error_count": int(parse_error.sum()),
                            "invalid_value_count": int(invalid.sum()), "cleaned_dtype": str(sold[column].dtype)})
        bad = sold[flag]
        invalid_values.append(pd.DataFrame({"SourceRecordNumber": sold.loc[bad, "SourceRecordNumber"],
                                           "field": column, "original_value": text[bad]}))

    sold["StateOrProvince"] = sold["StateOrProvince"].str.upper().replace({"CALIFORNIA": "CA"})

    # 4. Check date order without changing dates. Same-calendar-day events are allowed.
    date_pairs = {
        "listing_after_close_flag": ("ListingContractDate", "CloseDate"),
        "purchase_after_close_flag": ("PurchaseContractDate", "CloseDate"),
        "negative_timeline_flag": ("ListingContractDate", "PurchaseContractDate"),
    }
    date_report = []
    for flag, (first, second) in date_pairs.items():
        checked = sold[first].notna() & sold[second].notna()
        sold[flag] = checked & (sold[first].dt.normalize() > sold[second].dt.normalize())
        date_report.append({"flag": flag, "records_checked": int(checked.sum()),
                            "records_not_checked": int((~checked).sum()), "flag_count": int(sold[flag].sum())})
    sold["incomplete_timeline_flag"] = sold[DATE_FIELDS[:3]].isna().any(axis=1)

    # 5. Screen coordinates. The California box is approximate, not an exact state boundary.
    lat, lon = sold["Latitude"], sold["Longitude"]
    geo_checks = {
        "missing_coordinates_flag": lat.isna() | lon.isna(),
        "zero_coordinates_flag": lat.eq(0) | lon.eq(0),
        "positive_longitude_flag": lon.gt(0),
        "implausible_coordinates_flag": lat.abs().gt(90) | lon.abs().gt(180),
        "outside_california_bounds_flag": (lat.notna() & ~lat.between(32.5, 42.1)) | (lon.notna() & ~lon.between(-124.5, -114.0)),
        "out_of_state_flag": sold["StateOrProvince"].notna() & sold["StateOrProvince"].ne("CA"),
        "missing_state_flag": sold["StateOrProvince"].isna(),
    }
    for flag, check in geo_checks.items():
        sold[flag] = check.fillna(False).astype(bool)
    geo_flags = list(geo_checks) + ["Latitude_invalid_flag", "Longitude_invalid_flag"]
    sold["geographic_quality_flag"] = sold[geo_flags].any(axis=1)
    geographic_records = sold.loc[sold.geographic_quality_flag, ["SourceRecordNumber"] + geo_flags].join(
        original[["Latitude", "Longitude", "StateOrProvince"]].add_prefix("original_"))
    # Blank unusable coordinate pairs. A missing state alone does not erase a usable pair.
    unusable = sold[[flag for flag in geo_flags if flag != "missing_state_flag"]].any(axis=1)
    sold.loc[unusable, ["Latitude", "Longitude"]] = pd.NA
    sold["any_quality_issue_flag"] = sold.filter(like="_flag").any(axis=1)

    # 6. Confirm types and missingness after cleaning. Preserve every input row.
    fields = original.columns
    field_report = pd.DataFrame({"field": fields, "cleaned_dtype": sold[fields].dtypes.astype(str).values,
                                 "missing_before_cleaning": original.isna().sum().values,
                                 "missing_after_cleaning": sold[fields].isna().sum().values})
    geographic_summary = sold[geo_flags + ["geographic_quality_flag"]].sum().rename_axis("flag").reset_index(name="flag_count")
    geographic_summary["pct_of_all_rows"] = (geographic_summary.flag_count / max(len(sold), 1) * 100).round(4)
    reports = {"field_summary": field_report, "dropped_columns": dropped,
               "conversions": pd.DataFrame(conversions), "invalid_values": pd.concat(invalid_values, ignore_index=True),
               "date_consistency": pd.DataFrame(date_report), "geographic_summary": geographic_summary,
               "geographic_invalid_records": geographic_records}
    return sold, reports


# %% Save the dataset and reports
def run(input_path=INPUT_FILE, output_path=OUTPUT_FILE, report_dir=REPORT_DIR, encoding="utf-8"):
    """Load the raw CSV, clean it and save the deliverables locally."""
    input_path, output_path, report_dir = [Path(path).resolve() for path in (input_path, output_path, report_dir)]
    report_names = ["field_summary", "dropped_columns", "conversions", "invalid_values",
                    "date_consistency", "geographic_summary", "geographic_invalid_records"]
    targets = [output_path] + [report_dir / (name + ".csv") for name in report_names]
    targets += [report_dir / name for name in ["schema.json", "summary.json", "summary.md"]]
    if input_path in targets or len(set(targets)) != len(targets):
        raise ValueError("Input, output and report files must use distinct paths")
    print("Loading:", input_path.name, flush=True)
    raw = pd.read_csv(input_path, dtype="string", encoding=encoding, keep_default_na=False)
    sold, reports = clean_data(raw)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    sold.to_csv(output_path, index=False, date_format="%Y-%m-%dT%H:%M:%S.%f")
    for name, report in reports.items():
        report.to_csv(report_dir / (name + ".csv"), index=False)

    # CSV cannot store types. Use this schema's dtype and parse_dates when reloading.
    schema = {"parse_dates": DATE_FIELDS,
              "dtype": {column: str(dtype) for column, dtype in sold.dtypes.items() if column not in DATE_FIELDS}}
    (report_dir / "schema.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")
    summary = {"rows_before": len(raw), "rows_after": len(sold), "rows_removed": len(raw) - len(sold),
               "columns_removed": len(reports["dropped_columns"]),
               "rows_with_quality_issues": int(sold.any_quality_issue_flag.sum()),
               "date_consistency": reports["date_consistency"].to_dict("records"),
               "geographic_quality": reports["geographic_summary"].to_dict("records")}
    (report_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    notes = f"""# Weeks 4–5 Cleaning Summary

Rows before: {len(raw):,}. Rows after: {len(sold):,}. Rows removed: {len(raw) - len(sold):,}.
Unused source columns removed: {len(reports['dropped_columns'])}.

Missing values stay blank. Invalid numbers become blank with flags. Dates are parsed as UTC.
All transformations and their reasons are documented in the script and matching notebook.
negative_timeline_flag means purchase before listing. Same-day events are allowed.
Incomplete timelines: {int(sold.incomplete_timeline_flag.sum()):,}. False flags with missing dates are not validated timelines.

Date consistency:
{reports['date_consistency'].to_string(index=False)}

Geographic quality (overlapping categories; use the combined flag for unique records):
{reports['geographic_summary'].to_string(index=False)}

The California screening box is latitude 32.5–42.1 and longitude -124.5–-114.0, inclusive.
It is approximate. Points inside it can still be out of state. Non-CA state labels are also flagged.
Unusable coordinate pairs become blank; original coordinates are saved in geographic_invalid_records.csv.
See field_summary.csv for type confirmations and missing counts, and conversions.csv for invalid numeric counts.
Use schema.json to restore dates, numeric types, booleans and text identifiers when loading the CSV.
"""
    (report_dir / "summary.md").write_text(notes, encoding="utf-8")
    print(notes)
    print("Cleaned CSV:", output_path)
    return sold, reports


# %% Run the cleaning
if __name__ == "__main__":
    sold, reports = run()
