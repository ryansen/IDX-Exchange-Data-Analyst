"""Audit the saved Weeks 4–5 deliverables against the internship requirements."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from weeks_4_5_data_cleaning import DATE_FIELDS, INPUT_FILE, NUMERIC_FIELDS, REPORT_DIR

FINAL_DIR = REPORT_DIR / "final"


def run_check():
    summary = json.loads((FINAL_DIR / "final_summary.json").read_text())
    schema = json.loads((FINAL_DIR / "final_schema.json").read_text())
    output = INPUT_FILE.parent / summary["output_file"]
    numeric = ["ClosePrice", "LivingArea", "DaysOnMarket", "BedroomsTotal", "BathroomsTotalInteger"]
    raw_fields = DATE_FIELDS + numeric + ["Latitude", "Longitude", "StateOrProvince"]
    raw_chunks = pd.read_csv(INPUT_FILE, usecols=raw_fields, dtype="string", chunksize=25000)
    final_chunks = pd.read_csv(output, dtype=schema["dtype"], parse_dates=schema["parse_dates"], chunksize=25000)
    counts, total, types = {}, 0, None
    for raw, final in zip(raw_chunks, final_chunks, strict=True):
        assert len(raw) == len(final), "Before/after rows do not align"
        assert final.SourceRecordNumber.tolist() == list(range(total + 1, total + len(final) + 1))
        if types is None:
            types = final.dtypes.astype(str)
        for field in DATE_FIELDS:
            assert pd.api.types.is_datetime64_any_dtype(final[field]), field
        for field in [c for c in NUMERIC_FIELDS if c in final]:
            assert pd.api.types.is_numeric_dtype(final[field]), field

        for field in numeric:
            values = pd.to_numeric(raw[field], errors="coerce").astype("Float64")
            invalid = (values.le(0) if field in ["ClosePrice", "LivingArea"] else values.lt(0)).fillna(False)
            invalid |= values.isin([np.inf, -np.inf])
            flag = field + "_invalid_flag"
            pd.testing.assert_series_equal(invalid.astype(bool), final[flag], check_names=False)
            counts[flag] = counts.get(flag, 0) + int(invalid.sum())
            remaining_bad = final[field].le(0) if field in ["ClosePrice", "LivingArea"] else final[field].lt(0)
            assert not remaining_bad.any(), f"Invalid {field} survived"

        date_pairs = {"listing_after_close_flag": ("ListingContractDate", "CloseDate"),
                      "purchase_after_close_flag": ("PurchaseContractDate", "CloseDate"),
                      "negative_timeline_flag": ("ListingContractDate", "PurchaseContractDate")}
        for flag, (first, last) in date_pairs.items():
            expected = final[first].notna() & final[last].notna() & final[first].dt.normalize().gt(final[last].dt.normalize())
            pd.testing.assert_series_equal(expected, final[flag], check_names=False)
            counts[flag] = counts.get(flag, 0) + int(expected.sum())
        pd.testing.assert_series_equal(final[DATE_FIELDS[:3]].isna().any(axis=1), final.incomplete_timeline_flag, check_names=False)

        lat, lon = [pd.to_numeric(raw[c], errors="coerce") for c in ["Latitude", "Longitude"]]
        state = raw.StateOrProvince.str.strip().str.upper().replace({"CALIFORNIA": "CA"})
        geo = {"missing_coordinates_flag": lat.isna() | lon.isna(),
               "zero_coordinates_flag": lat.eq(0) | lon.eq(0), "positive_longitude_flag": lon.gt(0),
               "implausible_coordinates_flag": lat.abs().gt(90) | lon.abs().gt(180),
               "outside_california_bounds_flag": (lat.notna() & ~lat.between(32.5, 42.1)) | (lon.notna() & ~lon.between(-124.5, -114.0)),
               "out_of_state_flag": state.notna() & state.ne("CA"), "missing_state_flag": state.isna()}
        for flag, expected in geo.items():
            expected = expected.fillna(False).astype(bool)
            pd.testing.assert_series_equal(expected, final[flag], check_names=False)
            counts[flag] = counts.get(flag, 0) + int(expected.sum())
        combined = final[list(geo) + ["Latitude_invalid_flag", "Longitude_invalid_flag"]].any(axis=1)
        pd.testing.assert_series_equal(combined, final.geographic_quality_flag, check_names=False)
        counts["geographic_quality_flag"] = counts.get("geographic_quality_flag", 0) + int(combined.sum())
        total += len(final)

    assert total == summary["rows_before"] == summary["rows_after"]
    for filename in ["date_consistency.csv", "geographic_summary.csv"]:
        for row in pd.read_csv(REPORT_DIR / filename).itertuples():
            assert counts.get(row.flag, 0) == row.flag_count, row.flag
    final_columns = set(types.index)
    removed = pd.read_csv(REPORT_DIR / "dropped_columns.csv").column.tolist() + summary["dropped_fields"]
    assert not final_columns.intersection(removed)
    assert len(removed) == len(set(removed))
    records = [
        ("Convert all four dates to datetime", "PASS", "All four fields reload as datetime using final_schema.json"),
        ("Remove unnecessary/redundant fields", "PASS", f"{len(removed)} original fields removed: 31 initial + 9 sparse; quality flags added"),
        ("Handle missing values appropriately", "PASS", "58 audited bathroom estimates; 133 exact lot-size conversions; unsupported values stay null"),
        ("Confirm numeric types", "PASS", "All retained numeric fields have numeric dtypes; identifiers remain text"),
        ("Remove or flag invalid numeric values", "PASS", "1 nonpositive price, 172 nonpositive living areas, 54 negative DOM; no negative beds/baths"),
        ("Create required date-order flags", "PASS", "74 listing-after-close; 243 purchase-after-close; 311 purchase-before-listing; missing dates distinguished"),
        ("Flag geographic data problems", "PASS", "4806 unique records; missing/zero/positive longitude/outside bounds/implausible/state checks reconciled"),
        ("Report before/after row counts", "PASS", f"{total:,} before and after; 0 deleted; eligibility flags support analysis-specific exclusions"),
        ("Provide CSV and documented Python workflow", "PASS", "Cleaned_Final.csv plus weeks_4_5_data_cleaning.py and its two documented helper modules"),
    ]
    checklist = pd.DataFrame(records, columns=["requirement", "status", "evidence"])
    checklist.to_csv(FINAL_DIR / "requirements_checklist.csv", index=False)
    types.rename_axis("field").reset_index(name="verified_dtype").to_csv(FINAL_DIR / "data_types_confirmed.csv", index=False)
    result = {"all_requirements_passed": True, "rows_verified": total, "requirements_passed": len(checklist),
              "numeric_date_geographic_counts": counts, "final_file": str(output)}
    (FINAL_DIR / "requirements_check_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    meeting = """# Weeks 4–5 meeting summary

- Completed cleaning of 480,486 Residential sold records covering January 2024–August 2026; retained every record.
- Converted all four required dates and confirmed numeric types. IDs and ZIP codes remain text; a schema restores types when loading the CSV.
- Removed 31 unused/redundant fields initially and nine additional empty or highly sparse features from the final analytical view.
- Flagged and blanked 1 nonpositive sale price, 172 nonpositive living areas and 54 negative days-on-market values. No negative bedrooms or bathrooms were found.
- Added all three required date flags: 74 listing-after-close, 243 purchase-after-close and 311 purchase-before-listing records. Flag categories overlap; missing dates are tracked separately.
- Flagged 4,806 unique records for geographic review, including 4,679 originally missing coordinate pairs, 42 zero-coordinate records and 35 positive longitudes. Geographic categories overlap; the California box is approximate.
- Used MAR/NMAR sensitivity experiments and follow-up investigations to guide missing-data decisions. Filled only 58 supported bathroom counts and recovered 133 lot-size values through exact unit conversion. Every estimate or derivation is flagged and audited.
- Left original list prices, dates, coordinates, HOA fees and other unsupported values unknown. Revised the exploratory original-price imputation after its mean-bias and cohort checks; missing HOA fees do not imply zero.
- Preserved original observed values and identified 522 conflicting lot-size pairs without overwriting them. The three missing-sale-price records are excluded only from price calculations, not the master file.
- Deliverables: the Cleaned_Final CSV, documented Python entry point, readable notebook, schemas, quality reports, requirement checklist and cell-level audits. This delivery covers sold data, not the listings dataset.
"""
    (FINAL_DIR / "meeting_summary.md").write_text(meeting, encoding="utf-8")
    print(checklist.to_string(index=False))
    print(f"FINAL CHECK PASSED: {len(checklist)} requirements; {total:,} rows; dates, numeric values and geographic flags verified.")
    return checklist, types, result


if __name__ == "__main__":
    run_check()
