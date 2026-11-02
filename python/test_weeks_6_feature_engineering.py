"""Meaningful metric, provenance and spatial-join checks for Week 6."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from shapely.geometry import box

from weeks_6_feature_engineering import engineer_metrics, restore_offices, safe_ratio, summarize_segments
from weeks_6_school_districts import match_school_districts

ROOT = Path(__file__).resolve().parent.parent


def records():
    return pd.DataFrame({"SourceRecordNumber": pd.Series([1, 2, 3, 4], dtype="Int64"),
        "ListingKey": pd.Series(["001", "002", "003", "004"], dtype="string"),
        "SourceFile": pd.Series(["sold.csv"] * 4, dtype="string"),
        "SourceMonth": pd.Series(["202401"] * 4, dtype="string"),
        "CountyOrParish": ["A", "A", None, "B"], "PropertyType": ["Residential"] * 4,
        "ClosePrice": pd.Series([900000, 1200000, pd.NA, 500000], dtype="Float64"),
        "OriginalListPrice": pd.Series([1000000, 1000000, 0, pd.NA], dtype="Float64"),
        "LivingArea": pd.Series([1800, 2000, 0, pd.NA], dtype="Float64"),
        "DaysOnMarket": pd.Series([12, 0, pd.NA, 33], dtype="Int64"),
        "CloseDate": ["2024-01-31", "2024-02-29", "bad", "2024-03-01"],
        "ListingContractDate": ["2024-01-01", "2024-02-01", "2024-02-01", "2024-03-02"],
        "PurchaseContractDate": ["2024-01-11", "2024-02-11", None, "2024-03-03"]})


class MetricTests(unittest.TestCase):
    def test_required_equations_and_alias(self):
        result = engineer_metrics(records())
        self.assertAlmostEqual(result.loc[0, "PriceRatio"], 0.9)
        self.assertAlmostEqual(result.loc[0, "PricePerSqFt"], 500)
        self.assertEqual(result.loc[0, "ListingToContractDays"], 10)
        self.assertEqual(result.loc[0, "ContractToCloseDays"], 20)
        self.assertEqual(result.loc[1, "ContractToCloseDays"], 18)  # Leap year.
        pd.testing.assert_series_equal(result.PriceRatio, result.CloseToOriginalListRatio, check_names=False)
        pd.testing.assert_series_equal(result.DaysOnMarket, records().DaysOnMarket)

    def test_zero_negative_missing_and_infinite_inputs_stay_unavailable(self):
        numerator = pd.Series([100, 100, 100, -1, np.inf, pd.NA], dtype="Float64")
        denominator = pd.Series([0, -5, pd.NA, 2, 2, 2], dtype="Float64")
        self.assertTrue(safe_ratio(numerator, denominator).isna().all())
        result = engineer_metrics(records())
        self.assertTrue(result.loc[2:, ["PriceRatio", "PricePerSqFt"]].isna().all().all())

    def test_months_invalid_dates_and_contradictory_timelines(self):
        result = engineer_metrics(records())
        self.assertEqual(result.loc[1, "Year"], 2024)
        self.assertEqual(result.loc[1, "Month"], 2)
        self.assertEqual(result.loc[1, "YrMo"], "2024-02")
        self.assertEqual(result.loc[1, "CloseMonth"], pd.Timestamp("2024-02-01"))
        self.assertTrue(pd.isna(result.loc[2, "YrMo"]))
        self.assertTrue(result.loc[3, "EngineeredTimelineInvalidFlag"])
        self.assertTrue(result.loc[3, ["ListingToContractDays", "ContractToCloseDays"]].isna().all())

    def test_same_day_and_available_partial_timeline(self):
        data = records().iloc[:2].copy()
        data.loc[0, "ListingContractDate"] = "2024-01-31T20:00:00Z"
        data.loc[0, "PurchaseContractDate"] = "2024-01-31T23:00:00Z"
        data.loc[1, "ListingContractDate"] = None
        result = engineer_metrics(data)
        self.assertEqual(result.loc[0, "ListingToContractDays"], 0)
        self.assertEqual(result.loc[0, "ContractToCloseDays"], 0)
        self.assertTrue(pd.isna(result.loc[1, "ListingToContractDays"]))
        self.assertEqual(result.loc[1, "ContractToCloseDays"], 18)

    def test_segments_preserve_unknown_groups_and_valid_counts(self):
        data = engineer_metrics(records())
        summary = summarize_segments(data, ["CountyOrParish"])
        self.assertEqual(summary.Records.sum(), 4)
        self.assertAlmostEqual(summary.RecordSharePct.sum(), 100)
        unknown = summary[summary.CountyOrParish.isna()].iloc[0]
        self.assertEqual(unknown.ValidPriceRatioCount, 0)
        self.assertTrue(pd.isna(unknown.MedianClosePrice))
        group = summary[summary.CountyOrParish.eq("A")].iloc[0]
        self.assertAlmostEqual(group.MedianPriceRatio, 1.05)
        self.assertAlmostEqual(group.MedianPricePerSqFt, 550)

    def test_office_restoration_validates_full_identity(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "reports") as directory:
            path = Path(directory) / "source.csv"
            source = records()[["ListingKey", "SourceFile", "SourceMonth"]].copy()
            source["ListOfficeName"] = [" Office A ", "Office B", "", "Office D"]
            source["BuyerOfficeName"] = ["Buyer A"] * 4
            source.to_csv(path, index=False)
            result = restore_offices(records(), path)
            self.assertEqual(result.loc[0, "ListOfficeName"], "Office A")
            self.assertTrue(pd.isna(result.loc[2, "ListOfficeName"]))
            self.assertEqual(len(result), 4)
            changed = records()
            changed.loc[0, "ListingKey"] = "different"
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                restore_offices(changed, path)


class DistrictTests(unittest.TestCase):
    def polygons(self):
        polygons = [box(0, 0, 1, 1), box(0, 0, 1, 1), box(2, 0, 3, 1)]
        table = pd.DataFrame({"DistrictCode": ["0000001", "0000002", "0000003"],
                              "DistrictName": ["Elementary A", "High A", "Unified B"],
                              "DistrictType": ["Elementary", "High", "Unified"], "DistrictDataYear": ["2024-25"] * 3})
        return polygons, table

    def test_expected_overlaps_preserve_one_record(self):
        polygons, table = self.polygons()
        data = pd.DataFrame({"SourceRecordNumber": [1, 2, 3, 4, 5], "Latitude": [.5, .5, .5, np.nan, .5],
                             "Longitude": [.5, 2.5, 1.5, .5, .5], "map_analysis_eligible_flag": [True, True, True, False, False]})
        result, bridge = match_school_districts(data, polygons, table)
        self.assertEqual(len(result), 5)
        self.assertEqual(result.loc[0, "SchoolDistrictMatchCount"], 2)
        self.assertFalse(result.loc[0, "SchoolDistrictAmbiguousFlag"])
        self.assertEqual(result.loc[0, "ElementarySchoolDistrictCode"], "0000001")
        self.assertEqual(result.loc[0, "HighSchoolDistrict"], "High A")
        self.assertEqual(result.loc[1, "UnifiedSchoolDistrict"], "Unified B")
        self.assertEqual(result.loc[2, "SchoolDistrictMatchStatus"], "unmatched")
        self.assertTrue(result.loc[3:, "SchoolDistrictMatchStatus"].eq("excluded_geographic_quality").all())
        self.assertEqual(len(bridge), 3)

    def test_boundary_and_same_type_ambiguity(self):
        polygons, table = self.polygons()
        polygons.append(box(1, 0, 2, 1))
        table = pd.concat([table, pd.DataFrame([{"DistrictCode": "0000004", "DistrictName": "Elementary C",
                                                "DistrictType": "Elementary", "DistrictDataYear": "2024-25"}])], ignore_index=True)
        data = pd.DataFrame({"SourceRecordNumber": [1], "Latitude": [.5], "Longitude": [1.0]})
        result, bridge = match_school_districts(data, polygons, table)
        self.assertEqual(result.loc[0, "SchoolDistrictMatchStatus"], "ambiguous")
        self.assertTrue(result.loc[0, "SchoolDistrictBoundaryFlag"])
        self.assertTrue(pd.isna(result.loc[0, "ElementarySchoolDistrict"]))
        self.assertEqual(len(bridge), 3)

    def test_no_usable_coordinates(self):
        polygons, table = self.polygons()
        data = pd.DataFrame({"SourceRecordNumber": [1], "Latitude": [np.nan], "Longitude": [np.nan]})
        result, bridge = match_school_districts(data, polygons, table)
        self.assertEqual(result.loc[0, "SchoolDistrictMatchCount"], 0)
        self.assertEqual(result.loc[0, "SchoolDistrictMatchStatus"], "excluded_geographic_quality")
        self.assertTrue(bridge.empty)


if __name__ == "__main__":
    unittest.main()
