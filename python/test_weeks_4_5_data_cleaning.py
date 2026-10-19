"""Targeted checks for cleaning rules and CSV round-trip behavior."""

import json
import contextlib
import io
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from weeks_4_5_data_cleaning import PROJECT_DIR, clean_data, run


def source(**changes):
    row = {
        "ListingKey": "000123", "PostalCode": "01234", "CloseDate": "2024-03-01",
        "ListingContractDate": "2024-01-01", "PurchaseContractDate": "2024-02-01",
        "ClosePrice": "$1,234.50", "LivingArea": "1,200", "DaysOnMarket": "0",
        "BedroomsTotal": "0", "BathroomsTotalInteger": "0", "Latitude": "34",
        "Longitude": "-118", "StateOrProvince": " California ", "latfilled": "true",
    }
    row.update(changes)
    return row


class CleaningTests(unittest.TestCase):
    def test_valid_types_and_same_day(self):
        raw = pd.DataFrame([source(PurchaseContractDate="2024-01-01T01:00:00Z",
                                   ListingContractDate="2024-01-01T20:00:00Z")])
        cleaned, _ = clean_data(raw)
        self.assertEqual(cleaned.loc[0, "ClosePrice"], 1234.5)
        self.assertEqual(cleaned.loc[0, "LivingArea"], 1200)
        self.assertEqual(cleaned.loc[0, "PostalCode"], "01234")
        self.assertEqual(cleaned.loc[0, "StateOrProvince"], "CA")
        self.assertEqual(str(cleaned.DaysOnMarket.dtype), "Int64")
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(cleaned.CloseDate))
        self.assertFalse(cleaned.loc[0, "any_quality_issue_flag"])

    def test_numeric_errors_preserve_rows_and_original_tokens(self):
        raw = pd.DataFrame([source(ClosePrice="0", LivingArea="-1", DaysOnMarket="-2",
                                   BedroomsTotal="-1", BathroomsTotalInteger="-3"),
                            source(ClosePrice="12,34", LivingArea="inf", DaysOnMarket="1.5"),
                            source(DaysOnMarket="1e30")])
        cleaned, tables = clean_data(raw)
        self.assertEqual(len(cleaned), 3)
        self.assertTrue(cleaned.loc[0, ["ClosePrice", "LivingArea", "DaysOnMarket",
                                       "BedroomsTotal", "BathroomsTotalInteger"]].isna().all())
        self.assertTrue(cleaned.loc[1, ["ClosePrice", "LivingArea", "DaysOnMarket"]].isna().all())
        self.assertTrue(pd.isna(cleaned.loc[2, "DaysOnMarket"]))
        self.assertIn("12,34", tables["invalid_values"].original_value.tolist())

    def test_date_rules_and_missing_dates(self):
        raw = pd.DataFrame([source(ListingContractDate="2024-04-01"),
                            source(PurchaseContractDate="2024-04-01"),
                            source(CloseDate="bad", PurchaseContractDate=" N/A ")])
        cleaned, tables = clean_data(raw)
        self.assertTrue(cleaned.loc[0, "listing_after_close_flag"])
        self.assertTrue(cleaned.loc[0, "negative_timeline_flag"])
        self.assertTrue(cleaned.loc[1, "purchase_after_close_flag"])
        self.assertTrue(cleaned.loc[2, "incomplete_timeline_flag"])
        self.assertTrue(cleaned.loc[2, "CloseDate_parse_error_flag"])
        self.assertEqual(tables["date_consistency"].iloc[0].records_not_checked, 1)

    def test_geography_checks_independent_components(self):
        raw = pd.DataFrame([source(Latitude="", Longitude="118"),
                            source(Latitude="0"), source(Latitude="91"),
                            source(StateOrProvince="NV"), source(Longitude="-113"),
                            source(Latitude="bad"),
                            source(Latitude="32.5", Longitude="-124.5"),
                            source(StateOrProvince="")])
        cleaned, tables = clean_data(raw)
        self.assertTrue(cleaned.loc[0, "missing_coordinates_flag"])
        self.assertTrue(cleaned.loc[0, "positive_longitude_flag"])
        self.assertTrue(cleaned.loc[1, "zero_coordinates_flag"])
        self.assertTrue(cleaned.loc[2, "implausible_coordinates_flag"])
        self.assertTrue(cleaned.loc[3, "out_of_state_flag"])
        self.assertTrue(cleaned.loc[4, "outside_california_bounds_flag"])
        self.assertTrue(cleaned.loc[5, "Latitude_invalid_flag"])
        self.assertTrue(cleaned.loc[:5, ["Latitude", "Longitude"]].isna().all().all())
        self.assertFalse(cleaned.loc[6, "geographic_quality_flag"])
        self.assertTrue(cleaned.loc[7, "missing_state_flag"])
        self.assertEqual(cleaned.loc[7, "Latitude"], 34)
        self.assertEqual(len(tables["geographic_invalid_records"]), 7)

    def test_csv_round_trip_and_protect_source(self):
        report_folder = PROJECT_DIR / "reports"
        report_folder.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=report_folder) as directory:
            folder = Path(directory)
            original = folder / "source.csv"
            pd.DataFrame([source(), source()]).to_csv(original, index=False)
            saved = original.read_bytes()
            run(original, folder / "clean.csv", folder / "reports")
            schema = json.loads((folder / "reports/schema.json").read_text())
            loaded = pd.read_csv(folder / "clean.csv", dtype=schema["dtype"],
                                 parse_dates=schema["parse_dates"])
            self.assertEqual(len(loaded), 2)  # Repeated listing IDs are retained.
            self.assertEqual(loaded.loc[0, "ListingKey"], "000123")
            self.assertEqual(loaded.loc[0, "PostalCode"], "01234")
            self.assertEqual(original.read_bytes(), saved)
            with self.assertRaises(ValueError):
                run(original, original, folder / "reports")

    def test_empty_and_required_schema(self):
        cleaned, tables = clean_data(pd.DataFrame(columns=source().keys()))
        self.assertEqual(len(cleaned), 0)
        self.assertTrue(tables["date_consistency"].flag_count.eq(0).all())
        with self.assertRaisesRegex(ValueError, "Missing required columns"):
            clean_data(pd.DataFrame({"ClosePrice": [100]}))

    def test_notebook_steps_match_script(self):
        script = Path(__file__).with_name("weeks_4_5_data_cleaning.py")
        notebook = json.loads(script.with_suffix(".ipynb").read_text(encoding="utf-8"))
        code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
        self.assertGreaterEqual(len(code_cells), 20)
        self.assertTrue(all(len("".join(cell["source"]).splitlines()) <= 30 for cell in code_cells))
        self.assertGreaterEqual(sum("chart" in cell["metadata"].get("tags", []) for cell in code_cells), 5)
        raw = pd.DataFrame([
            source(), source(ClosePrice="0", LivingArea="inf", DaysOnMarket="1.5"),
            source(ListingContractDate="2024-04-01", PurchaseContractDate="2024-04-02"),
            source(CloseDate="bad", PurchaseContractDate=" N/A "),
            source(Latitude="", Longitude="118"), source(Latitude="0"),
            source(StateOrProvince="NV"), source(Latitude="bad"),
            source(StateOrProvince=""), source(ClosePrice="12,34", DaysOnMarket="1e30"),
        ])
        expected, expected_reports = clean_data(raw)
        namespace = {"__name__": "notebook_test", "raw": raw}
        # Execute the individual transformation cells without loading the full MLS file.
        with contextlib.redirect_stdout(io.StringIO()):
            for cell in code_cells:
                tags = cell["metadata"].get("tags", [])
                if "setup" in tags or "transform" in tags:
                    exec("".join(cell["source"]), namespace)
                    namespace["display"] = lambda *args: None
        self.assertEqual(namespace["PROJECT_DIR"], PROJECT_DIR)
        pd.testing.assert_frame_equal(namespace["sold"], expected)
        for name, report in expected_reports.items():
            pd.testing.assert_frame_equal(namespace["reports"][name], report)


if __name__ == "__main__":
    unittest.main()
