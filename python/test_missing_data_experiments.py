"""Checks for masking, leakage prevention and conservative output handling."""
import unittest

import numpy as np
import pandas as pd

from weeks_4_5_missing_data_experiments import (
    TARGETS, group_predictions, mask_indices, matching_donor_counts, predictors, prepare_data,
)


class MissingDataTests(unittest.TestCase):
    def fixture(self):
        rows = 100
        frame = pd.DataFrame({"SourceRecordNumber": range(1, rows + 1),
                              "ListingKey": pd.Series([str(i) for i in range(rows)], dtype="string"),
                              "CountyOrParish": "Orange", "PropertySubType": "Condominium", "SourceMonth": "202401",
                              "CloseDate": pd.Timestamp("2024-01-01"), "ClosePrice": 500000.0,
                              "ListPrice": np.arange(rows) * 1000 + 400000.0, "Latitude": 34.0, "Longitude": -118.0})
        for field in TARGETS:
            frame[field] = pd.Series(np.arange(rows) + 1000.0, dtype="Float64")
        frame["Unused"] = pd.NA
        return frame

    def test_mar_masks_do_not_depend_on_hidden_target(self):
        frame = self.fixture()
        first = mask_indices(frame, np.arange(100), "MAR", 42)
        second = mask_indices(frame, np.arange(100)[::-1], "MAR", 42)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(len(set(first)), 20)

    def test_nmar_masks_prefer_the_assumed_tail(self):
        frame = pd.concat([self.fixture()] * 100, ignore_index=True)
        truth = np.arange(len(frame))
        high = mask_indices(frame, truth, "NMAR_high", 42)
        low = mask_indices(frame, truth, "NMAR_low", 42)
        self.assertGreater(truth[high].mean(), truth.mean())
        self.assertLess(truth[low].mean(), truth.mean())

    def test_predictors_exclude_target_and_missing_flags(self):
        frame = self.fixture()
        frame["LivingArea_invalid_flag"] = True
        x, numeric = predictors(frame, "LivingArea")
        self.assertNotIn("LivingArea", x.columns)
        self.assertNotIn("LivingArea", numeric)
        self.assertFalse(any(column.endswith("_flag") for column in x))

    def test_group_medians_use_training_values_only(self):
        train, test = self.fixture(), self.fixture().iloc[:3].copy()
        test["LivingArea"] = 1e12
        result = group_predictions(train, test, "LivingArea")
        np.testing.assert_allclose(result, train.LivingArea.median())

    def test_unsupported_missing_rows_stay_missing_and_observed_values_survive(self):
        data = self.fixture()
        data.loc[[0, 1], "LivingArea"] = pd.NA
        data.loc[1, "PropertySubType"] = "BoatSlip"  # No observed comparable donors.
        counts = matching_donor_counts(data, "LivingArea")
        self.assertEqual(counts.iloc[0], 98)
        self.assertEqual(counts.iloc[1], 0)
        decisions = pd.DataFrame([
            {"field": "LivingArea", "action": "impute", "best_tested_method": "group_median", "integer_output": False},
            {"field": "Unused", "action": "drop_column", "best_tested_method": "none", "integer_output": False},
        ])
        prepared, audit, sensitivity = prepare_data(data, decisions)
        self.assertEqual(len(prepared), len(data))
        self.assertTrue(prepared.loc[0, "LivingArea_imputed_flag"])
        self.assertFalse(prepared.loc[1, "LivingArea_imputed_flag"])
        self.assertTrue(pd.isna(prepared.loc[1, "LivingArea"]))
        self.assertEqual(len(audit), 1)
        self.assertEqual(set(sensitivity.imputed_cells), {1})
        self.assertNotIn("Unused", prepared)
        pd.testing.assert_series_equal(prepared.LivingArea.iloc[2:], data.LivingArea.iloc[2:])


if __name__ == "__main__":
    unittest.main()
