"""Final policy, exact conversions and preservation checks."""
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd

from weeks_4_5_missing_data_finalization import derive_lot_sizes, final_decisions
import test_missing_data_experiments as experiment_tests


class FinalizationTests(unittest.TestCase):
    def test_exact_conversion_without_guessing_zero_or_unknown_units(self):
        original = pd.DataFrame({"SourceRecordNumber": range(1, 7),
            "LotSizeAcres": pd.Series([pd.NA, 2, pd.NA, 0, 1, pd.NA], dtype="Float64"),
            "LotSizeSquareFeet": pd.Series([43560, pd.NA, 0, pd.NA, 1000, pd.NA], dtype="Float64"),
            "LotSizeArea": [999, 999, 999, 999, 999, 999]})
        prepared = original.copy()
        audit = derive_lot_sizes(original, prepared)
        self.assertEqual(prepared.loc[0, "LotSizeAcres"], 1)
        self.assertEqual(prepared.loc[1, "LotSizeSquareFeet"], 87120)
        self.assertTrue(pd.isna(prepared.loc[2, "LotSizeAcres"]))
        self.assertTrue(pd.isna(prepared.loc[3, "LotSizeSquareFeet"]))
        self.assertTrue(pd.isna(prepared.loc[5, "LotSizeAcres"]))
        self.assertTrue(prepared.loc[4, "lot_size_conflict_flag"])
        self.assertEqual(prepared.loc[4, "LotSizeSquareFeet"], 1000)
        self.assertEqual(len(audit), 2)
        for field in ["LotSizeAcres", "LotSizeSquareFeet"]:
            observed = original[field].notna()
            pd.testing.assert_series_equal(original.loc[observed, field], prepared.loc[observed, field])

    def test_mean_bias_guard_revises_the_exploratory_price_imputation(self):
        data = experiment_tests.MissingDataTests().fixture()
        profile = pd.DataFrame({"field": ["OriginalListPrice", "BathroomsTotalInteger"]})
        preliminary = pd.DataFrame({"field": ["OriginalListPrice", "BathroomsTotalInteger"],
            "action": ["impute", "impute"], "best_tested_method": ["model", "model"], "reason": ["", ""]})
        evidence = pd.DataFrame({"field": ["OriginalListPrice", "BathroomsTotalInteger"],
            "method": ["model", "model"], "scenario": ["NMAR_high", "NMAR_high"],
            "abs_mean_bias_pct": [9.35, 3.73]})
        with patch("weeks_4_5_missing_data_finalization.experiments.profile_missingness", return_value=(profile, None, None)), \
             patch("weeks_4_5_missing_data_finalization.experiments.actual_donor_support", return_value=None), \
             patch("weeks_4_5_missing_data_finalization.experiments.choose_policy", return_value=preliminary):
            decisions = final_decisions(data, evidence).set_index("field")
        self.assertEqual(decisions.loc["OriginalListPrice", "action"], "keep_null")
        self.assertEqual(decisions.loc["BathroomsTotalInteger", "action"], "impute")


if __name__ == "__main__":
    unittest.main()
