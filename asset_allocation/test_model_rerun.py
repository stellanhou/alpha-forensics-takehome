"""Checks that the timing intervention keeps sample membership and coverage."""
import unittest

import numpy as np
import pandas as pd

from retrain_pit_model import matched_split, validate_predictions


class ModelRerunTests(unittest.TestCase):
    def setUp(self):
        self.raw = pd.DataFrame({
            "date": pd.bdate_range("2021-01-04", periods=20).strftime("%Y-%m-%d"),
            "asset_id": "A", "factor_03": np.arange(20.),
        })

    def test_missing_pit_values_do_not_change_split_membership(self):
        pit = self.raw.copy()
        pit.loc[:2, "factor_03"] = np.nan
        sample, train, valid, assignments, proof = matched_split(self.raw, pit)
        self.assertEqual((len(train), len(valid), len(sample)), (16, 4, 20))
        self.assertEqual(sample.factor_03.isna().sum(), 3)
        self.assertEqual(assignments.split.value_counts().to_dict(), {"train": 16, "valid": 4})
        for name in ["train", "valid"]:
            self.assertEqual(proof[name]["original_ordered_keys_sha256"], proof[name]["pit_ordered_keys_sha256"])

    def test_different_input_order_cannot_silently_change_samples(self):
        pit = self.raw.iloc[::-1].reset_index(drop=True)
        with self.assertRaises(AssertionError):
            matched_split(self.raw, pit)

    def test_missing_prediction_rows_are_rejected_before_inner_join(self):
        predictions = self.raw[["date", "asset_id"]].assign(prediction=0.)
        validate_predictions(predictions, self.raw)
        with self.assertRaises(AssertionError):
            validate_predictions(predictions.iloc[:-1], self.raw)


if __name__ == "__main__":
    unittest.main()
