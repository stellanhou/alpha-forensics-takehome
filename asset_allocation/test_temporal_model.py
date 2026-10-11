"""Temporal-boundary, prediction-universe and fixed-training contracts."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd

from factor_timing import FACTORS, KEYS
from temporal_model import (
    Fold, compare_old_predictions, fit_fold, load_signal_panel, make_folds,
    make_model, split_fold, validate_coverage,
)


class TemporalModelTests(unittest.TestCase):
    def test_unmatured_historical_answers_are_excluded_at_cutoff(self):
        panel = pd.DataFrame({
            "date": pd.to_datetime(["2021-12-20", "2021-12-23", "2021-12-30", "2022-01-03"]),
            "asset_id": "A",
            "exit_date": pd.to_datetime(["2021-12-28", "2021-12-31", "2022-01-07", "2022-01-11"]),
            "forward_5d": [1., 2., 3., np.nan],
        })
        train, signals = split_fold(panel, make_folds()[0])
        self.assertEqual(train.date.dt.strftime("%Y-%m-%d").tolist(), ["2021-12-20", "2021-12-23"])
        self.assertEqual(len(signals), 1)
        self.assertTrue(signals.forward_5d.isna().all())

    def test_fit_has_no_random_internal_holdout_and_predicts_missing_label(self):
        dates = pd.bdate_range("2021-10-01", "2022-01-05")
        panel = pd.DataFrame({"date": dates, "asset_id": "A"})
        panel["entry_date"] = panel.date + pd.offsets.BDay(1)
        panel["exit_date"] = panel.date + pd.offsets.BDay(6)
        panel["forward_5d"] = np.sin(np.arange(len(panel))) / 100
        panel.loc[panel.date >= "2022-01-03", "forward_5d"] = np.nan
        for i, factor in enumerate(FACTORS):
            panel[factor] = np.cos(np.arange(len(panel)) + i)
        self.assertFalse(make_model().get_params()["early_stopping"])
        model, predictions, record = fit_fold(panel, make_folds()[0])
        self.assertFalse(model.do_early_stopping_)
        self.assertEqual(model.n_iter_, 55)
        self.assertEqual(len(predictions), 3)
        self.assertEqual(record["prediction_rows_without_label"], 3)
        self.assertTrue(np.isfinite(predictions.prediction).all())
        self.assertLessEqual(record["train_label_exit_max"], record["train_cutoff"])

    def test_folds_partition_all_synthetic_dates_without_train_overlap(self):
        dates = pd.bdate_range("2021-01-04", "2024-12-31")
        panel = pd.DataFrame({"date": dates, "asset_id": "A", "forward_5d": .01})
        panel["exit_date"] = panel.date + pd.offsets.BDay(6)
        predictions = []
        folds = make_folds()
        self.assertEqual(len(folds), 9)
        for fold in folds:
            train, signals = split_fold(panel, fold)
            self.assertEqual(len(train.set_index(KEYS).index.intersection(signals.set_index(KEYS).index)), 0)
            predictions.append(signals[KEYS].assign(train_cutoff=fold.train_cutoff))
        predictions = pd.concat(predictions, ignore_index=True)
        expected = panel.loc[panel.date >= "2022-01-03", KEYS].reset_index(drop=True)
        pd.testing.assert_frame_equal(predictions[KEYS], expected)
        validate_coverage(predictions, panel, folds)
        with self.assertRaisesRegex(AssertionError, "coverage"):
            validate_coverage(predictions.iloc[:-1], panel, folds)

    def test_pit_loader_keeps_terminal_rows_and_aligns_before_membership(self):
        dates = pd.bdate_range("2021-01-04", periods=12)
        keys = pd.DataFrame({"date": dates, "asset_id": "A"})
        market = keys.assign(open=np.arange(12.) + 100)
        factors = keys.assign(**{factor: np.arange(12.) + i * 100 for i, factor in enumerate(FACTORS)})
        universe = keys.assign(is_listed=True, is_member=np.arange(12) >= 3)
        availability = keys.assign(
            factor_03_available_on=dates + pd.offsets.BDay(3),
            factor_12_available_on=dates + pd.offsets.BDay(1), published_at="after_close",
        )
        with TemporaryDirectory() as tmp:
            data = Path(tmp) / "data"
            data.mkdir()
            for name, frame in [("market_daily", market), ("factors", factors),
                                ("universe", universe), ("factor_availability", availability)]:
                frame.to_csv(data / f"{name}.csv.gz", index=False)
            panel = load_signal_panel(Path(tmp))
        self.assertEqual(len(panel), 9)
        self.assertEqual(panel.iloc[0].factor_03_reference_date, dates[0])
        self.assertEqual(panel.iloc[-1].date, dates[-1])
        self.assertEqual(panel.forward_5d.isna().sum(), 6)
        self.assertEqual(panel.iloc[-1].factor_01, factors.iloc[-1].factor_01)

    def test_comparison_reranks_identical_keys_and_labels_old_arms(self):
        keys = pd.DataFrame({"date": pd.to_datetime(["2023-01-03"] * 70), "asset_id": [f"A{i:03}" for i in range(70)]})
        predictions = keys.assign(prediction=np.arange(70.), forward_5d=np.arange(70.) / 1000)
        predictions.loc[69, "forward_5d"] = np.nan
        with TemporaryDirectory() as tmp:
            old_root = Path(tmp)
            for variant in ["original", "pit"]:
                (old_root / variant).mkdir()
                old_rows = predictions.iloc[:65] if variant == "original" else predictions.iloc[5:]
                old_rows[KEYS + ["prediction"]].to_csv(
                    old_root / variant / "baseline_predictions.csv.gz", index=False,
                )
            comparison, daily = compare_old_predictions(predictions, old_root)
        combined = comparison.loc[comparison.period == "2023_2024_oos"]
        self.assertEqual(combined.labelled_rows.tolist(), [60, 60, 60])
        self.assertEqual(combined.new_signal_rows.tolist(), [70, 70, 70])
        self.assertEqual(combined.new_labelled_rows.tolist(), [69, 69, 69])
        self.assertEqual(combined.common_sample_rows.tolist(), [60, 60, 60])
        self.assertEqual(combined.original_unmatched_labelled_rows.tolist(), [4, 4, 4])
        self.assertEqual(combined.pit_unmatched_labelled_rows.tolist(), [5, 5, 5])
        self.assertEqual(combined.common_excluded_labelled_rows.tolist(), [9, 9, 9])
        self.assertEqual(combined.common_excluded_signal_rows.tolist(), [10, 10, 10])
        self.assertEqual(combined.is_temporal_oos.tolist(), [False, False, True])
        self.assertEqual(combined.common_keys_sha256.nunique(), 1)
        self.assertEqual(daily.long_short_5d_spread.nunique(), 1)


if __name__ == "__main__":
    unittest.main()
