"""Tests for the declared label, preprocessing, ranking and evidence contracts."""
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from compare_models import (
    CANDIDATES, FACTORS, daily_metrics, decile_spread, model_pipeline, moving_block_interval,
    object_hash, paired_mean_difference, phase_numbers, sha256, temporal_split,
    select_candidate, validate_panel, verify_confirmation, write_json,
)


def example_panel(dates):
    dates = pd.DatetimeIndex(dates)
    frame = pd.DataFrame({"date": dates, "asset_id": "A",
                          "entry_date": dates + pd.offsets.BDay(1),
                          "exit_date": dates + pd.offsets.BDay(6),
                          "forward_5d": np.arange(len(dates)) / 100})
    return frame.assign(**{factor: np.arange(len(dates), dtype=float) for factor in FACTORS})


class ModelComparisonTests(unittest.TestCase):
    def test_real_five_day_label_boundaries_are_purged_in_both_stages(self):
        panel = example_panel([
            "2022-12-22", "2022-12-23", "2022-12-26", "2023-01-02",
            "2023-12-21", "2023-12-22", "2024-01-01",
        ])
        train, evaluation, manifest = temporal_split(panel, "validation")
        self.assertEqual(train.date.tolist(), [pd.Timestamp("2022-12-22")])
        self.assertEqual(manifest["purged_training_rows"], 2)
        self.assertEqual(manifest["purged_evaluation_rows"], 1)
        self.assertEqual(evaluation.date.tolist(),
                         [pd.Timestamp("2023-01-02"), pd.Timestamp("2023-12-21")])
        train, evaluation, _ = temporal_split(panel, "confirmation")
        self.assertIn(pd.Timestamp("2023-12-21"), train.date.tolist())
        self.assertNotIn(pd.Timestamp("2023-12-22"), train.date.tolist())
        self.assertLess(train.exit_date.max(), evaluation.date.min())

    def test_preprocessors_are_fit_only_on_training_rows(self):
        x = pd.DataFrame({"first": [0., 2., np.nan, 4.], "second": [2., 4., 6., 8.]})
        evaluation = pd.DataFrame({"first": [1e9, np.nan], "second": [-1e9, np.nan]})
        model = model_pipeline(Ridge(alpha=100), linear=True)
        model.fit(x, np.array([1., 2., 3., 4.]))
        model.predict(evaluation)
        np.testing.assert_array_equal(model.named_steps["imputer"].statistics_, [2., 5.])
        np.testing.assert_array_equal(model.named_steps["scaler"].mean_, [2., 5.])
        np.testing.assert_allclose(model.named_steps["scaler"].scale_, [np.sqrt(2), np.sqrt(5)])

    def test_constant_scores_have_no_rank_ic_or_decile_spread(self):
        frame = pd.DataFrame({"date": pd.Timestamp("2023-01-02"),
                              "asset_id": list("ABCDEFGHIJ"), "prediction": np.ones(10),
                              "forward_5d": np.arange(10, dtype=float)})
        for part in [frame, frame.sample(frac=1, random_state=9)]:
            result = daily_metrics(part).iloc[0]
            self.assertTrue(np.isnan(result.rank_ic))
            self.assertTrue(np.isnan(result.gross_decile_spread))

    def test_tied_cutoffs_share_weight_and_ignore_asset_order(self):
        scores = np.array([0., 0., 1., 2., 3., 4., 5., 6., 7., 7.])
        returns = np.array([1., 3., 0., 0., 0., 0., 0., 0., 10., 20.])
        self.assertAlmostEqual(decile_spread(scores, returns), 13.)
        order = np.random.default_rng(4).permutation(10)
        self.assertAlmostEqual(decile_spread(scores[order], returns[order]), 13.)

    def test_phase_uses_business_calendar_and_survives_missing_dates(self):
        dates = pd.to_datetime(["2021-01-01", "2021-01-04", "2021-01-05", "2021-01-08"])
        np.testing.assert_array_equal(phase_numbers(dates), [0, 1, 2, 0])
        np.testing.assert_array_equal(phase_numbers(dates.delete(1)), [0, 2, 0])

    def test_paired_bootstrap_uses_same_dates_and_jointly_defined_values(self):
        dates = pd.bdate_range("2023-01-02", periods=50)
        reference = pd.Series(np.arange(40.), index=dates[:40])
        candidate = pd.Series(np.arange(5., 45.) + 2, index=dates[5:45])
        candidate.loc[dates[10]] = np.nan
        result = paired_mean_difference(candidate, reference, draws=100)
        self.assertEqual(result["n_dates"], 34)
        self.assertEqual(result["mean"], 2.)
        self.assertEqual(result["ci_low"], 2.)
        self.assertEqual(result["ci_high"], 2.)
        identical = paired_mean_difference(reference, reference, draws=100)
        self.assertEqual(identical["ci_low"], 0.)
        self.assertEqual(identical["ci_high"], 0.)

    def test_fixed_bootstrap_is_reproducible_and_never_shortens_blocks(self):
        dates = pd.bdate_range("2023-01-02", periods=50)
        series = pd.Series(np.arange(50.) ** 2, index=dates)
        self.assertEqual(moving_block_interval(series, draws=100),
                         moving_block_interval(series, draws=100))
        with self.assertRaisesRegex(ValueError, "Fewer than 20"):
            moving_block_interval(series.iloc[:19], draws=100)

    def test_broken_panel_contracts_fail_explicitly(self):
        panel = example_panel(["2022-01-03", "2023-01-02"])
        duplicate = pd.concat([panel, panel.iloc[[0]]], ignore_index=True)
        with self.assertRaises(ValueError):
            validate_panel(duplicate)
        broken_label = panel.copy()
        broken_label.loc[0, "exit_date"] += pd.offsets.BDay(1)
        with self.assertRaisesRegex(ValueError, "Label exit_date"):
            validate_panel(broken_label)
        infinite = panel.copy()
        infinite.loc[0, FACTORS[0]] = np.inf
        with self.assertRaisesRegex(ValueError, "Infinite factor"):
            validate_panel(infinite)
        missing = panel.copy()
        missing.loc[0, FACTORS[0]] = np.nan
        with self.assertRaisesRegex(ValueError, "entirely missing"):
            temporal_split(missing, "validation")

    def test_confirmation_rejects_changed_code_or_validation_evidence(self):
        protocol = {"parameters_sha256": "params", "code_sha256": {"source": "code"}}
        digest = object_hash(protocol)
        with TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / "validation").mkdir()
            write_json(out / "protocol.json", {"protocol": protocol, "protocol_sha256": digest})
            selection = {"selected_model": "HGB", "protocol_sha256": digest,
                         "parameters_sha256": "params", "code_sha256": {"source": "code"}}
            write_json(out / "selection.json", selection)
            write_json(out / "validation" / "run_manifest.json", {
                "status": "completed", "protocol_sha256": digest,
                "outputs_sha256": {"selection.json": sha256(out / "selection.json")},
            })
            self.assertEqual(verify_confirmation(out, protocol), selection)
            with self.assertRaisesRegex(ValueError, "Frozen protocol"):
                verify_confirmation(out, {**protocol, "code_sha256": {"source": "changed"}})
            (out / "selection.json").write_text(json.dumps({**selection, "selected_model": "Ridge"}))
            with self.assertRaisesRegex(ValueError, "Validation output changed"):
                verify_confirmation(out, protocol)

    def test_constant_candidate_is_reported_ineligible_without_hiding_other_results(self):
        rows = [{"model": name, "rank_ic_mean": np.nan} for name in CANDIDATES]
        with self.assertRaisesRegex(ValueError, "no valid selection"):
            select_candidate(rows)
        rows[1]["rank_ic_mean"] = .05
        chosen, scores, reasons = select_candidate(rows)
        self.assertEqual(chosen, "Ridge")
        self.assertIsNone(scores["HGB"])
        self.assertIn("No defined daily RankIC", reasons["HGB"])
        self.assertNotIn("Ridge", reasons)


if __name__ == "__main__":
    unittest.main()
