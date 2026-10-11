"""Contracts for point-in-time risk joins and industry/size attribution.

The independent reference below computes partial Spearman by projecting the
original average ranks, then taking Pearson correlation of the residuals.
"""
import unittest

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import industry_size_attribution as study


def risk_fixture():
    panel = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-03", "2024-01-02", "2024-01-03", "2024-01-02"]),
        "asset_id": ["B", "A", "A", "B"],
        "prediction": [.2, .4, .1, .3],
    }, index=[30, 10, 20, 40])
    risk = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-02", "2024-01-03"]),
        "asset_id": ["A", "A", "B", "B"],
        "industry": ["old_A", "new_A", "old_B", "new_B"],
        "market_cap": [10., 11., 20., 21.],
    })
    return panel, risk


def independent_partial_rank(prediction, returns, industry, market_cap, kind):
    """Reference projection with a full set of dummies and pseudoinverse."""
    columns = [np.ones(len(prediction))]
    if kind in {"industry", "industry_size"}:
        columns.extend(np.asarray(industry) == category for category in np.unique(industry))
    if kind in {"size", "industry_size"}:
        columns.append(rankdata(np.log(market_cap), method="average"))
    design = np.column_stack(columns)
    projector = np.eye(len(prediction)) - design @ np.linalg.pinv(design)
    prediction_residual = projector @ rankdata(prediction, method="average")
    return_residual = projector @ rankdata(returns, method="average")
    return np.corrcoef(prediction_residual, return_residual)[0, 1], prediction_residual, return_residual


def fitted_preprocessors(estimator):
    """Inspect fitted preprocessing without depending on pipeline step names."""
    if isinstance(estimator, (StandardScaler, OneHotEncoder)):
        yield estimator
    elif isinstance(estimator, Pipeline):
        for _, child in estimator.steps:
            yield from fitted_preprocessors(child)
    elif isinstance(estimator, ColumnTransformer):
        for _, child, _ in estimator.transformers_:
            if not isinstance(child, str):
                yield from fitted_preprocessors(child)


class IndustrySizeAttributionTests(unittest.TestCase):
    def test_risk_join_uses_both_same_day_asset_keys_and_preserves_row_order(self):
        panel, risk = risk_fixture()
        merged = study.merge_risk(panel, risk.sample(frac=1, random_state=4))
        self.assertEqual(len(merged), len(panel))
        self.assertEqual(list(zip(merged.date, merged.asset_id)), list(zip(panel.date, panel.asset_id)))
        np.testing.assert_array_equal(merged.prediction, panel.prediction)
        self.assertEqual(merged.industry.tolist(), ["new_B", "old_A", "new_A", "old_B"])
        np.testing.assert_array_equal(merged.market_cap, [21., 10., 11., 20.])

    def test_missing_duplicate_or_invalid_risk_data_fail_explicitly(self):
        panel, risk = risk_fixture()
        malformed = {
            "unmatched_same_day_asset": risk.iloc[1:].copy(),
            "duplicate_key": pd.concat([risk, risk.iloc[[0]]], ignore_index=True),
        }
        missing_industry = risk.copy()
        missing_industry.loc[0, "industry"] = None
        malformed["missing_industry"] = missing_industry
        for value in [0., -1., np.nan, np.inf, -np.inf]:
            invalid_cap = risk.copy()
            invalid_cap.loc[0, "market_cap"] = value
            malformed[f"market_cap_{value}"] = invalid_cap
        for name, invalid in malformed.items():
            with self.subTest(case=name), self.assertRaises(ValueError):
                study.merge_risk(panel, invalid)

    def test_industry_projection_is_within_industry_demeaning_and_orthogonal(self):
        frame = pd.DataFrame({"industry": ["A", "B", "A", "C", "B", "C", "C"],
                              "market_cap": [1., 2., 3., 4., 5., 6., 7.]})
        values = np.array([2., 8., 6., -1., 14., 3., 10.])
        design = study.controls_design(frame, "industry")
        residual, r_squared = study.project_residuals(values, design)
        expected = values - pd.Series(values).groupby(frame.industry).transform("mean").to_numpy()
        np.testing.assert_allclose(residual, expected, atol=1e-12)
        np.testing.assert_allclose(design.T @ residual, 0., atol=1e-12)
        self.assertAlmostEqual(r_squared, 1 - np.sum(expected ** 2) / np.sum((values - values.mean()) ** 2))

    def test_size_design_projects_log_cap_or_its_average_rank_as_declared(self):
        log_cap = np.array([2., 5., 1., 1., 3., 6., 4.])
        frame = pd.DataFrame({"industry": ["A"] * len(log_cap), "market_cap": np.exp(log_cap)})
        for rank_size in [False, True]:
            with self.subTest(rank_size=rank_size):
                size = rankdata(log_cap, method="average") if rank_size else log_cap
                values = 7. + 3. * size
                design = study.controls_design(frame, "size", rank_size=rank_size)
                residual, r_squared = study.project_residuals(values, design)
                np.testing.assert_allclose(residual, 0., atol=1e-12)
                self.assertAlmostEqual(r_squared, 1.)

    def test_partial_spearman_matches_rank_projection_without_reranking_residuals(self):
        rng = np.random.default_rng(701)
        n = 96
        industry = np.tile(np.arange(4), 24)
        log_cap = rng.normal(size=n)
        signal = rng.normal(size=n)
        prediction = 1.2 * industry + .7 * log_cap + signal
        returns = -.8 * industry + .6 * log_cap + .5 * signal + rng.normal(size=n)
        market_cap = np.exp(log_cap)
        for kind in ["industry", "size", "industry_size"]:
            with self.subTest(kind=kind):
                expected, pred_residual, return_residual = independent_partial_rank(
                    prediction, returns, industry, market_cap, kind)
                actual = study.partial_rank_correlation(prediction, returns, industry, market_cap, kind)
                self.assertAlmostEqual(actual, expected, places=11)
                if kind == "industry_size":
                    wrong_reranked = spearmanr(pred_residual, return_residual).statistic
                    self.assertGreater(abs(expected - wrong_reranked), .005)

    def test_average_rank_ties_are_preserved_in_partial_correlation(self):
        industry = np.repeat(["A", "B", "C"], 8)
        prediction = np.tile([1., 1., 2., 3., 3., 4., 6., 6.], 3)
        returns = np.array([0., 2., 2., 1., 4., 3., 5., 6.] * 3)
        market_cap = np.exp(np.tile([0., 1., 1., 2., 3., 4., 4., 5.], 3))
        expected, _, _ = independent_partial_rank(prediction, returns, industry, market_cap, "industry_size")
        actual = study.partial_rank_correlation(prediction, returns, industry, market_cap, "industry_size")
        self.assertAlmostEqual(actual, expected, places=11)

    def test_fully_explained_rank_prediction_has_no_defined_partial_correlation(self):
        n = 96
        rng = np.random.default_rng(602)
        industry = np.tile(np.arange(4), 24)
        market_cap = np.exp(rng.uniform(-3., 3., size=n))
        returns = rng.normal(size=n)
        cases = [(industry.astype(float), "industry"),
                 (market_cap, "size"),
                 (market_cap, "industry_size"),
                 (np.ones(n), "industry_size")]
        for prediction, kind in cases:
            with self.subTest(kind=kind, distinct_predictions=len(np.unique(prediction))):
                actual = study.partial_rank_correlation(prediction, returns, industry, market_cap, kind)
                self.assertTrue(np.isnan(actual))

    def test_tied_tail_cutoffs_share_weight_and_ignore_input_row_order(self):
        scores = np.array([0., 1., 1., 2., 3., 4., 5., 6., 7., 8.,
                           9., 10., 11., 12., 13., 14., 15., 18., 18., 20.])
        industry = np.array(["A", "B", "C", "A", "B"] * 4)
        low, high = study.tail_weights(scores)
        expected_low = np.zeros(20)
        expected_low[:3] = [.5, .25, .25]
        expected_high = np.zeros(20)
        expected_high[-3:] = [.25, .25, .5]
        np.testing.assert_allclose(low, expected_low)
        np.testing.assert_allclose(high, expected_high)
        self.assertAlmostEqual(np.sum(low), 1.)
        self.assertAlmostEqual(np.sum(high), 1.)
        order = np.random.default_rng(21).permutation(len(scores))
        reordered_low, reordered_high = study.tail_weights(scores[order])
        np.testing.assert_allclose(reordered_low, low[order])
        np.testing.assert_allclose(reordered_high, high[order])
        for category in np.unique(industry):
            self.assertAlmostEqual(high[industry == category].sum(),
                                   reordered_high[industry[order] == category].sum())

    def test_constant_scores_allocate_identical_equal_weight_tails(self):
        low, high = study.tail_weights(np.ones(20))
        np.testing.assert_allclose(low, np.full(20, .05))
        np.testing.assert_allclose(high, np.full(20, .05))

    def test_control_preprocessing_is_fit_only_on_training_data(self):
        training = pd.DataFrame({"industry": ["A", "B"] * 8,
                                 "log_market_cap": np.arange(16, dtype=float) / 4})
        target = np.sin(np.arange(16)) / 100
        evaluation = pd.DataFrame({"industry": ["A", "B"], "log_market_cap": [1e8, -1e8]})
        for name in ["Ridge", "HGB"]:
            with self.subTest(model=name):
                model = study.make_control_model(name)
                model.fit(training, target)
                preprocessors = list(fitted_preprocessors(model))
                encoders = [item for item in preprocessors if isinstance(item, OneHotEncoder)]
                self.assertEqual(len(encoders), 1)
                np.testing.assert_array_equal(encoders[0].categories_[0], ["A", "B"])
                scalers = [item for item in preprocessors if isinstance(item, StandardScaler)]
                if name == "Ridge":
                    self.assertEqual(len(scalers), 1)
                    np.testing.assert_allclose(scalers[0].mean_, [.5, .5, training.log_market_cap.mean()])
                states = [(item.mean_.copy(), item.scale_.copy()) for item in scalers]
                model.predict(evaluation)
                for scaler, (mean, scale) in zip(scalers, states):
                    np.testing.assert_array_equal(scaler.mean_, mean)
                    np.testing.assert_array_equal(scaler.scale_, scale)
                np.testing.assert_array_equal(encoders[0].categories_[0], ["A", "B"])
                with self.assertRaises(ValueError):
                    model.predict(evaluation.assign(industry=["A", "NEW_INDUSTRY"]))

    def test_evaluation_perturbations_do_not_change_control_fit_statistics_or_other_predictions(self):
        log_cap = np.linspace(1., 5., 80)
        training = pd.DataFrame({"industry": ["A", "B"] * 40,
                                 "market_cap": np.exp(log_cap),
                                 "forward_5d": .01 * log_cap + np.tile([-.003, .003], 40)})
        evaluation = pd.DataFrame({"industry": ["A", "B", "A"],
                                   "market_cap": np.exp([2., 3., 4.]),
                                   "forward_5d": [0., 0., 0.]})
        changed = evaluation.copy()
        changed.loc[0, "market_cap"] = np.exp(30.)
        changed["forward_5d"] = 1e8
        for name in ["Ridge", "HGB"]:
            with self.subTest(model=name):
                prediction, metadata = study.fit_controls(training, evaluation, name)
                changed_prediction, changed_metadata = study.fit_controls(training, changed, name)
                np.testing.assert_allclose(prediction[1:], changed_prediction[1:], rtol=0., atol=1e-14)
                self.assertAlmostEqual(metadata["training_mean"], training.forward_5d.mean())
                for key in ["training_mean", "industry_categories", "feature_names", "parameters"]:
                    self.assertEqual(metadata[key], changed_metadata[key])
                if name == "Ridge":
                    for key in ["x_scaler_mean", "x_scaler_scale", "x_scaler_var"]:
                        np.testing.assert_array_equal(metadata[key], changed_metadata[key])
                    np.testing.assert_allclose(metadata["x_scaler_mean"], [.5, .5, log_cap.mean()])

    def test_partial_summary_compares_only_jointly_defined_dates(self):
        raw = np.linspace(.05, .35, 40)
        raw[:5] = .9
        partial = raw + .02
        partial[:5] = np.nan
        part = pd.DataFrame({
            "date": pd.bdate_range("2024-01-02", periods=40),
            "raw_rank_ic": raw,
            "partial_rank_ic": partial,
            "status": ["zero_prediction_residual"] * 5 + ["defined"] * 35,
            "prediction_rank_control_r_squared": [.1] * 40,
            "return_rank_control_r_squared": [.2] * 40,
        })
        summary = study.summarize_partial(part)
        self.assertEqual(summary["signal_dates"], 40)
        self.assertEqual(summary["common_valid_dates"], 35)
        self.assertAlmostEqual(summary["raw_on_common_mean"], raw[5:].mean())
        self.assertAlmostEqual(summary["partial_on_common_mean"], partial[5:].mean())
        self.assertAlmostEqual(summary["partial_minus_raw_mean"], .02)
        self.assertEqual(summary["partial_minus_raw_n_dates"], 35)
        self.assertEqual(summary["status_zero_prediction_residual_dates"], 5)


if __name__ == "__main__":
    unittest.main()
