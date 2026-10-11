"""Contract tests for purged, training-only factor selection and frozen evidence."""
import json
import unittest
import warnings
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from sklearn.exceptions import ConvergenceWarning

import factor_selection_study as study
from compare_models import FACTORS, object_hash, sha256, write_json


def example_panel(dates, n_assets=8):
    """Deterministic panel following the existing t+1/t+6 label contract."""
    dates = pd.DatetimeIndex(dates)
    rng = np.random.default_rng(17)
    frame = pd.DataFrame({
        "date": np.repeat(dates, n_assets),
        "asset_id": np.tile([f"A{i:03d}" for i in range(n_assets)], len(dates)),
    })
    frame["entry_date"] = frame.date + pd.offsets.BDay(1)
    frame["exit_date"] = frame.date + pd.offsets.BDay(6)
    frame["forward_5d"] = rng.normal(size=len(frame)) / 100
    for factor in FACTORS:
        frame[factor] = rng.normal(size=len(frame))
    return frame


def complete_fold_scores():
    return pd.DataFrame([
        {"learner": learner, "selector_id": candidate["selector_id"],
         "family": candidate["family"], "fold": fold, "status": "eligible",
         "reason": "", "rank_ic_mean": .01, "rank_ic_valid_dates": 20}
        for learner in study.LEARNERS
        for candidate in study.SELECTOR_CANDIDATES
        for fold in study.INNER_FOLDS
    ])


class FactorSelectionStudyTests(unittest.TestCase):
    def test_each_inner_fold_purges_actual_label_exit_at_its_boundary(self):
        panel = example_panel(pd.bdate_range("2021-01-01", "2022-12-30"), n_assets=2)
        panel = panel.loc[panel.exit_date < pd.Timestamp("2023-01-02")].copy()
        folds = study.inner_splits(panel)
        self.assertEqual([fold[0] for fold in folds], list(study.INNER_FOLDS))
        for name, train, evaluation, _ in folds:
            with self.subTest(fold=name):
                boundary = evaluation.date.min()
                expected = panel.loc[(panel.date < boundary) & (panel.exit_date < boundary)]
                actual_keys = train[["date", "asset_id"]].sort_values(["date", "asset_id"])
                expected_keys = expected[["date", "asset_id"]].sort_values(["date", "asset_id"])
                pd.testing.assert_frame_equal(actual_keys.reset_index(drop=True),
                                              expected_keys.reset_index(drop=True))
                self.assertLess(train.exit_date.max(), boundary)
                self.assertEqual(set(evaluation.date.dt.to_period("Q").astype(str)), {name})
                next_quarter = pd.offsets.BDay().rollforward((pd.Period(name, freq="Q") + 1).start_time)
                self.assertLess(evaluation.exit_date.max(), next_quarter)
                if name == "2022Q4":
                    self.assertTrue((evaluation.exit_date.dt.year <= 2022).all())
                crossing = panel.loc[(panel.date < boundary) & (panel.exit_date >= boundary)]
                self.assertFalse(crossing.empty)
                self.assertTrue(crossing.exit_date.eq(boundary).any())
                self.assertTrue(set(train.date).isdisjoint(crossing.date))

    def test_future_rows_do_not_change_inner_training_selectors(self):
        dates = pd.bdate_range("2021-12-01", "2022-12-30")
        panel = example_panel(dates)
        panel = panel.loc[panel.exit_date < pd.Timestamp("2023-01-02")].copy()
        original_train = study.inner_splits(panel)[0][1]
        changed = panel.copy()
        future = changed.date >= pd.Timestamp("2022-01-03")
        changed.loc[future, FACTORS] = 1e9
        changed.loc[future, "forward_5d"] = -1e9
        changed_train = study.inner_splits(changed)[0][1]
        pd.testing.assert_frame_equal(original_train, changed_train)
        before = study.selector_statistics(original_train)
        after = study.selector_statistics(changed_train)
        for key in ["ic", "daily_ic", "quarterly_ic", "mean_corr", "pair_valid_days"]:
            pd.testing.assert_frame_equal(before[key], after[key])
        self.assertEqual(before["imputer_medians"], after["imputer_medians"])
        np.testing.assert_array_equal(before["distance"], after["distance"])
        self.assertEqual(study.select_ic(before, 5), study.select_ic(after, 5))
        self.assertEqual(study.select_hac(before, 5)[0], study.select_hac(after, 5)[0])

    def test_hac_clusters_opposite_sign_duplicates_using_condensed_distances(self):
        panel = example_panel(["2021-01-04", "2021-04-05", "2021-07-05", "2021-10-04"],
                              n_assets=40)
        panel[FACTORS[1]] = -panel[FACTORS[0]]
        panel["forward_5d"] = panel[FACTORS[0]] / 100
        stats = study.selector_statistics(panel)
        self.assertAlmostEqual(stats["mean_corr"].loc[FACTORS[0], FACTORS[1]], -1.)
        expected_distance = 1 - np.abs(stats["mean_corr"].to_numpy())
        np.fill_diagonal(expected_distance, 0.)
        np.testing.assert_allclose(stats["distance"], expected_distance, atol=1e-12)
        expected_linkage = linkage(squareform(expected_distance, checks=False), method="average")
        np.testing.assert_allclose(stats["linkage"], expected_linkage, atol=1e-12)
        features, clusters = study.select_hac(stats, 15)
        self.assertEqual(len(features), 15)
        self.assertEqual(clusters.cluster.nunique(), 15)
        indexed = clusters.set_index("factor")
        self.assertEqual(indexed.loc[FACTORS[0], "cluster"],
                         indexed.loc[FACTORS[1], "cluster"])
        self.assertTrue(indexed.loc[FACTORS[0], "selected"])
        self.assertFalse(indexed.loc[FACTORS[1], "selected"])
        self.assertEqual(features, [factor for factor in FACTORS if factor in features])

    def test_exact_hac_cluster_count_survives_tied_distances_and_representation_ranking(self):
        ic = pd.DataFrame({
            "factor": FACTORS, "mean_daily_ic": [.2] * len(FACTORS),
            "abs_mean_daily_ic": [.2] * len(FACTORS),
            "valid_ic_days": [20] * len(FACTORS),
            "quarter_sign_stability": [1.] * len(FACTORS),
            "ic_rank": np.arange(1, len(FACTORS) + 1),
        })
        distance = np.ones((len(FACTORS), len(FACTORS))) - np.eye(len(FACTORS))
        stats = {"ic": ic, "distance": distance,
                 "linkage": linkage(squareform(distance), method="average")}
        for k in [5, 10, 15]:
            with self.subTest(k=k):
                features, clusters = study.select_hac(stats, k)
                self.assertEqual(len(features), k)
                self.assertEqual(clusters.cluster.nunique(), k)
                self.assertEqual(int(clusters.selected.sum()), k)
                for _, group in clusters.groupby("cluster"):
                    self.assertEqual(group.loc[group.selected, "factor"].item(),
                                     min(group.factor))

    def test_ic_priority_is_absolute_mean_then_sign_stability_then_factor_id(self):
        panel = example_panel(["2021-01-04", "2021-01-05", "2021-01-06", "2021-04-05",
                               "2021-07-05", "2021-10-04", "2021-10-05", "2021-10-06"],
                              n_assets=3)
        panel["forward_5d"] = np.tile([0., 1., 2.], 8)
        rank_values = {.5: [0., 2., 1.], -.5: [2., 0., 1.]}
        correlations = {
            FACTORS[0]: [.5, .5, .5, -.5, -.5, .5, .5, .5],
            FACTORS[1]: [.5, .5, -.5, .5, .5, .5, .5, -.5],
            FACTORS[2]: [-.5, -.5, .5, -.5, -.5, -.5, -.5, .5],
            FACTORS[3]: [.5] * 8,
        }
        for factor, values in correlations.items():
            panel[factor] = np.concatenate([rank_values[value] for value in values])
        ic = study.selector_statistics(panel)["ic"].set_index("factor")
        self.assertAlmostEqual(ic.loc[FACTORS[0], "abs_mean_daily_ic"], .25)
        self.assertEqual(ic.loc[FACTORS[0], "abs_mean_daily_ic"],
                         ic.loc[FACTORS[1], "abs_mean_daily_ic"])
        self.assertEqual(ic.loc[FACTORS[1], "abs_mean_daily_ic"],
                         ic.loc[FACTORS[2], "abs_mean_daily_ic"])
        self.assertEqual(ic.loc[FACTORS[1], "quarter_sign_stability"], 1.)
        self.assertEqual(ic.loc[FACTORS[0], "quarter_sign_stability"], .5)
        self.assertLess(ic.loc[FACTORS[3], "ic_rank"], ic.loc[FACTORS[1], "ic_rank"])
        self.assertLess(ic.loc[FACTORS[1], "ic_rank"], ic.loc[FACTORS[2], "ic_rank"])
        self.assertLess(ic.loc[FACTORS[2], "ic_rank"], ic.loc[FACTORS[0], "ic_rank"])

    def test_elastic_net_empty_support_is_explicitly_ineligible(self):
        panel = example_panel(["2021-01-04", "2021-04-05"], n_assets=4)
        x = np.tile([-3., -1., 1., 3.], 2)
        for index, factor in enumerate(FACTORS):
            panel[factor] = x * (1 if index % 2 else -1)
        panel["forward_5d"] = x ** 2 / 100
        result = study.fit_elastic_selector(panel, .001)
        self.assertEqual(result["status"], "ineligible")
        self.assertEqual(result["features"], [])
        self.assertEqual(result["nonzero_count"], 0)
        self.assertTrue(result["reason"])

    def test_strong_elastic_net_penalty_does_not_fall_back_from_empty_support(self):
        panel = example_panel(["2021-01-04", "2021-04-05"], n_assets=20)
        panel["forward_5d"] = panel[FACTORS[0]] / 100
        result = study.fit_elastic_selector(panel, 1e6)
        self.assertEqual(result["status"], "ineligible")
        self.assertEqual(result["features"], [])
        self.assertEqual(result["nonzero_count"], 0)
        self.assertTrue(result["reason"])

    def test_elastic_net_convergence_warning_makes_nonempty_support_ineligible(self):
        panel = example_panel(["2021-01-04", "2021-04-05"], n_assets=20)
        panel["forward_5d"] = panel[FACTORS[0]] / 100
        model = study.elastic_estimator(.001)
        original_fit = model.fit

        def fit_with_warning(x, y):
            fitted = original_fit(x, y)
            warnings.warn("Fixture convergence warning", ConvergenceWarning)
            return fitted

        with patch.object(study, "elastic_estimator", return_value=model), \
                patch.object(model, "fit", side_effect=fit_with_warning):
            result = study.fit_elastic_selector(panel, .001)
        self.assertGreater(result["nonzero_count"], 0)
        self.assertEqual(result["status"], "ineligible")
        self.assertIn("Fixture convergence warning", result["reason"])

    def test_one_invalid_fold_cannot_be_skipped_to_select_an_en_configuration(self):
        scores = complete_fold_scores()
        scores.loc[scores.family == "EN", "rank_ic_mean"] = .99
        invalid = (scores.family == "EN") & (scores.fold == study.INNER_FOLDS[0])
        scores.loc[invalid, "status"] = "ineligible"
        scores.loc[invalid, "reason"] = "ElasticNet selected zero factors"
        scores.loc[invalid, "rank_ic_mean"] = np.nan
        scores.loc[invalid, "rank_ic_valid_dates"] = 0
        selected, _ = study.choose_inner_configs(scores)
        for learner in study.LEARNERS:
            self.assertIsNone(selected[learner]["EN"])
            self.assertIsNotNone(selected[learner]["IC"])
            self.assertIsNotNone(selected[learner]["HAC"])

    def test_inner_score_is_weighted_by_defined_dates_and_ties_use_declared_order(self):
        scores = complete_fold_scores()
        first = scores.selector_id.eq("IC_k5")
        scores.loc[first, "rank_ic_mean"] = .10
        scores.loc[first, "rank_ic_valid_dates"] = 1
        long_fold = first & scores.fold.eq(study.INNER_FOLDS[-1])
        scores.loc[long_fold, "rank_ic_mean"] = 0.
        scores.loc[long_fold, "rank_ic_valid_dates"] = 97
        scores.loc[scores.selector_id.eq("IC_k10"), "rank_ic_mean"] = .02
        selected, _ = study.choose_inner_configs(scores)
        for learner in study.LEARNERS:
            self.assertEqual(selected[learner]["IC"]["selector_id"], "IC_k10")
            self.assertEqual(selected[learner]["HAC"]["selector_id"], "HAC_k5")

    def test_missing_or_duplicate_inner_fold_is_an_incomplete_experiment(self):
        scores = complete_fold_scores()
        for malformed in [scores.iloc[1:], pd.concat([scores, scores.iloc[[0]]], ignore_index=True)]:
            with self.subTest(rows=len(malformed)), self.assertRaises(ValueError):
                study.choose_inner_configs(malformed)

    def test_outer_hac_and_en_controls_use_same_number_of_ic_ranked_factors(self):
        panel = example_panel(["2021-01-04", "2021-04-05", "2021-07-05", "2021-10-04"])
        stats = study.selector_statistics(panel)
        wanted = {"ALL": "all20", "IC": "IC_k10", "HAC": "HAC_k5", "EN": "EN_alpha0.001"}
        tuned = {family: next(candidate for candidate in study.SELECTOR_CANDIDATES
                              if candidate["selector_id"] == selector_id)
                 for family, selector_id in wanted.items()}
        selected_features = [FACTORS[0], FACTORS[5], FACTORS[6]]
        fake_en = {"status": "eligible", "reason": "", "features": selected_features,
                   "coefficients": {factor: float(factor in selected_features) for factor in FACTORS},
                   "nonzero_count": 3, "n_iter": 1, "dual_gap": 0., "scalers": {}}
        with patch.object(study, "fit_elastic_selector", return_value=fake_en):
            arms = study.build_outer_arms("Ridge", tuned, panel, stats)
        by_name = {arm["arm_id"]: arm for arm in arms}
        for family in ["HAC", "EN"]:
            main = next(arm for arm in arms if arm["family"] == family and arm["primary"])
            control = next(arm for name, arm in by_name.items() if name.endswith(f"IC_match_{family}"))
            self.assertFalse(control["primary"])
            self.assertEqual(len(control["features"]), len(main["features"]))
            self.assertEqual(control["features"], study.select_ic(stats, len(main["features"])))

    def test_pipeline_selection_excludes_controls_and_uses_family_then_learner_tie_order(self):
        expected_order = [f"{learner}__{family}" for family in ["ALL", "IC", "HAC", "EN"]
                          for learner in ["Ridge", "HGB"]]
        rows = [{"arm_id": arm_id, "primary": True, "status": "eligible", "rank_ic_mean": .1}
                for arm_id in expected_order]
        rows.extend({"arm_id": f"{learner}__IC_match_{family}", "primary": False,
                     "status": "eligible", "rank_ic_mean": .99}
                    for learner in ["Ridge", "HGB"] for family in ["HAC", "EN"])
        remaining = pd.DataFrame(rows).sample(frac=1, random_state=23)
        for expected in expected_order:
            with self.subTest(expected=expected):
                self.assertEqual(study.choose_pipeline(remaining).arm_id, expected)
                remaining = remaining.loc[remaining.arm_id.ne(expected)]
        with self.assertRaises(ValueError):
            study.choose_pipeline(remaining)

    def test_frozen_protocol_rejects_changed_source_or_parameters(self):
        protocol = {"code_sha256": {"factor_selection_study.py": "source"},
                    "parameters_sha256": "parameters"}
        digest = object_hash(protocol)
        with TemporaryDirectory() as tmp:
            out = Path(tmp)
            write_json(out / "protocol.json", {"protocol": protocol, "protocol_sha256": digest})
            study.verify_existing_protocol(out, protocol)
            for changed in [{**protocol, "parameters_sha256": "changed"},
                            {**protocol, "code_sha256": {"factor_selection_study.py": "changed"}}]:
                with self.subTest(changed=changed), self.assertRaises(ValueError):
                    study.verify_existing_protocol(out, changed)

    def test_completed_stage_rejects_tampered_chosen_configuration_evidence(self):
        with TemporaryDirectory() as tmp:
            out = Path(tmp)
            stage_dir = out / "inner"
            stage_dir.mkdir()
            chosen_path = stage_dir / "chosen.json"
            write_json(chosen_path, {"Ridge": {"IC": "IC_k5"}})
            manifest = {"status": "completed", "protocol_sha256": "frozen",
                        "inputs_sha256": {},
                        "outputs_sha256": {"inner/chosen.json": sha256(chosen_path)}}
            write_json(stage_dir / "run_manifest.json", manifest)
            self.assertEqual(study.verify_stage(out, "inner", "frozen"), manifest)
            with self.assertRaises(ValueError):
                study.verify_stage(out, "inner", "changed")
            chosen_path.write_text(json.dumps({"Ridge": {"IC": "IC_k15"}}))
            with self.assertRaises(ValueError):
                study.verify_stage(out, "inner", "frozen")

    def test_validation_freeze_rejects_changed_upstream_tuning_manifest(self):
        with TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / "tune").mkdir()
            (out / "validation").mkdir()
            tune_path = out / "tune" / "run_manifest.json"
            tune_manifest = {"status": "completed", "protocol_sha256": "frozen",
                             "inputs_sha256": {}, "outputs_sha256": {}}
            write_json(tune_path, tune_manifest)
            validation_manifest = {
                "status": "completed", "protocol_sha256": "frozen",
                "inputs_sha256": {"tune_manifest": sha256(tune_path)}, "outputs_sha256": {},
            }
            write_json(out / "validation" / "run_manifest.json", validation_manifest)
            self.assertEqual(study.verify_stage(out, "validation", "frozen"), validation_manifest)
            tune_path.write_text(json.dumps({**tune_manifest, "stage": "changed"}))
            with self.assertRaises(ValueError):
                study.verify_stage(out, "validation", "frozen")


if __name__ == "__main__":
    unittest.main()
