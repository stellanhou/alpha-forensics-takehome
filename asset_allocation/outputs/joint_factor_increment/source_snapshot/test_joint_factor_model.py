"""Contracts for the authorized nested feature experiment and same-sample pairs."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd

from compare_models import FACTORS, KEYS, key_hash, sha256, temporal_split
from industry_size_attribution import make_control_model, merge_risk
import joint_factor_model as study


def small_panel():
    dates = (pd.bdate_range("2021-01-04", periods=25).tolist()
             + pd.bdate_range("2022-12-12", "2022-12-30").tolist()
             + pd.bdate_range("2023-01-02", periods=30).tolist()
             + pd.bdate_range("2024-01-01", periods=30).tolist())
    frame = pd.MultiIndex.from_product([dates, ["A", "B", "C", "D", "E", "F"]],
                                       names=KEYS).to_frame(index=False)
    rng = np.random.default_rng(405)
    for factor in FACTORS:
        frame[factor] = rng.normal(size=len(frame))
    frame.loc[frame.index % 13 == 0, "factor_03"] = np.nan
    frame["entry_date"] = frame.date + pd.offsets.BDay(1)
    frame["exit_date"] = frame.date + pd.offsets.BDay(6)
    frame["forward_5d"] = .01 * frame.factor_01 + .005 * rng.normal(size=len(frame))
    # This eligible final member has no observed outcome and must still be scored.
    frame.loc[frame.index[-1], "forward_5d"] = np.nan
    risk = frame[KEYS].assign(industry=np.tile(["I1", "I2", "I1", "I2", "I1", "I2"], len(dates)),
                              market_cap=np.exp(rng.uniform(1, 5, len(frame))))
    return frame, risk


def archive_controls(full, risk, source):
    canonical = full.loc[full.forward_5d.notna()].reset_index(drop=True)
    enriched = merge_risk(full, risk)
    enriched_canonical = merge_risk(canonical, risk)
    (source / "models").mkdir(parents=True)
    (source / "predictions").mkdir()
    (source / "protocol.json").write_text("{}")
    fits, parts = [], {name: [] for name in study.MODELS}
    for year, stage in study.STAGES:
        training, _, _ = temporal_split(enriched_canonical, stage)
        signals = enriched.loc[enriched.date.dt.year == year].reset_index(drop=True)
        for name in study.MODELS:
            model = make_control_model(name).fit(training[["industry", "log_market_cap"]], training.forward_5d)
            joblib.dump(model, source / "models" / f"{year}_{name}_CONTROLS.joblib")
            parts[name].append(signals[KEYS + ["entry_date", "exit_date", "forward_5d"]].assign(
                prediction=model.predict(signals[["industry", "log_market_cap"]]),
                train_cutoff=pd.Timestamp(year, 1, 1) - pd.offsets.BDay(1)))
            fits.append({"year": year, "predictor": name + "_CONTROLS", "training_rows": len(training),
                         "training_keys_sha256": key_hash(training),
                         "max_training_label_exit": str(training.exit_date.max().date())})
    pd.DataFrame(fits).to_csv(source / "fits.csv", index=False)
    for name, frames in parts.items():
        pd.concat(frames, ignore_index=True).to_csv(source / "predictions" / f"{name}_CONTROLS.csv.gz", index=False)
    manifest = {"status": "completed", "outputs_sha256": {
        str(path.relative_to(source)): sha256(path) for path in source.rglob("*") if path.is_file()}}
    (source / "run_manifest.json").write_text(json.dumps(manifest))
    return canonical


class JointFactorModelTests(unittest.TestCase):
    def test_preprocessing_uses_all_factors_and_training_only_statistics(self):
        full, risk = small_panel()
        training = merge_risk(full.iloc[:90], risk).copy()
        evaluation = training.iloc[:3].copy()
        evaluation[FACTORS] = 1e9
        evaluation.loc[evaluation.index[0], FACTORS] = np.nan
        for name in study.MODELS:
            with self.subTest(model=name):
                model = study.make_joint_model(name).fit(training[study.FEATURES], training.forward_5d)
                transform = model.named_steps["preprocessing"]
                imputer = transform.named_transformers_["factors"]
                medians = training[FACTORS].median().to_numpy()
                np.testing.assert_allclose(imputer.statistics_, medians)
                names = transform.get_feature_names_out().tolist()
                self.assertEqual(len(names), 23)
                self.assertEqual(names[-20:], ["factors__" + factor for factor in FACTORS])
                matrix = transform.transform(training[study.FEATURES])
                if name == "Ridge":
                    scaler = model.named_steps["scaler"]
                    np.testing.assert_allclose(scaler.mean_, matrix.mean(axis=0))
                    before_mean = scaler.mean_.copy()
                self.assertTrue(np.isfinite(model.predict(evaluation[study.FEATURES])).all())
                np.testing.assert_allclose(imputer.statistics_, medians)
                if name == "Ridge":
                    np.testing.assert_array_equal(scaler.mean_, before_mean)
                unknown = evaluation[study.FEATURES].copy()
                unknown["industry"] = "new-industry"
                with self.assertRaises(ValueError):
                    model.predict(unknown)

    def test_evaluation_uses_canonical_labels_and_rejects_omitted_or_changed_rows(self):
        full, _ = small_panel()
        _, evaluation, _ = temporal_split(full.dropna(subset=["forward_5d"]), "confirmation")
        predictions = full.loc[full.date.dt.year == 2024,
                               KEYS + ["entry_date", "exit_date", "forward_5d"]].assign(prediction=.01)
        aligned = study.evaluation_predictions(predictions, evaluation)
        self.assertEqual(key_hash(aligned), key_hash(evaluation))
        self.assertEqual(len(predictions) - len(aligned), 1)
        self.assertEqual(predictions.forward_5d.isna().sum(), 1)
        for change in ["missing", "label", "endpoint"]:
            invalid = predictions.copy()
            if change == "missing":
                invalid = invalid.iloc[1:]
            elif change == "label":
                invalid.loc[invalid.index[0], "forward_5d"] += .1
            else:
                invalid.loc[invalid.index[0], "entry_date"] += pd.offsets.BDay(1)
            with self.subTest(change=change), self.assertRaises(ValueError):
                study.evaluation_predictions(invalid, evaluation)

    def test_small_annual_run_reuses_controls_keeps_unknown_outcomes_and_pairs_days(self):
        full, risk = small_panel()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, out = root / "source", root / "result"
            (root / "data").mkdir()
            out.mkdir()
            risk.to_csv(root / "data" / "risk_exposure.csv.gz", index=False)
            canonical = archive_controls(full, risk, source)
            panel_path = root / "panel.csv.gz"
            canonical.to_csv(panel_path, index=False)
            with patch.object(study, "load_signal_panel", return_value=full):
                predictions = study.fit_joint_predictions(root, panel_path, source, out)
            expected = full.loc[full.date.dt.year.isin([2023, 2024])].reset_index(drop=True)
            for arm in study.ARMS:
                self.assertEqual(key_hash(predictions[arm]), key_hash(expected))
                self.assertEqual(predictions[arm].forward_5d.isna().sum(), 1)
                self.assertTrue(np.isfinite(predictions[arm].prediction).all())
            for name in study.MODELS:
                old = pd.read_csv(source / "predictions" / f"{name}_CONTROLS.csv.gz")
                np.testing.assert_array_equal(predictions[name + "_CONTROLS"].prediction, old.prediction)
            self.assertEqual(len(list((out / "models").glob("*.joblib"))), 4)
            metadata = json.loads((out / "models" / "2023_Ridge_JOINT_fit_metadata.json").read_text())
            self.assertLess(pd.Timestamp(metadata["max_training_label_exit"]), pd.Timestamp("2023-01-02"))
            result = study.evaluate_predictions(predictions, panel_path, out)
            self.assertEqual(len(result["summary"]), 8)
            self.assertEqual(len(result["paired_comparison"]), 12)
            paired = result["paired_comparison"]
            np.testing.assert_allclose(paired["mean"], paired.joint_on_common_mean - paired.controls_on_common_mean,
                                       atol=1e-14)
            row = result["summary"].query("year == 2024 and predictor == 'Ridge_JOINT'").iloc[0]
            _, evaluation, _ = temporal_split(canonical, "confirmation")
            checked = study.evaluation_predictions(predictions["Ridge_JOINT"], evaluation)
            self.assertAlmostEqual(row.rmse, np.sqrt(np.mean((checked.prediction - checked.forward_5d) ** 2)))
            with self.assertRaises(FileExistsError):
                study.evaluate_predictions(predictions, panel_path, out)


if __name__ == "__main__":
    unittest.main()
