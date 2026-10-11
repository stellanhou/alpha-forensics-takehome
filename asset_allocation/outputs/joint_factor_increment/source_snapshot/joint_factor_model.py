"""Frozen direct increment: industry/size versus industry/size plus 20 factors.

Annual training and complete signal membership follow portfolio_model_comparison.
The caller freezes the study protocol before fitting. No tuning or selection is
performed; 2023 is descriptive and 2024 is the primary inspected-history period.
"""
from __future__ import annotations

import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from compare_models import (
    FACTORS, KEYS, error_metrics, jsonable, key_hash, moving_block_interval,
    paired_mean_difference, sha256, temporal_split, write_json,
)
from industry_size_attribution import (
    LABEL_ATOL, make_control_model, merge_risk, prediction_diagnostics,
)
from portfolio_model_comparison import COMPRESSION, select_complete_signals, verify_archived
from temporal_model import load_signal_panel


MODELS = ["Ridge", "HGB"]
FEATURES = ["industry", "log_market_cap"] + FACTORS
ARMS = [f"{name}_{variant}" for name in MODELS for variant in ["CONTROLS", "JOINT"]]
STAGES = [(2023, "validation"), (2024, "confirmation")]
METRICS = ["rank_ic", "daily_rmse", "gross_decile_spread"]


def make_joint_model(name: str):
    """Use the existing learner and control encoding, adding PIT factors only."""
    model = make_control_model(name)
    model.set_params(preprocessing=ColumnTransformer([
        ("industry", OneHotEncoder(handle_unknown="error", sparse_output=False,
                                   dtype="float64"), ["industry"]),
        ("size", "passthrough", ["log_market_cap"]),
        ("factors", SimpleImputer(strategy="median"), FACTORS),
    ], remainder="drop", sparse_threshold=0))
    return model


def _fit_metadata(model, training: pd.DataFrame, year: int, arm: str) -> dict:
    transform = model.named_steps["preprocessing"]
    metadata = {
        "year": year, "predictor": arm, "training_rows": len(training),
        "training_keys_sha256": key_hash(training),
        "max_training_label_exit": str(training.exit_date.max().date()),
        "training_mean": float(training.forward_5d.mean()),
        "input_features": model.feature_names_in_.tolist(),
        "feature_names": transform.get_feature_names_out().tolist(),
        "industry_categories": transform.named_transformers_["industry"].categories_[0].tolist(),
        "parameters": jsonable(model.get_params(deep=True)),
        "preprocessing_fit_scope": "training only; full industry dummies and natural log market_cap; "
                                   "JOINT adds all 20 PIT factors with training-median imputation",
    }
    if arm.endswith("_JOINT"):
        medians = transform.named_transformers_["factors"].statistics_
        if not np.isfinite(medians).all():
            raise AssertionError("A joint training factor lacks a finite median")
        metadata["factor_imputation_medians"] = dict(zip(FACTORS, medians.tolist()))
    if arm.startswith("Ridge_"):
        scaler = model.named_steps["scaler"]
        metadata.update(x_scaler_mean=scaler.mean_.tolist(),
                        x_scaler_scale=scaler.scale_.tolist(),
                        x_scaler_var=scaler.var_.tolist())
    else:
        estimator = model.named_steps["model"]
        if estimator.n_iter_ != 55 or estimator.do_early_stopping_:
            raise AssertionError("HGB iteration or early-stopping contract changed")
        metadata.update(fitted_iterations=int(estimator.n_iter_),
                        early_stopping_used=bool(estimator.do_early_stopping_))
    return metadata


def _check_source(portfolio_source: Path) -> pd.DataFrame:
    manifest = json.loads((portfolio_source / "run_manifest.json").read_text())
    if manifest["status"] != "completed":
        raise ValueError("Control model source has not completed")
    paths = ["fits.csv", "protocol.json"]
    paths += [f"predictions/{name}_CONTROLS.csv.gz" for name in MODELS]
    paths += [f"models/{year}_{name}_CONTROLS.joblib" for year, _ in STAGES for name in MODELS]
    for relative in paths:
        if sha256(portfolio_source / relative) != manifest["outputs_sha256"][relative]:
            raise ValueError(f"Changed archived control input: {relative}")
    return pd.read_csv(portfolio_source / "fits.csv")


def evaluation_predictions(result: pd.DataFrame, evaluation: pd.DataFrame) -> pd.DataFrame:
    """Restrict diagnostics to canonical outcomes without changing trading signals."""
    result.set_index(KEYS, verify_integrity=True)
    if result[KEYS].isna().any().any() or not np.isfinite(result.prediction).all():
        raise ValueError("Missing prediction key or nonfinite score")
    labels = ["entry_date", "exit_date", "forward_5d"]
    joined = evaluation[KEYS + labels].merge(
        result[KEYS + labels + ["prediction"]], on=KEYS, how="left",
        suffixes=("", "_predicted"), sort=False, validate="one_to_one",
    )
    if joined.prediction.isna().any():
        raise ValueError("Predictions omit canonical evaluation keys")
    for column in ["entry_date", "exit_date"]:
        if not joined[column].eq(joined[f"{column}_predicted"]).all():
            raise ValueError(f"Prediction {column} differs from canonical evaluation")
    if not np.allclose(joined.forward_5d, joined.forward_5d_predicted,
                       rtol=0, atol=LABEL_ATOL):
        raise ValueError("Prediction labels differ from canonical evaluation")
    return joined[KEYS + labels + ["prediction"]]


def fit_joint_predictions(root: Path, panel_path: Path, portfolio_source: Path,
                          out: Path) -> dict[str, pd.DataFrame]:
    """Fit four joint models and verify/reuse the four archived control models."""
    source_fits = _check_source(portfolio_source)
    canonical = pd.read_csv(panel_path, parse_dates=["date", "entry_date", "exit_date"])
    full = load_signal_panel(root)
    risk = pd.read_csv(root / "data" / "risk_exposure.csv.gz", parse_dates=["date"])
    canonical_risk, full_risk = merge_risk(canonical, risk), merge_risk(full, risk)
    for folder in ["models", "predictions", "splits"]:
        (out / folder).mkdir()
    archives = {name: pd.read_csv(portfolio_source / "predictions" / f"{name}_CONTROLS.csv.gz",
                                 parse_dates=["date", "entry_date", "exit_date", "train_cutoff"])
                for name in MODELS}
    parts = {arm: [] for arm in ARMS}
    fits, coverage = [], []
    for year, stage in STAGES:
        training, _, split = temporal_split(canonical, stage)
        risk_training, _, risk_split = temporal_split(canonical_risk, stage)
        if risk_split != split:
            raise AssertionError("Joint and control training samples differ")
        signals = select_complete_signals(full_risk, year)
        if not training.exit_date.max() < signals.date.min():
            raise AssertionError("Training outcomes cross the first complete signal date")
        write_json(out / "splits" / f"{stage}.json", split)
        for name in MODELS:
            control_arm = name + "_CONTROLS"
            source_fit = source_fits.loc[(source_fits.year == year) &
                                         (source_fits.predictor == control_arm)]
            if len(source_fit) != 1:
                raise ValueError("Missing or duplicate archived annual control fit")
            source_fit = source_fit.iloc[0]
            if (source_fit.training_rows != len(training)
                    or source_fit.training_keys_sha256 != key_hash(training)
                    or source_fit.max_training_label_exit != str(training.exit_date.max().date())):
                raise AssertionError("Training differs from the archived control fit")
            for variant in ["CONTROLS", "JOINT"]:
                arm = name + "_" + variant
                tick = time.perf_counter()
                if variant == "CONTROLS":
                    source_model = portfolio_source / "models" / f"{year}_{arm}.joblib"
                    model = joblib.load(source_model)
                    if jsonable(model.get_params(deep=True)) != jsonable(make_control_model(name).get_params(deep=True)):
                        raise AssertionError("Archived control parameters changed")
                    scores = model.predict(signals[["industry", "log_market_cap"]])
                else:
                    print(f"Fitting {year} {arm}: {len(training)} mature rows -> {len(signals)} full signals", flush=True)
                    model = make_joint_model(name)
                    model.fit(risk_training[FEATURES], training.forward_5d)
                    scores = model.predict(signals[FEATURES])
                if not np.isfinite(scores).all():
                    raise AssertionError("Nonfinite complete-universe prediction")
                result = signals[KEYS + ["entry_date", "exit_date", "forward_5d"]].assign(
                    prediction=scores, train_cutoff=pd.Timestamp(year, 1, 1) - pd.offsets.BDay(1),
                )
                metadata = _fit_metadata(model, training, year, arm)
                metadata["fit_predict_seconds"] = time.perf_counter() - tick
                if variant == "CONTROLS":
                    archive = archives[name].loc[archives[name].date.dt.year == year].reset_index(drop=True)
                    if not result[KEYS].equals(archive[KEYS]):
                        raise AssertionError("Complete member universe differs from archived controls")
                    difference = verify_archived(result, archive)
                    for column in ["entry_date", "exit_date", "train_cutoff"]:
                        if not result[column].eq(archive[column]).all():
                            raise AssertionError(f"Control archive {column} mismatch")
                    np.testing.assert_allclose(result.forward_5d, archive.forward_5d,
                                               rtol=0, atol=LABEL_ATOL, equal_nan=True)
                    metadata.update(source_model=str(source_model.resolve()),
                                    source_model_sha256=sha256(source_model),
                                    max_archived_prediction_difference=difference,
                                    action="verified archived control; no refit")
                    result["prediction"] = archive.prediction.to_numpy()
                else:
                    with (out / "models" / f"{year}_{arm}.joblib").open("xb") as handle:
                        joblib.dump(model, handle)
                    metadata["action"] = "annual joint fit; no tuning"
                write_json(out / "models" / f"{year}_{arm}_fit_metadata.json", metadata)
                parts[arm].append(result)
                fits.append({key: metadata[key] for key in [
                    "year", "predictor", "training_rows", "training_keys_sha256",
                    "max_training_label_exit", "fit_predict_seconds", "action"]})
                coverage.append({"year": year, "predictor": arm, "signal_rows": len(result),
                                 "signal_dates": result.date.nunique(),
                                 "missing_labels": int(result.forward_5d.isna().sum()),
                                 "signal_keys_sha256": key_hash(result)})
    expected = full.loc[full.date.dt.year.between(2023, 2024), KEYS].reset_index(drop=True)
    predictions = {}
    for arm, frames in parts.items():
        result = pd.concat(frames, ignore_index=True).sort_values(KEYS).reset_index(drop=True)
        if not result[KEYS].equals(expected):
            raise AssertionError("Incomplete joint-study signal coverage")
        result.to_csv(out / "predictions" / f"{arm}.csv.gz", mode="x", index=False, compression=COMPRESSION)
        predictions[arm] = result
    pd.DataFrame(fits).to_csv(out / "fits.csv", mode="x", index=False)
    pd.DataFrame(coverage).to_csv(out / "signal_coverage.csv", mode="x", index=False)
    return predictions


def evaluate_predictions(predictions: dict[str, pd.DataFrame], panel_path: Path,
                         out: Path) -> dict[str, pd.DataFrame]:
    """Paired daily metrics on the original canonical temporal evaluation rows."""
    panel = pd.read_csv(panel_path, parse_dates=["date", "entry_date", "exit_date"])
    destination = out / "prediction_evaluation"
    destination.mkdir()
    all_summaries, all_comparisons, all_daily = [], [], []
    for year, stage in STAGES:
        training, evaluation, split = temporal_split(panel, stage)
        stage_out = destination / stage
        stage_out.mkdir()
        write_json(stage_out / "split_manifest.json", split)
        summaries, comparisons, daily_parts = [], [], []
        for name in MODELS:
            by_variant = {}
            for variant in ["CONTROLS", "JOINT"]:
                arm = name + "_" + variant
                result = evaluation_predictions(predictions[arm], evaluation)
                daily = prediction_diagnostics(result)
                indexed = daily.set_index("date")
                by_variant[variant] = indexed
                row = {"year": year, "stage": stage, "model": name, "variant": variant,
                       "predictor": arm, **error_metrics(result, float(training.forward_5d.mean()))}
                for metric in METRICS:
                    row.update({f"{metric}_{key}": value for key, value in
                                moving_block_interval(indexed[metric]).items()})
                summaries.append(row)
                daily_parts.append(daily.assign(year=year, stage=stage, model=name,
                                                variant=variant, predictor=arm))
            for metric in METRICS:
                joint, controls = by_variant["JOINT"][metric], by_variant["CONTROLS"][metric]
                common = joint.notna() & controls.notna()
                comparisons.append({"year": year, "stage": stage, "model": name, "metric": metric,
                                    "comparison": "JOINT minus CONTROLS",
                                    "joint_on_common_mean": float(joint[common].mean()),
                                    "controls_on_common_mean": float(controls[common].mean()),
                                    **paired_mean_difference(joint, controls)})
        tables = {"summary": pd.DataFrame(summaries), "paired_comparison": pd.DataFrame(comparisons),
                  "daily_metrics": pd.concat(daily_parts, ignore_index=True)}
        for label, frame in tables.items():
            frame.to_csv(stage_out / f"{label}.csv", mode="x", index=False)
        all_summaries.append(tables["summary"])
        all_comparisons.append(tables["paired_comparison"])
        all_daily.append(tables["daily_metrics"])
    combined = {"summary": pd.concat(all_summaries, ignore_index=True),
                "paired_comparison": pd.concat(all_comparisons, ignore_index=True),
                "daily_metrics": pd.concat(all_daily, ignore_index=True)}
    for label, frame in combined.items():
        frame.to_csv(destination / f"{label}.csv", mode="x", index=False)
    return combined
