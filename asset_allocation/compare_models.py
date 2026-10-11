"""Frozen CPU model comparison on the previously inspected PIT research panel.

Run validation first, then confirmation in the SAME output directory. Neither
stage overwrites previous artifacts. A changed protocol requires a new directory.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay
from sklearn.base import BaseEstimator
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT = Path(__file__).resolve().parent
FACTORS = [f"factor_{i:02d}" for i in range(1, 21)]
KEYS = ["date", "asset_id"]
SEED = 1021
THREADS = 4
BLOCK_DAYS = 20
BOOTSTRAP_DRAWS = 2000
PHASE_ORIGIN = "2021-01-01"
CANDIDATES = ["HGB", "Ridge", "ElasticNet", "ExtraTrees", "LightGBM", "XGBoost", "CatBoost"]
SCOPE = (
    "Historical temporal holdout on a previously inspected full-period dataset; "
    "2024 is not an untouched new sample. The 2023-selected model is the sole "
    "prespecified confirmation candidate; other 2024 models are descriptive. "
    "Five-day decile spreads are gross grouping diagnostics, not executable "
    "portfolio returns; no fees, NAV or Sharpe are inferred."
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def jsonable(value):
    """Represent complete estimator parameters without unstable object reprs."""
    # CatBoost implements get_params but does not inherit sklearn BaseEstimator.
    catboost_regressor = (type(value).__module__ == "catboost.core"
                          and type(value).__name__ == "CatBoostRegressor")
    if isinstance(value, BaseEstimator) or catboost_regressor:
        return {"class": f"{type(value).__module__}.{type(value).__name__}",
                "params": jsonable(value.get_params(deep=False))}
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Unsupported parameter type: {type(value)}")


def object_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def write_json(path: Path, value) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def key_hash(frame: pd.DataFrame) -> str:
    return hashlib.sha256(frame[KEYS].to_csv(index=False).encode()).hexdigest()


def model_pipeline(estimator, *, linear=False) -> Pipeline:
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if linear:
        steps.append(("scaler", StandardScaler()))
    return Pipeline(steps + [("model", estimator)])


def make_models() -> dict[str, Pipeline]:
    # Imports are delayed so the contract tests need only numpy/pandas/sklearn.
    from catboost import CatBoostRegressor
    from lightgbm import LGBMRegressor
    from xgboost import XGBRegressor

    return {
        "HGB": model_pipeline(HistGradientBoostingRegressor(
            max_iter=55, learning_rate=.065, max_leaf_nodes=17,
            l2_regularization=.2, early_stopping=False, random_state=SEED)),
        "Ridge": model_pipeline(Ridge(alpha=100), linear=True),
        "ElasticNet": model_pipeline(TransformedTargetRegressor(
            regressor=ElasticNet(alpha=.001, l1_ratio=.5, max_iter=5000,
                                 tol=1e-5, random_state=SEED),
            transformer=StandardScaler()), linear=True),
        "ExtraTrees": model_pipeline(ExtraTreesRegressor(
            n_estimators=64, max_depth=8, min_samples_leaf=100,
            max_features=1.0, n_jobs=THREADS, random_state=SEED)),
        "LightGBM": model_pipeline(LGBMRegressor(
            n_estimators=160, learning_rate=.05, num_leaves=17,
            min_child_samples=100, reg_lambda=1, n_jobs=THREADS,
            deterministic=True, force_col_wise=True, device_type="cpu",
            random_state=SEED, verbosity=-1)),
        "XGBoost": model_pipeline(XGBRegressor(
            n_estimators=160, learning_rate=.05, max_depth=4,
            min_child_weight=100, reg_lambda=1, tree_method="hist",
            device="cpu", n_jobs=THREADS, random_state=SEED)),
        "CatBoost": model_pipeline(CatBoostRegressor(
            iterations=180, learning_rate=.05, depth=6, l2_leaf_reg=3,
            task_type="CPU", thread_count=THREADS, random_seed=SEED,
            allow_writing_files=False, verbose=False)),
        "Dummy": model_pipeline(DummyRegressor(strategy="mean")),
    }


def validate_panel(frame: pd.DataFrame) -> None:
    frame.set_index(KEYS, verify_integrity=True)
    if frame[KEYS + ["entry_date", "exit_date"]].isna().any().any():
        raise ValueError("Missing keys or label endpoint dates")
    if not np.is_busday(frame.date.to_numpy(dtype="datetime64[D]")).all():
        raise ValueError("Signal date is outside the synthetic business-day calendar")
    for column, offset in [("entry_date", 1), ("exit_date", 6)]:
        if not frame[column].eq(frame.date + BDay(offset)).all():
            raise ValueError(f"Label {column} does not match the t+1/t+6 contract")
    if np.isinf(frame[FACTORS].to_numpy(dtype=float)).any():
        raise ValueError("Infinite factor value")
    if not np.isfinite(frame.forward_5d).all():
        raise ValueError("Non-finite forward_5d label")


def temporal_split(frame: pd.DataFrame, stage: str):
    validate_panel(frame)
    year = {"validation": 2023, "confirmation": 2024}[stage]
    train_candidates = frame.loc[frame.date.dt.year.between(2021, year - 1)]
    eval_candidates = frame.loc[frame.date.dt.year == year]
    evaluation = eval_candidates
    if stage == "validation":
        evaluation = evaluation.loc[evaluation.exit_date < pd.Timestamp("2024-01-01")]
    if train_candidates.empty or evaluation.empty:
        raise ValueError("Empty training or evaluation period")
    boundary = evaluation.date.min()
    train = train_candidates.loc[train_candidates.exit_date < boundary]
    if train.empty or train[FACTORS].isna().all().any():
        raise ValueError("Empty training data or entirely missing training factor")
    if not train.exit_date.max() < boundary or not train.date.max() < boundary:
        raise ValueError("Training label period overlaps evaluation")
    if not train.set_index(KEYS).index.intersection(evaluation.set_index(KEYS).index).empty:
        raise ValueError("Training and evaluation keys overlap")
    train = train.sort_values(KEYS).reset_index(drop=True)
    evaluation = evaluation.sort_values(KEYS).reset_index(drop=True)

    def describe(part):
        return {
            "rows": len(part), "signal_dates": int(part.date.nunique()),
            "signal_start": str(part.date.min().date()),
            "signal_end": str(part.date.max().date()),
            "min_label_entry": str(part.entry_date.min().date()),
            "max_label_exit": str(part.exit_date.max().date()),
            "ordered_keys_sha256": key_hash(part),
            "factor_missing_rows": part[FACTORS].isna().sum().astype(int).to_dict(),
        }

    manifest = {
        "stage": stage, "train": describe(train), "evaluation": describe(evaluation),
        "train_candidates": len(train_candidates), "eval_candidates": len(eval_candidates),
        "purged_training_rows": len(train_candidates) - len(train),
        "purged_evaluation_rows": len(eval_candidates) - len(evaluation),
        "training_exit_strictly_before_first_evaluation_signal": True,
        "validation_exit_strictly_before_2024": stage == "validation",
    }
    return train, evaluation, manifest


def safe_correlation(left, right) -> float:
    left, right = np.asarray(left), np.asarray(right)
    if len(left) < 2 or np.ptp(left) == 0 or np.ptp(right) == 0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def decile_spread(scores, returns) -> float:
    """Equal inclusion at tied cutoffs; neither asset IDs nor row order enter."""
    scores, returns = np.asarray(scores), np.asarray(returns)
    if np.ptp(scores) == 0:
        return float("nan")
    levels, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    del levels
    tail_size = len(scores) * .1
    below = np.cumsum(counts) - counts
    above = len(scores) - np.cumsum(counts)
    low_weights = np.clip(tail_size - below, 0, counts) / counts
    high_weights = np.clip(tail_size - above, 0, counts) / counts
    return float(np.dot(high_weights[inverse] - low_weights[inverse], returns) / tail_size)


def daily_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    predictions.set_index(KEYS, verify_integrity=True)
    if not np.isfinite(predictions[["prediction", "forward_5d"]]).all().all():
        raise ValueError("Non-finite prediction or label")
    records = []
    for date, part in predictions.groupby("date", sort=True):
        records.append({
            "date": date, "n_assets": len(part),
            "rank_ic": safe_correlation(part.prediction.rank(method="average"),
                                        part.forward_5d.rank(method="average")),
            "gross_decile_spread": decile_spread(part.prediction, part.forward_5d),
        })
    return pd.DataFrame(records)


def phase_numbers(dates) -> np.ndarray:
    return np.busday_count(np.datetime64(PHASE_ORIGIN),
                           np.asarray(dates, dtype="datetime64[D]")) % 5


def moving_block_interval(series: pd.Series, *, seed=SEED,
                          draws=BOOTSTRAP_DRAWS) -> dict:
    """Resample 20 consecutive synthetic trading dates, keeping missing dates."""
    if not isinstance(series.index, pd.DatetimeIndex) or series.index.has_duplicates:
        raise ValueError("Bootstrap requires unique datetime keys")
    series = series.sort_index()
    count = int(series.notna().sum())
    if count == 0:
        return {"n_dates": 0, "mean": np.nan, "ci_low": np.nan, "ci_high": np.nan}
    calendar = pd.bdate_range(series.index.min(), series.index.max())
    if len(calendar) < BLOCK_DAYS:
        raise ValueError("Fewer than 20 trading dates for the fixed block bootstrap")
    values = series.reindex(calendar).to_numpy(dtype=float)
    if np.isinf(values).any():
        raise ValueError("Infinite bootstrap observation")
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(values) - BLOCK_DAYS + 1,
                          size=(draws, math.ceil(len(values) / BLOCK_DAYS)))
    indices = (starts[..., None] + np.arange(BLOCK_DAYS)).reshape(draws, -1)[:, :len(values)]
    sampled = values[indices]
    valid_counts = np.isfinite(sampled).sum(axis=1)
    if (valid_counts == 0).any():
        raise ValueError("Bootstrap draw has no defined metric observations")
    means = np.nansum(sampled, axis=1) / valid_counts
    low, high = np.quantile(means, [.025, .975])
    return {"n_dates": count, "mean": float(series.mean()),
            "ci_low": float(low), "ci_high": float(high)}


def paired_mean_difference(candidate: pd.Series, reference: pd.Series, **kwargs) -> dict:
    if candidate.index.has_duplicates or reference.index.has_duplicates:
        raise ValueError("Paired comparison requires unique dates")
    paired = pd.concat([candidate.rename("candidate"), reference.rename("reference")],
                       axis=1, join="inner")
    # Keep undefined joint dates as NaN on the calendar instead of compressing time.
    return moving_block_interval(paired.candidate - paired.reference, **kwargs)


def error_metrics(predictions: pd.DataFrame, training_mean: float) -> dict:
    y, pred = predictions.forward_5d.to_numpy(), predictions.prediction.to_numpy()
    mse = float(np.mean((y - pred) ** 2))
    baseline_mse = float(np.mean((y - training_mean) ** 2))
    if baseline_mse == 0:
        raise ValueError("Training-mean baseline has zero evaluation error")
    return {"rows": len(predictions), "pearson": safe_correlation(pred, y),
            "rmse": math.sqrt(mse), "training_mean_baseline_rmse": math.sqrt(baseline_mse),
            "r2_against_training_mean": 1 - mse / baseline_mse}


def summarize_model(name, predictions, daily, reference, training_mean):
    indexed = daily.set_index("date")
    rank = moving_block_interval(indexed.rank_ic)
    spread = moving_block_interval(indexed.gross_decile_spread)
    paired = paired_mean_difference(indexed.rank_ic, reference)
    summary = {"model": name, **error_metrics(predictions, training_mean)}
    for label, result in [("rank_ic", rank), ("gross_decile_spread", spread),
                          ("rank_ic_minus_HGB", paired)]:
        summary.update({f"{label}_{key}": value for key, value in result.items()})
    quarterly = []
    for quarter, part in predictions.groupby(predictions.date.dt.to_period("Q"), sort=True):
        days = daily.loc[daily.date.dt.to_period("Q") == quarter]
        quarterly.append({"model": name, "quarter": str(quarter),
                          **error_metrics(part, training_mean),
                          "rank_ic_mean": days.rank_ic.mean(),
                          "rank_ic_valid_dates": int(days.rank_ic.notna().sum()),
                          "gross_decile_spread_mean": days.gross_decile_spread.mean()})
    phase_table = daily.assign(phase=phase_numbers(daily.date))
    phases = []
    for phase in range(5):
        part = phase_table.loc[phase_table.phase == phase]
        phases.append({"model": name, "phase": phase, "signal_dates": len(part),
                       "rank_ic_valid_dates": int(part.rank_ic.notna().sum()),
                       "rank_ic_mean": part.rank_ic.mean(),
                       "gross_decile_spread_mean": part.gross_decile_spread.mean()})
    return summary, quarterly, phases


def build_protocol(panel_path: Path, models: dict) -> dict:
    timing_manifest = panel_path.parent / "run_manifest.json"
    timing = json.loads(timing_manifest.read_text())
    panel_hash = sha256(panel_path)
    if panel_hash != timing["outputs_sha256"][panel_path.name]:
        raise ValueError("PIT panel differs from the factor-timing manifest")
    parameters = {name: jsonable(model.get_params(deep=True)) for name, model in models.items()}
    return {
        "format_version": 1, "scope": SCOPE, "seed": SEED, "cpu_threads": THREADS,
        "factor_columns": FACTORS, "candidate_order": CANDIDATES,
        "model_parameters": parameters, "parameters_sha256": object_hash(parameters),
        "code_sha256": {path.name: sha256(path) for path in [
            Path(__file__), PROJECT / "test_compare_models.py"]},
        "panel_path": str(panel_path.resolve()),
        "inputs_sha256": {"pit_panel": panel_hash, "timing_manifest": sha256(timing_manifest)},
        "versions": {"python": platform.python_version(), **{
            name: importlib.metadata.version(name)
            for name in ["numpy", "pandas", "scikit-learn", "lightgbm", "xgboost", "catboost"]}},
        "threads_environment": {name: os.environ.get(name) for name in [
            "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]},
        "label": "open(t+6 business days) / open(t+1 business day) - 1",
        "validation": {"training_signal_years": [2021, 2022], "evaluation_signal_year": 2023,
                       "evaluation_exit_before": "2024-01-01"},
        "confirmation": {"training_signal_years": [2021, 2022, 2023], "evaluation_signal_year": 2024},
        "training_purge": "exit_date strictly before the first evaluation signal date",
        "fitting": "All training rows; training-only median imputation; no tuning, early stopping or eval_set",
        "selection": "Highest defined 2023 mean daily RankIC; undefined candidates explicitly ineligible; exact ties use candidate_order",
        "deciles": "Each tail has n*0.1 weight; all tied cutoff scores share equal fractional inclusion; constant scores undefined",
        "phases": {"calendar": "Monday-Friday synthetic trading dates", "origin": PHASE_ORIGIN,
                   "assignment": "busday_count(origin, signal_date) % 5"},
        "bootstrap": {"kind": "noncircular moving blocks on complete business-day calendar",
                      "block_days": BLOCK_DAYS, "draws": BOOTSTRAP_DRAWS,
                      "seed": SEED, "interval": "2.5/97.5 percentiles",
                      "paired": "candidate minus HGB on the same defined dates, resampled jointly"},
    }


def verify_confirmation(out: Path, protocol: dict) -> dict:
    frozen = json.loads((out / "protocol.json").read_text())
    digest = object_hash(protocol)
    if frozen["protocol_sha256"] != digest or object_hash(frozen["protocol"]) != digest:
        raise ValueError("Frozen protocol, parameters, versions, inputs or code changed; use a new output directory")
    completed = json.loads((out / "validation" / "run_manifest.json").read_text())
    if completed["status"] != "completed" or completed["protocol_sha256"] != digest:
        raise ValueError("Validation is not complete under this frozen protocol")
    for relative, expected in completed["outputs_sha256"].items():
        if sha256(out / relative) != expected:
            raise ValueError(f"Validation output changed: {relative}")
    selection = json.loads((out / "selection.json").read_text())
    if (selection["protocol_sha256"] != digest
            or selection["parameters_sha256"] != protocol["parameters_sha256"]
            or selection["code_sha256"] != protocol["code_sha256"]
            or selection["selected_model"] not in CANDIDATES):
        raise ValueError("Selection does not match the frozen protocol")
    return selection


def select_candidate(summaries: list[dict]) -> tuple[str, dict, dict]:
    scores = {row["model"]: row["rank_ic_mean"] for row in summaries if row["model"] in CANDIDATES}
    if set(scores) != set(CANDIDATES):
        raise ValueError("Validation results do not contain every prespecified candidate")
    eligible = [name for name in CANDIDATES if np.isfinite(scores[name])]
    ineligible = {name: "No defined daily RankIC; ineligible for model selection"
                  for name in CANDIDATES if name not in eligible}
    if not eligible:
        raise ValueError("Every candidate has undefined validation RankIC; no valid selection")
    chosen = max(eligible, key=lambda name: scores[name])
    reported_scores = {name: float(scores[name]) if name in eligible else None for name in CANDIDATES}
    return chosen, reported_scores, ineligible


def run(stage: str, panel_path: Path, out: Path) -> None:
    started = utc_now()
    models = make_models()
    protocol = build_protocol(panel_path, models)
    protocol_digest = object_hash(protocol)
    if stage == "validation":
        out.mkdir(parents=True, exist_ok=True)
        write_json(out / "protocol.json", {"frozen_at_utc": utc_now(),
                   "protocol_sha256": protocol_digest, "protocol": protocol})
        selection = None
    else:
        selection = verify_confirmation(out, protocol)
    stage_out = out / stage
    stage_out.mkdir()  # Existing stages, including failed partial runs, are never overwritten.
    panel = pd.read_csv(panel_path, parse_dates=["date", "entry_date", "exit_date"])
    train, evaluation, split = temporal_split(panel, stage)
    write_json(stage_out / "split_manifest.json", split)
    x, y = train[FACTORS], train.forward_5d
    training_mean = float(y.mean())
    print(f"{stage}: train={len(train)} evaluation={len(evaluation)}; protocol={protocol_digest}", flush=True)
    summaries, quarters, phases, timings, daily_parts = [], [], [], [], []
    reference = None
    for name, model in models.items():
        print(f"{utc_now()} fitting {name}", flush=True)
        tick = time.perf_counter()
        model.fit(x, y)
        fit_seconds = time.perf_counter() - tick
        tick = time.perf_counter()
        prediction = model.predict(evaluation[FACTORS])
        prediction_seconds = time.perf_counter() - tick
        if not np.isfinite(prediction).all():
            raise ValueError(f"{name}: non-finite predictions")
        result = evaluation[KEYS + ["entry_date", "exit_date", "forward_5d"]].assign(prediction=prediction)
        result.to_csv(stage_out / f"{name}_predictions.csv.gz", index=False,
                      compression={"method": "gzip", "compresslevel": 1, "mtime": 0})
        daily = daily_metrics(result)
        if name == "HGB":
            reference = daily.set_index("date").rank_ic
        summary, quarterly, phase_table = summarize_model(name, result, daily, reference, training_mean)
        summary["prespecified_confirmation_candidate"] = bool(selection and name == selection["selected_model"])
        summaries.append(summary)
        quarters.extend(quarterly)
        phases.extend(phase_table)
        daily_parts.append(daily.assign(model=name))
        timings.append({"model": name, "fit_seconds": fit_seconds,
                        "prediction_seconds": prediction_seconds})
        fitted = {"training_mean": training_mean,
                  "imputer_medians": model.named_steps["imputer"].statistics_.tolist()}
        if "scaler" in model.named_steps:
            fitted["x_scaler_mean"] = model.named_steps["scaler"].mean_.tolist()
            fitted["x_scaler_scale"] = model.named_steps["scaler"].scale_.tolist()
        if name == "ElasticNet":
            fitted["y_scaler_mean"] = model.named_steps["model"].transformer_.mean_.tolist()
            fitted["y_scaler_scale"] = model.named_steps["model"].transformer_.scale_.tolist()
        write_json(stage_out / f"{name}_fit_metadata.json", fitted)
        print(f"{name}: RankIC={summary['rank_ic_mean']:.6f} RMSE={summary['rmse']:.6f} fit={fit_seconds:.2f}s", flush=True)

    pd.DataFrame(summaries).to_csv(stage_out / "summary.csv", index=False)
    pd.concat(daily_parts, ignore_index=True).to_csv(stage_out / "daily_metrics.csv", index=False)
    pd.DataFrame(quarters).to_csv(stage_out / "quarterly.csv", index=False)
    pd.DataFrame(phases).to_csv(stage_out / "phases.csv", index=False)
    pd.DataFrame(timings).to_csv(stage_out / "timing.csv", index=False)
    if stage == "validation":
        chosen, candidates, ineligible = select_candidate(summaries)
        selection = {"frozen_at_utc": utc_now(), "selected_model": chosen,
                     "selection_stage": "2023 validation only", "validation_mean_daily_rank_ic": candidates,
                     "ineligible_candidates": ineligible,
                     "protocol_sha256": protocol_digest, "parameters_sha256": protocol["parameters_sha256"],
                     "code_sha256": protocol["code_sha256"],
                     "selected_parameters": protocol["model_parameters"][chosen],
                     "validation_summary_sha256": sha256(stage_out / "summary.csv"), "scope": SCOPE}
        write_json(out / "selection.json", selection)
    paths = sorted(stage_out.iterdir()) + [out / "selection.json", out / "protocol.json"]
    write_json(stage_out / "run_manifest.json", {
        "status": "completed", "stage": stage, "started_at_utc": started, "finished_at_utc": utc_now(),
        "command": sys.orig_argv, "scope": SCOPE, "protocol_sha256": protocol_digest,
        "selected_model": selection["selected_model"], "code_sha256": protocol["code_sha256"],
        "inputs_sha256": protocol["inputs_sha256"],
        "outputs_sha256": {str(path.relative_to(out)): sha256(path) for path in paths},
    })
    print(f"{stage} completed; prespecified confirmation candidate={selection['selected_model']}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["validation", "confirmation"], required=True)
    parser.add_argument("--panel", type=Path,
                        default=PROJECT / "outputs" / "factor_timing" / "pit_research_panel.csv.gz")
    parser.add_argument("--out", type=Path, default=PROJECT / "outputs" / "model_benchmark")
    args = parser.parse_args()
    run(args.stage, args.panel, args.out)
