"""Frozen industry/size diagnostics for the existing ALL20 Ridge and HGB signals.

This is additional historical analysis, not a new untouched confirmation sample.
Only industry/log-market-cap baselines are fitted; archived ALL20 signals are
verified and reused. Cross-sectional return projections are evaluation only.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from compare_models import (
    BLOCK_DAYS, BOOTSTRAP_DRAWS, KEYS, PHASE_ORIGIN, PROJECT, SEED,
    daily_metrics, error_metrics, jsonable, key_hash, moving_block_interval,
    object_hash, paired_mean_difference, phase_numbers, sha256, temporal_split,
    utc_now, write_json,
)


MODELS = ["Ridge", "HGB"]
CONTROL_KINDS = ["industry", "size", "industry_size"]
ZERO_RESIDUAL_RTOL = 1e-10
LABEL_ATOL = 1e-14
DEFAULT_RISK = (PROJECT.parent / "Lab404-CUFEL-E" / "03-资产配置策略方向"
                / "Alpha_Forensics_Candidate_v1" / "data" / "risk_exposure.csv.gz")
SCOPE = (
    "Post-inspection historical diagnostics on the existing 2023/2024 temporal "
    "holdouts, not an independent blind test or a new model-selection round. "
    "Partial RankIC measures residual rank association, not causal attribution "
    "or neutralized portfolio returns. Gross five-day spreads exclude trading "
    "implementation and costs."
)


def validate_risk_values(frame: pd.DataFrame) -> None:
    if frame.industry.isna().any() or frame.industry.astype(str).str.strip().eq("").any():
        raise ValueError("Missing or empty industry")
    caps = frame.market_cap.to_numpy(dtype=float)
    if not np.isfinite(caps).all() or (caps <= 0).any():
        raise ValueError("market_cap must be finite and strictly positive")


def merge_risk(panel: pd.DataFrame, risk: pd.DataFrame) -> pd.DataFrame:
    """Exact same-date one-to-one merge without removing any research row."""
    if set(["industry", "market_cap", "log_market_cap"]).intersection(panel.columns):
        raise ValueError("Research panel already contains risk fields")
    for name, frame in [("panel", panel), ("risk", risk)]:
        if frame[KEYS].isna().any().any():
            raise ValueError(f"Missing {name} key")
        frame.set_index(KEYS, verify_integrity=True)
    selected = risk[KEYS + ["industry", "market_cap"]].copy()
    validate_risk_values(selected)
    result = panel.merge(selected, on=KEYS, how="left", validate="one_to_one",
                         sort=False, indicator=True)
    if not result._merge.eq("both").all():
        raise ValueError("Missing same-date risk exposure for a research-panel key")
    if len(result) != len(panel) or not result[KEYS].reset_index(drop=True).equals(
            panel[KEYS].reset_index(drop=True)):
        raise ValueError("Risk merge changed the research sample or row order")
    result = result.drop(columns="_merge")
    result["log_market_cap"] = np.log(result.market_cap)
    return result


def controls_design(frame: pd.DataFrame, kind: str, rank_size: bool = False) -> np.ndarray:
    """Daily OLS controls: intercept, reduced industry dummies and/or size."""
    if kind not in CONTROL_KINDS:
        raise ValueError(f"Unknown control kind: {kind}")
    if frame.empty:
        raise ValueError("Empty cross section")
    validate_risk_values(frame)
    columns = [np.ones((len(frame), 1))]
    if kind in ["industry", "industry_size"]:
        columns.append(pd.get_dummies(frame.industry, drop_first=True, dtype=float).to_numpy())
    if kind in ["size", "industry_size"]:
        size = pd.Series(np.log(frame.market_cap.to_numpy(dtype=float)))
        if rank_size:
            size = size.rank(method="average")
        centered = size.to_numpy() - size.mean()
        scale = float(np.sqrt(np.mean(centered ** 2)))
        columns.append((centered / scale if scale > 0 else centered).reshape(-1, 1))
    return np.column_stack(columns)


def project_residuals(values, design) -> tuple[np.ndarray, float]:
    """Least-squares residuals and centered R² for a one-dimensional vector."""
    values, design = np.asarray(values, dtype=float), np.asarray(design, dtype=float)
    if values.ndim != 1 or design.ndim != 2 or len(values) != len(design) or not len(values):
        raise ValueError("Projection requires a nonempty vector and aligned design matrix")
    if not np.isfinite(values).all() or not np.isfinite(design).all():
        raise ValueError("Non-finite projection input")
    residuals = values - design @ np.linalg.lstsq(design, values, rcond=None)[0]
    total = float(np.sum((values - values.mean()) ** 2))
    r_squared = 1 - float(residuals @ residuals) / total if total > 0 else float("nan")
    return residuals, r_squared


def _zero_residual(residuals: np.ndarray, original: np.ndarray) -> bool:
    scale = max(1.0, float(np.linalg.norm(original - original.mean())))
    return bool(np.linalg.norm(residuals - residuals.mean()) <= ZERO_RESIDUAL_RTOL * scale)


def partial_rank_diagnostics(prediction, returns, industry, market_cap, kind: str) -> dict:
    frame = pd.DataFrame({"prediction": prediction, "returns": returns,
                          "industry": industry, "market_cap": market_cap})
    if not np.isfinite(frame[["prediction", "returns"]]).all().all():
        raise ValueError("Non-finite prediction or return")
    ranks = frame[["prediction", "returns"]].rank(method="average").to_numpy()
    design = controls_design(frame, kind, rank_size=True)
    score_residual, score_r2 = project_residuals(ranks[:, 0], design)
    return_residual, return_r2 = project_residuals(ranks[:, 1], design)
    zero_score = _zero_residual(score_residual, ranks[:, 0])
    zero_return = _zero_residual(return_residual, ranks[:, 1])
    status = ("zero_both_residuals" if zero_score and zero_return else
              "zero_prediction_residual" if zero_score else
              "zero_return_residual" if zero_return else "defined")
    correlation = (float(np.corrcoef(score_residual, return_residual)[0, 1])
                   if status == "defined" else float("nan"))
    raw = (float(np.corrcoef(ranks[:, 0], ranks[:, 1])[0, 1])
           if np.ptp(ranks[:, 0]) > 0 and np.ptp(ranks[:, 1]) > 0 else float("nan"))
    return {"partial_rank_ic": correlation, "raw_rank_ic": raw, "status": status,
            "prediction_rank_control_r_squared": score_r2,
            "return_rank_control_r_squared": return_r2,
            "control_design_columns": design.shape[1],
            "control_design_rank": int(np.linalg.matrix_rank(design))}


def partial_rank_correlation(prediction, returns, industry, market_cap, kind: str) -> float:
    """Pearson correlation of projected average ranks; no second ranking."""
    return partial_rank_diagnostics(prediction, returns, industry, market_cap, kind)["partial_rank_ic"]


def tail_weights(scores, fraction: float = .1) -> tuple[np.ndarray, np.ndarray]:
    """Normalized tail weights, allocating tied cutoff scores symmetrically."""
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all():
        raise ValueError("Tail weights require a nonempty finite score vector")
    if not 0 < fraction <= .5:
        raise ValueError("Tail fraction must lie in (0, .5]")
    _, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    target = len(scores) * fraction
    below, above = np.cumsum(counts) - counts, len(scores) - np.cumsum(counts)
    low = (np.clip(target - below, 0, counts) / counts)[inverse] / target
    high = (np.clip(target - above, 0, counts) / counts)[inverse] / target
    return low, high


def make_control_model(name: str) -> Pipeline:
    preprocessing = ColumnTransformer([
        ("industry", OneHotEncoder(handle_unknown="error", sparse_output=False, dtype="float64"), ["industry"]),
        ("size", "passthrough", ["log_market_cap"]),
    ], remainder="drop", sparse_threshold=0)
    steps = [("preprocessing", preprocessing)]
    if name == "Ridge":
        steps.extend([("scaler", StandardScaler()), ("model", Ridge(alpha=100))])
    elif name == "HGB":
        steps.append(("model", HistGradientBoostingRegressor(
            max_iter=55, learning_rate=.065, max_leaf_nodes=17,
            l2_regularization=.2, early_stopping=False, random_state=SEED)))
    else:
        raise ValueError(f"Unknown model: {name}")
    return Pipeline(steps)


def fit_controls(training: pd.DataFrame, evaluation: pd.DataFrame, name: str):
    validate_risk_values(training)
    validate_risk_values(evaluation)
    if training.empty or evaluation.empty or not np.isfinite(training.forward_5d).all():
        raise ValueError("Empty model sample or non-finite training labels")
    x_train = training[["industry", "market_cap"]].assign(log_market_cap=np.log(training.market_cap))
    x_eval = evaluation[["industry", "market_cap"]].assign(log_market_cap=np.log(evaluation.market_cap))
    model = make_control_model(name)
    tick = time.perf_counter()
    model.fit(x_train, training.forward_5d)
    fit_seconds = time.perf_counter() - tick
    prediction = model.predict(x_eval)
    if not np.isfinite(prediction).all():
        raise ValueError(f"{name} controls generated non-finite predictions")
    transform = model.named_steps["preprocessing"]
    metadata = {
        "training_mean": float(training.forward_5d.mean()),
        "industry_categories": transform.named_transformers_["industry"].categories_[0].tolist(),
        "feature_names": transform.get_feature_names_out().tolist(),
        "parameters": jsonable(model.get_params(deep=True)), "fit_seconds": fit_seconds,
        "training_rows": len(training), "evaluation_rows": len(evaluation),
        "preprocessing_fit_scope": "training only; full industry dummies and natural log market_cap",
    }
    if name == "Ridge":
        scaler = model.named_steps["scaler"]
        metadata.update(x_scaler_mean=scaler.mean_.tolist(),
                        x_scaler_scale=scaler.scale_.tolist(),
                        x_scaler_var=scaler.var_.tolist())
    return prediction, metadata


def _interval_fields(label: str, result: dict) -> dict:
    return {f"{label}_{name}": value for name, value in result.items()}


def build_protocol(panel_path: Path, risk_path: Path, benchmark: Path) -> dict:
    paths = {"pit_panel": panel_path, "risk": risk_path,
             "timing_manifest": panel_path.parent / "run_manifest.json",
             "benchmark_protocol": benchmark / "protocol.json"}
    timing = json.loads(paths["timing_manifest"].read_text())
    if sha256(panel_path) != timing["outputs_sha256"][panel_path.name]:
        raise ValueError("PIT panel differs from the factor-timing manifest")
    frozen = json.loads(paths["benchmark_protocol"].read_text())
    if object_hash(frozen["protocol"]) != frozen["protocol_sha256"]:
        raise ValueError("Benchmark frozen protocol hash mismatch")
    if frozen["protocol"]["inputs_sha256"]["pit_panel"] != sha256(panel_path):
        raise ValueError("Benchmark and attribution PIT panels differ")
    for stage in ["validation", "confirmation"]:
        manifest_path = benchmark / stage / "run_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        if manifest["status"] != "completed" or manifest["protocol_sha256"] != frozen["protocol_sha256"]:
            raise ValueError(f"{stage}: benchmark not completed under its frozen protocol")
        if manifest["code_sha256"]["compare_models.py"] != sha256(PROJECT / "compare_models.py"):
            raise ValueError(f"{stage}: current benchmark methods differ from archived code")
        paths[f"benchmark_{stage}_manifest"] = manifest_path
        for filename in ["split_manifest.json"] + [
                f"{name}_{suffix}" for name in MODELS
                for suffix in ["predictions.csv.gz", "fit_metadata.json"]]:
            path = benchmark / stage / filename
            if sha256(path) != manifest["outputs_sha256"][f"{stage}/{filename}"]:
                raise ValueError(f"{stage}: benchmark file hash mismatch: {filename}")
            paths[f"benchmark_{stage}_{filename}"] = path
    parameters = {name: jsonable(make_control_model(name).get_params(deep=True)) for name in MODELS}
    code_paths = [Path(__file__), PROJECT / "compare_models.py", PROJECT / "test_industry_size_attribution.py"]
    return {
        "format_version": 1, "scope": SCOPE, "models": MODELS, "control_kinds": CONTROL_KINDS,
        "model_parameters": parameters, "parameters_sha256": object_hash(parameters),
        "code_sha256": {path.name: sha256(path) for path in code_paths},
        "input_paths": {name: str(path.resolve()) for name, path in paths.items()},
        "inputs_sha256": {name: sha256(path) for name, path in paths.items()},
        "benchmark_protocol_sha256": frozen["protocol_sha256"],
        "versions": {"python": platform.python_version(), **{
            name: importlib.metadata.version(name) for name in ["numpy", "pandas", "scikit-learn"]}},
        "threads_environment": {name: os.environ.get(name) for name in [
            "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]},
        "risk_timing": "Assume risk_exposure dated t is known after close t; independent publication dates are unavailable",
        "risk_merge": "Exact date/asset_id one-to-one left merge; missing industry or nonpositive/nonfinite market_cap is an error",
        "risk_fields": ["industry", "market_cap"], "size_transform": "natural log(market_cap); never use size_z",
        "splits": "Identical compare_models.temporal_split annual validation 2023 and confirmation 2024; mature training labels only",
        "all20": "Reuse archived Ridge/HGB predictions after manifest, key and label checks; no retraining or new winner selection",
        "labels_max_absolute_tolerance": LABEL_ATOL,
        "baseline_fit": "Full training-only industry one-hot encoding (unknown=error) plus log market_cap; Ridge standardizes whole design; HGB unscaled",
        "exposure": "Daily score-vs-logcap Spearman; score OLS R2 for industry/size/both; exact fractional-tie 10% tails",
        "partial_rank_ic": "Average-rank score and return once; project each on intercept, reduced industry dummies and/or standardized average rank logcap; Pearson residual correlation without reranking",
        "projection_scope": "Contemporaneous return projections are ex-post evaluation only, never inputs to fitting or signals",
        "zero_residual": {"relative_tolerance": ZERO_RESIDUAL_RTOL,
                          "scale": "max(1, norm of centered original average ranks)",
                          "behavior": "Undefined NaN with reason; retain date and valid counts"},
        "paired_partial": "partial minus raw RankIC, restricted to jointly finite daily observations; show both common-date means",
        "rmse_comparison": "Pooled RMSE reported for each model; paired interval concerns mean daily cross-sectional RMSE difference",
        "bootstrap": {"kind": "noncircular moving blocks on full business-day calendar",
                      "block_days": BLOCK_DAYS, "draws": BOOTSTRAP_DRAWS, "seed": SEED,
                      "interval": "2.5/97.5 percentiles", "paired": "same dates resampled jointly"},
        "phases": {"origin": PHASE_ORIGIN, "assignment": "busday_count(origin, signal_date) % 5"},
    }


def load_all20(benchmark: Path, stage: str, name: str, evaluation: pd.DataFrame,
               split: dict, training_mean: float) -> pd.DataFrame:
    archived_split = json.loads((benchmark / stage / "split_manifest.json").read_text())
    if archived_split != split:
        raise ValueError(f"{stage}: attribution split differs from original benchmark")
    result = pd.read_csv(benchmark / stage / f"{name}_predictions.csv.gz",
                         parse_dates=["date", "entry_date", "exit_date"])
    result.set_index(KEYS, verify_integrity=True)
    if len(result) != len(evaluation) or key_hash(result) != key_hash(evaluation):
        raise ValueError(f"{stage}/{name}: prediction keys do not match evaluation")
    for column in ["entry_date", "exit_date"]:
        if not result[column].equals(evaluation[column]):
            raise ValueError(f"{stage}/{name}: {column} mismatch")
    if not np.allclose(result.forward_5d, evaluation.forward_5d, rtol=0, atol=LABEL_ATOL):
        raise ValueError(f"{stage}/{name}: archived labels differ from the panel")
    if not np.isfinite(result[["prediction", "forward_5d"]]).all().all():
        raise ValueError(f"{stage}/{name}: non-finite archived prediction or label")
    fitted = json.loads((benchmark / stage / f"{name}_fit_metadata.json").read_text())
    if not np.isclose(fitted["training_mean"], training_mean, rtol=0, atol=LABEL_ATOL):
        raise ValueError(f"{stage}/{name}: archived training mean differs")
    # Use the canonical panel label for every same-sample comparison.
    result["forward_5d"] = evaluation.forward_5d.to_numpy()
    return result


def prediction_diagnostics(result: pd.DataFrame) -> pd.DataFrame:
    daily = daily_metrics(result)
    errors = result.assign(squared_error=(result.prediction - result.forward_5d) ** 2)
    rmse = errors.groupby("date", sort=True).squared_error.mean().pow(.5).rename("daily_rmse")
    return daily.merge(rmse, on="date", validate="one_to_one")


def exposure_diagnostics(result: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    exposures, industries, partials = [], [], []
    for date, frame in result.groupby("date", sort=True):
        low, high = tail_weights(frame.prediction)
        logcap = np.log(frame.market_cap.to_numpy())
        size_percentile = pd.Series(logcap).rank(method="average", pct=True).to_numpy()
        score_rank = frame.prediction.rank(method="average").to_numpy()
        size_rank = pd.Series(logcap).rank(method="average").to_numpy()
        record = {"date": date, "n_assets": len(frame),
                  "score_logcap_spearman": (float(np.corrcoef(score_rank, size_rank)[0, 1])
                                             if np.ptp(score_rank) > 0 and np.ptp(size_rank) > 0 else np.nan),
                  "low_log_market_cap": float(low @ logcap),
                  "high_log_market_cap": float(high @ logcap),
                  "high_minus_low_log_market_cap": float((high - low) @ logcap),
                  "low_size_percentile": float(low @ size_percentile),
                  "high_size_percentile": float(high @ size_percentile),
                  "high_minus_low_size_percentile": float((high - low) @ size_percentile)}
        for kind in CONTROL_KINDS:
            _, r_squared = project_residuals(frame.prediction, controls_design(frame, kind))
            record[f"score_{kind}_r_squared"] = r_squared
            partials.append({"date": date, "n_assets": len(frame), "control": kind,
                             **partial_rank_diagnostics(frame.prediction.to_numpy(),
                                                        frame.forward_5d.to_numpy(),
                                                        frame.industry.to_numpy(),
                                                        frame.market_cap.to_numpy(), kind)})
        exposures.append(record)
        for industry in sorted(frame.industry.unique()):
            mask = frame.industry.eq(industry).to_numpy()
            pool, low_share, high_share = float(mask.mean()), float(low[mask].sum()), float(high[mask].sum())
            industries.append({"date": date, "industry": industry, "pool_weight": pool,
                               "low_weight": low_share, "high_weight": high_share,
                               "low_minus_pool_weight": low_share - pool,
                               "high_minus_pool_weight": high_share - pool,
                               "high_minus_low_weight": high_share - low_share})
    return pd.DataFrame(exposures), pd.DataFrame(industries), pd.DataFrame(partials)


def summarize_partial(part: pd.DataFrame) -> dict:
    indexed = part.set_index("date")
    common = indexed.raw_rank_ic.notna() & indexed.partial_rank_ic.notna()
    paired_raw = indexed.raw_rank_ic.where(common)
    paired_partial = indexed.partial_rank_ic.where(common)
    return {
        "signal_dates": len(part), "common_valid_dates": int(common.sum()),
        "raw_on_common_mean": float(paired_raw.mean()),
        "partial_on_common_mean": float(paired_partial.mean()),
        **_interval_fields("raw_rank_ic", moving_block_interval(indexed.raw_rank_ic)),
        **_interval_fields("partial_rank_ic", moving_block_interval(indexed.partial_rank_ic)),
        **_interval_fields("partial_minus_raw", paired_mean_difference(paired_partial, paired_raw)),
        **{f"status_{status}_dates": int(part.status.eq(status).sum()) for status in [
            "defined", "zero_both_residuals", "zero_prediction_residual", "zero_return_residual"]},
        "prediction_rank_control_r_squared_mean": float(part.prediction_rank_control_r_squared.mean()),
        "return_rank_control_r_squared_mean": float(part.return_rank_control_r_squared.mean()),
    }


def summarize_exposures(frame: pd.DataFrame, group_columns: list[str], metrics: list[str]):
    summaries, quarters = [], []
    for key, group in frame.groupby(group_columns, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        identity = dict(zip(group_columns, key))
        for metric in metrics:
            summaries.append({**identity, "metric": metric,
                              **moving_block_interval(group.set_index("date")[metric])})
        for quarter, part in group.groupby(group.date.dt.to_period("Q"), sort=True):
            for metric in metrics:
                quarters.append({**identity, "quarter": str(quarter), "metric": metric,
                                 **moving_block_interval(part.set_index("date")[metric])})
    return pd.DataFrame(summaries), pd.DataFrame(quarters)


def run_stage(stage: str, panel: pd.DataFrame, benchmark: Path, out: Path, protocol: dict) -> None:
    started = utc_now()
    stage_out = out / stage
    stage_out.mkdir()
    training, evaluation, split = temporal_split(panel, stage)
    write_json(stage_out / "split_manifest.json", split)
    training_mean = float(training.forward_5d.mean())
    print(f"{stage}: train={len(training)}, evaluation={len(evaluation)}", flush=True)
    summaries, quarters, phases, daily_parts, comparisons = [], [], [], [], []
    exposures, industries, partials = [], [], []
    for name in MODELS:
        original = load_all20(benchmark, stage, name, evaluation, split, training_mean)
        original_path = benchmark / stage / f"{name}_predictions.csv.gz"
        shutil.copyfile(original_path, stage_out / f"{name}_ALL20_predictions.csv.gz")
        print(f"{utc_now()} fitting {name} industry/size baseline", flush=True)
        predicted, metadata = fit_controls(training, evaluation, name)
        control = evaluation[KEYS + ["entry_date", "exit_date", "forward_5d"]].assign(prediction=predicted)
        control.to_csv(stage_out / f"{name}_CONTROLS_predictions.csv.gz", index=False,
                       compression={"method": "gzip", "compresslevel": 1, "mtime": 0})
        write_json(stage_out / f"{name}_CONTROLS_fit_metadata.json", metadata)
        daily_by_variant = {}
        for variant, predictions in [("ALL20", original), ("CONTROLS", control)]:
            daily = prediction_diagnostics(predictions)
            indexed = daily.set_index("date")
            daily_by_variant[variant] = indexed
            summary = {"model": name, "variant": variant,
                       **error_metrics(predictions, training_mean)}
            for metric in ["rank_ic", "gross_decile_spread", "daily_rmse"]:
                summary.update(_interval_fields(metric, moving_block_interval(indexed[metric])))
            summaries.append(summary)
            daily_parts.append(daily.assign(model=name, variant=variant))
            for quarter, part in predictions.groupby(predictions.date.dt.to_period("Q"), sort=True):
                days = daily.loc[daily.date.dt.to_period("Q") == quarter].set_index("date")
                row = {"model": name, "variant": variant, "quarter": str(quarter),
                       **error_metrics(part, training_mean)}
                for metric in ["rank_ic", "gross_decile_spread", "daily_rmse"]:
                    row.update(_interval_fields(metric, moving_block_interval(days[metric])))
                quarters.append(row)
            phased = daily.assign(phase=phase_numbers(daily.date))
            for phase, days in phased.groupby("phase", sort=True):
                phases.append({"model": name, "variant": variant, "phase": int(phase),
                               "signal_dates": len(days),
                               **{f"{metric}_mean": float(days[metric].mean()) for metric in [
                                   "rank_ic", "gross_decile_spread", "daily_rmse"]}})
        for metric in ["rank_ic", "gross_decile_spread", "daily_rmse"]:
            all20, baseline = daily_by_variant["ALL20"][metric], daily_by_variant["CONTROLS"][metric]
            common = all20.notna() & baseline.notna()
            comparisons.append({"model": name, "metric": metric,
                                "comparison": "ALL20 minus CONTROLS",
                                "all20_on_common_mean": float(all20[common].mean()),
                                "controls_on_common_mean": float(baseline[common].mean()),
                                **paired_mean_difference(all20, baseline)})
        enriched = original.merge(evaluation[KEYS + ["industry", "market_cap"]], on=KEYS,
                                  how="left", sort=False, validate="one_to_one")
        exposure, industry, partial = exposure_diagnostics(enriched)
        exposures.append(exposure.assign(model=name))
        industries.append(industry.assign(model=name))
        partials.append(partial.assign(model=name))
    tables = {"summary.csv": pd.DataFrame(summaries), "quarterly.csv": pd.DataFrame(quarters),
              "phases.csv": pd.DataFrame(phases), "daily_metrics.csv": pd.concat(daily_parts),
              "paired_baseline_comparison.csv": pd.DataFrame(comparisons),
              "exposure_daily.csv": pd.concat(exposures),
              "industry_exposure_daily.csv": pd.concat(industries),
              "partial_rank_daily.csv": pd.concat(partials)}
    partial_summaries, partial_quarters, partial_phases = [], [], []
    for (name, kind), part in tables["partial_rank_daily.csv"].groupby(["model", "control"], sort=True):
        partial_summaries.append({"model": name, "control": kind, **summarize_partial(part)})
        for quarter, days in part.groupby(part.date.dt.to_period("Q"), sort=True):
            partial_quarters.append({"model": name, "control": kind, "quarter": str(quarter),
                                     **summarize_partial(days)})
        phased = part.assign(phase=phase_numbers(part.date))
        for phase, days in phased.groupby("phase", sort=True):
            common = days.raw_rank_ic.notna() & days.partial_rank_ic.notna()
            partial_phases.append({"model": name, "control": kind, "phase": int(phase),
                                   "signal_dates": len(days), "common_valid_dates": int(common.sum()),
                                   "raw_on_common_mean": float(days.loc[common, "raw_rank_ic"].mean()),
                                   "partial_on_common_mean": float(days.loc[common, "partial_rank_ic"].mean()),
                                   "partial_minus_raw_mean": float((days.loc[common, "partial_rank_ic"]
                                                                    - days.loc[common, "raw_rank_ic"]).mean())})
    tables.update({"partial_rank_summary.csv": pd.DataFrame(partial_summaries),
                   "partial_rank_quarterly.csv": pd.DataFrame(partial_quarters),
                   "partial_rank_phases.csv": pd.DataFrame(partial_phases)})
    for prefix, group_cols in [("exposure", ["model"]), ("industry_exposure", ["model", "industry"])]:
        frame = tables[f"{prefix}_daily.csv"]
        metrics = [column for column in frame.columns if column not in group_cols + ["date", "n_assets"]]
        summary, quarterly = summarize_exposures(frame, group_cols, metrics)
        tables[f"{prefix}_summary.csv"], tables[f"{prefix}_quarterly.csv"] = summary, quarterly
    for filename, frame in tables.items():
        frame.to_csv(stage_out / filename, index=False)
    write_json(stage_out / "run_manifest.json", {
        "status": "completed", "stage": stage, "started_at_utc": started, "finished_at_utc": utc_now(),
        "protocol_sha256": object_hash(protocol), "code_sha256": protocol["code_sha256"],
        "inputs_sha256": protocol["inputs_sha256"],
        "outputs_sha256": {path.name: sha256(path) for path in sorted(stage_out.iterdir())},
    })
    print(f"{stage} completed", flush=True)


def run(panel_path: Path, risk_path: Path, benchmark: Path, out: Path) -> None:
    if out.exists():
        raise FileExistsError(f"Output directory already exists: {out}; choose a new directory")
    started = utc_now()
    protocol = build_protocol(panel_path, risk_path, benchmark)
    out.mkdir(parents=True)
    write_json(out / "protocol.json", {"frozen_at_utc": utc_now(),
                                       "protocol_sha256": object_hash(protocol), "protocol": protocol})
    panel = pd.read_csv(panel_path, parse_dates=["date", "entry_date", "exit_date"])
    risk = pd.read_csv(risk_path, usecols=KEYS + ["industry", "market_cap"], parse_dates=["date"])
    enriched = merge_risk(panel, risk)
    write_json(out / "risk_merge_audit.json", {
        "panel_rows": len(panel), "risk_rows": len(risk), "merged_rows": len(enriched),
        "panel_ordered_keys_sha256": key_hash(panel), "merged_ordered_keys_sha256": key_hash(enriched),
        "missing_risk_rows": 0, "industry_count": int(enriched.industry.nunique()),
        "market_cap_min": float(enriched.market_cap.min()), "market_cap_max": float(enriched.market_cap.max()),
        "assumption": protocol["risk_timing"],
    })
    for stage in ["validation", "confirmation"]:
        run_stage(stage, enriched, benchmark, out, protocol)
    for name, path in protocol["input_paths"].items():
        if sha256(Path(path)) != protocol["inputs_sha256"][name]:
            raise ValueError(f"Input changed during run: {name}")
    for name, expected in protocol["code_sha256"].items():
        if sha256(PROJECT / name) != expected:
            raise ValueError(f"Source changed during run: {name}")
    write_json(out / "run_manifest.json", {
        "status": "completed", "started_at_utc": started, "finished_at_utc": utc_now(),
        "command": sys.orig_argv, "scope": SCOPE, "protocol_sha256": object_hash(protocol),
        "code_sha256": protocol["code_sha256"], "inputs_sha256": protocol["inputs_sha256"],
        "outputs_sha256": {str(path.relative_to(out)): sha256(path)
                           for path in sorted(out.rglob("*")) if path.is_file()},
    })
    print(f"Completed industry/size attribution: {out}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path,
                        default=PROJECT / "outputs" / "factor_timing" / "pit_research_panel.csv.gz")
    parser.add_argument("--risk", type=Path, default=DEFAULT_RISK)
    parser.add_argument("--benchmark", type=Path, default=PROJECT / "outputs" / "model_benchmark")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.panel, args.risk, args.benchmark, args.out)
