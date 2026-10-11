"""Fixed-parameter, expanding-window models with strictly temporal predictions.

Run from the workspace root with:
    OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B asset_allocation/temporal_model.py

The five-day spread below is a signal diagnostic, not a daily portfolio return.
Trading code must consume the complete predictions, including unlabelled rows.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys

import joblib
import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor

from factor_timing import (
    DEFAULT_DATA, DELAYED, FACTORS, KEYS, PROJECT, align_factor, build_labels,
    daily_rank_ic, sha256, validate_keys,
)


PARAMETERS = dict(
    max_iter=55, learning_rate=.065, max_leaf_nodes=17,
    l2_regularization=.2, random_state=1021, early_stopping=False,
)
DATA_FILES = ["market_daily", "factors", "universe", "factor_availability"]


@dataclass(frozen=True)
class Fold:
    fold_id: str
    role: str
    signal_start: pd.Timestamp
    signal_end: pd.Timestamp
    train_cutoff: pd.Timestamp


def make_folds() -> list[Fold]:
    """2021 history -> 2022 validation; quarterly expanding windows in 2023-24."""
    periods = [("validation_2022", "validation", "2022-01-01", "2022-12-31")]
    for period in pd.period_range("2023Q1", "2024Q4", freq="Q"):
        periods.append((str(period), "oos", period.start_time, period.end_time.normalize()))
    folds = []
    for fold_id, role, start, end in periods:
        dates = pd.bdate_range(start, end)
        folds.append(Fold(fold_id, role, dates[0], dates[-1], dates[0] - BDay(1)))
    return folds


def load_signal_panel(root: Path) -> pd.DataFrame:
    """Build PIT inputs before membership filtering; never filter on future labels."""
    data = root / "data"
    market = pd.read_csv(data / "market_daily.csv.gz", parse_dates=["date"])
    factors = pd.read_csv(data / "factors.csv.gz", parse_dates=["date"])
    universe = pd.read_csv(data / "universe.csv.gz", parse_dates=["date"])
    availability = pd.read_csv(
        data / "factor_availability.csv.gz",
        parse_dates=["date"] + [f"{factor}_available_on" for factor in DELAYED],
    )
    market_keys = validate_keys(market, "market")
    for name, frame in [("factors", factors), ("universe", universe), ("availability", availability)]:
        if not market_keys.equals(validate_keys(frame, name)):
            raise ValueError(f"{name}: keys differ from market")
    if not availability.published_at.eq("after_close").all():
        raise ValueError("availability: expected after_close release convention")
    if universe[["is_listed", "is_member"]].isna().any().any():
        raise ValueError("universe: missing signal-date eligibility")

    panel = build_labels(market).merge(factors, on=KEYS, validate="one_to_one").merge(
        universe, on=KEYS, validate="one_to_one"
    )
    for factor in DELAYED:
        available = f"{factor}_available_on"
        snapshots = factors[KEYS + [factor]].merge(
            availability[KEYS + [available]], on=KEYS, validate="one_to_one"
        )
        aligned = align_factor(market[KEYS], snapshots, factor)
        panel = panel.drop(columns=factor).merge(aligned, on=KEYS, validate="one_to_one")
    return panel.loc[panel.is_listed & panel.is_member].sort_values(KEYS).reset_index(drop=True)


def split_fold(panel: pd.DataFrame, fold: Fold) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The training answer must be observable by the pre-period close."""
    train = panel.loc[
        (panel.date <= fold.train_cutoff)
        & (panel.exit_date <= fold.train_cutoff)
        & panel.forward_5d.notna()
    ]
    signals = panel.loc[panel.date.between(fold.signal_start, fold.signal_end)]
    if train.empty or signals.empty:
        raise ValueError(f"{fold.fold_id}: empty training or prediction period")
    if not train.exit_date.max() <= fold.train_cutoff < signals.date.min():
        raise AssertionError(f"{fold.fold_id}: training labels cross the prediction boundary")
    if len(train.set_index(KEYS).index.intersection(signals.set_index(KEYS).index)):
        raise AssertionError(f"{fold.fold_id}: training and prediction keys overlap")
    return train, signals


def make_model() -> HistGradientBoostingRegressor:
    """Disable the estimator's random internal validation split explicitly."""
    return HistGradientBoostingRegressor(**PARAMETERS)


def key_hash(frame: pd.DataFrame) -> str:
    return hashlib.sha256(frame[KEYS].to_csv(index=False).encode("utf-8")).hexdigest()


def fit_fold(panel: pd.DataFrame, fold: Fold):
    train, signals = split_fold(panel, fold)
    model = make_model()
    model.fit(train[FACTORS].to_numpy(dtype=np.float32), train.forward_5d.to_numpy(dtype=np.float32))
    prediction = model.predict(signals[FACTORS].to_numpy(dtype=np.float32))
    if not np.isfinite(prediction).all():
        raise ValueError(f"{fold.fold_id}: non-finite predictions")
    if model.n_iter_ != PARAMETERS["max_iter"] or model.do_early_stopping_:
        raise AssertionError(f"{fold.fold_id}: unexpected early stopping")
    result = signals[KEYS + ["entry_date", "exit_date", "forward_5d"]].copy()
    result["prediction"] = prediction
    result["fold_id"] = fold.fold_id
    result["role"] = fold.role
    result["train_cutoff"] = fold.train_cutoff
    record = {
        "fold_id": fold.fold_id, "role": fold.role,
        "train_cutoff": fold.train_cutoff, "signal_start": fold.signal_start,
        "signal_end": fold.signal_end, "train_rows": len(train),
        "train_signal_min": train.date.min(), "train_signal_max": train.date.max(),
        "train_label_exit_min": train.exit_date.min(), "train_label_exit_max": train.exit_date.max(),
        "train_dates": train.date.nunique(), "train_assets": train.asset_id.nunique(),
        "prediction_rows": len(result), "prediction_dates": result.date.nunique(),
        "prediction_date_min": result.date.min(), "prediction_date_max": result.date.max(),
        "prediction_rows_without_label": int(result.forward_5d.isna().sum()),
        "train_keys_sha256": key_hash(train), "prediction_keys_sha256": key_hash(signals),
        "training_prediction_key_overlap": 0, "maturity_boundary_passed": True,
        "fitted_iterations": int(model.n_iter_), "early_stopping_used": bool(model.do_early_stopping_),
        "model_parameters": json.dumps(model.get_params(), sort_keys=True),
    }
    return model, result, record


def validate_coverage(predictions: pd.DataFrame, panel: pd.DataFrame, folds: list[Fold]) -> None:
    """Exactly one prediction for every eligible signal in the fixed periods."""
    actual = validate_keys(predictions, "predictions")
    selected = pd.Series(False, index=panel.index)
    for fold in folds:
        current = panel.date.between(fold.signal_start, fold.signal_end)
        if (selected & current).any():
            raise AssertionError("Prediction fold dates overlap")
        selected |= current
    expected = validate_keys(panel.loc[selected], "expected signals")
    if not actual.equals(expected):
        raise AssertionError("Prediction coverage differs from signal-date eligibility")
    if not (predictions.train_cutoff < predictions.date).all():
        raise AssertionError("Prediction is not strictly after its training cutoff")


def pearson(prediction: pd.Series, target: pd.Series) -> float:
    """Return undefined correlation as NaN when either series is constant."""
    if len(prediction) < 2 or prediction.nunique() < 2 or target.nunique() < 2:
        return float("nan")
    return float(np.corrcoef(prediction, target)[0, 1])


def daily_diagnostics(predictions: pd.DataFrame) -> pd.DataFrame:
    """Reproduce the original labelled-universe spread, only as a diagnostic.

    Missing future outcomes are excluded here for evaluation, never for signal
    generation. Quantiles are recomputed on the evaluable universe, as in the
    handoff code; a trading ledger must instead rank complete signal universes.
    """
    labelled = predictions.dropna(subset=["forward_5d"])
    daily_ic = daily_rank_ic(labelled, ["prediction"])
    records = []
    for date, signals in predictions.groupby("date", sort=True):
        sub = labelled.loc[labelled.date == date]
        long_return = short_return = float("nan")
        long_count = short_count = 0
        if len(sub) >= 60:
            long = sub.loc[sub.prediction >= sub.prediction.quantile(.9), "forward_5d"]
            short = sub.loc[sub.prediction <= sub.prediction.quantile(.1), "forward_5d"]
            long_return, short_return = long.mean(), short.mean()
            long_count, short_count = len(long), len(short)
        records.append({
            "date": date, "signal_rows": len(signals), "labelled_rows": len(sub),
            "missing_label_rows": len(signals) - len(sub),
            "pearson": pearson(sub.prediction, sub.forward_5d),
            "rmse": float(np.sqrt(np.mean((sub.prediction - sub.forward_5d) ** 2))) if len(sub) else np.nan,
            "rank_ic": daily_ic.loc[date, "prediction"] if len(sub) else np.nan,
            "long_assets": long_count, "short_assets": short_count,
            "long_5d_return": long_return, "short_5d_return": short_return,
            "long_short_5d_spread": long_return - short_return,
        })
    return pd.DataFrame(records)


def period_metrics(predictions: pd.DataFrame, daily: pd.DataFrame, include_validation: bool = True) -> pd.DataFrame:
    periods = [("2023_2024_oos", 2023, 2024), ("2023", 2023, 2023), ("2024", 2024, 2024)]
    if include_validation:
        periods.insert(0, ("2022_validation", 2022, 2022))
    records = []
    for period, start_year, end_year in periods:
        signals = predictions.loc[predictions.date.dt.year.between(start_year, end_year)]
        sub = signals.dropna(subset=["forward_5d"])
        dates = daily.loc[daily.date.dt.year.between(start_year, end_year)]
        records.append({
            "period": period, "signal_rows": len(signals), "labelled_rows": len(sub),
            "missing_label_rows": len(signals) - len(sub), "signal_dates": signals.date.nunique(),
            "pearson": pearson(sub.prediction, sub.forward_5d),
            "rmse": float(np.sqrt(np.mean((sub.prediction - sub.forward_5d) ** 2))) if len(sub) else np.nan,
            "mean_daily_rank_ic": dates.rank_ic.mean(), "valid_rank_ic_dates": dates.rank_ic.notna().sum(),
            "mean_forward_5d_spread": dates.long_short_5d_spread.mean(),
            "spread_dates": dates.long_short_5d_spread.notna().sum(),
            "mean_long_5d_return": dates.long_5d_return.mean(),
            "metric_scope": "overlapping forward-return signal diagnostic; not daily portfolio P&L",
        })
    return pd.DataFrame(records)


def compare_old_predictions(predictions: pd.DataFrame, old_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """All three models evaluated on identical labelled 2023-24 date/asset keys."""
    new_signals = predictions.loc[predictions.date.dt.year.between(2023, 2024)]
    matched = new_signals.loc[
        new_signals.forward_5d.notna(),
        KEYS + ["forward_5d", "prediction"],
    ].rename(columns={"prediction": "temporal_oos"})
    for variant in ["original", "pit"]:
        old = pd.read_csv(old_root / variant / "baseline_predictions.csv.gz", parse_dates=["date"])
        validate_keys(old, variant)
        if not np.isfinite(old.prediction).all():
            raise ValueError(f"{variant}: non-finite predictions")
        matched = matched.merge(
            old[KEYS + ["prediction"]].rename(columns={"prediction": variant}),
            on=KEYS, how="left", validate="one_to_one",
        )
    common = matched.dropna(subset=["original", "pit"])
    if common.empty:
        raise ValueError("No common evaluable 2023-24 signals across the three models")
    summaries, daily_records = [], []
    for variant in ["original", "pit", "temporal_oos"]:
        sub = common[KEYS + ["forward_5d", variant]].rename(columns={variant: "prediction"})
        daily = daily_diagnostics(sub)
        summary = period_metrics(sub, daily, include_validation=False)
        for index, period in summary.period.items():
            years = [2023, 2024] if period == "2023_2024_oos" else [int(period)]
            source = new_signals.loc[new_signals.date.dt.year.isin(years)]
            labelled = matched.loc[matched.date.dt.year.isin(years)]
            paired = common.loc[common.date.dt.year.isin(years)]
            counts = {
                "new_signal_rows": len(source), "new_labelled_rows": len(labelled),
                "common_sample_rows": len(paired),
                "original_unmatched_labelled_rows": int(labelled.original.isna().sum()),
                "pit_unmatched_labelled_rows": int(labelled.pit.isna().sum()),
                "common_excluded_labelled_rows": len(labelled) - len(paired),
                "common_excluded_signal_rows": len(source) - len(paired),
            }
            for name, value in counts.items():
                summary.loc[index, name] = value
        summary[list(counts)] = summary[list(counts)].astype(int)
        summary["variant"] = variant
        summary["is_temporal_oos"] = variant == "temporal_oos"
        summary["evaluation_design"] = (
            "pre-period mature-label expanding training" if variant == "temporal_oos"
            else "original random split / full-panel predictions; includes training observations; NOT temporal OOS"
        )
        summary["common_keys_sha256"] = key_hash(sub)
        summaries.append(summary)
        daily_records.append(daily.assign(variant=variant))
    return pd.concat(summaries, ignore_index=True), pd.concat(daily_records, ignore_index=True)


def run(root: Path, old_root: Path, out: Path) -> None:
    started = datetime.now(timezone.utc).isoformat()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing results: {out}")
    input_paths = [root / "data" / f"{name}.csv.gz" for name in DATA_FILES]
    input_paths += [root / "data" / "data_dictionary.md"]
    input_paths += [old_root / variant / "baseline_predictions.csv.gz" for variant in ["original", "pit"]]
    input_hashes = {str(path.resolve()): sha256(path) for path in input_paths}
    print("Loading complete PIT signal panel (future labels do not filter signals)", flush=True)
    panel = load_signal_panel(root)
    folds = make_folds()
    out.mkdir(parents=True, exist_ok=True)
    (out / "models").mkdir()
    predictions, records = [], []
    for fold in folds:
        train, signals = split_fold(panel, fold)
        print(f"{fold.fold_id}: train={len(train)} predict={len(signals)} cutoff={fold.train_cutoff.date()}", flush=True)
        model, result, record = fit_fold(panel, fold)
        model_path = out / "models" / f"{fold.fold_id}.joblib"
        joblib.dump(model, model_path)
        record["model_file"] = str(model_path.relative_to(out))
        predictions.append(result)
        records.append(record)
        print(f"{fold.fold_id}: completed {model.n_iter_} iterations; unlabelled signals={record['prediction_rows_without_label']}", flush=True)
    predictions = pd.concat(predictions, ignore_index=True).sort_values(KEYS).reset_index(drop=True)
    validate_coverage(predictions, panel, folds)
    predictions.to_csv(
        out / "predictions.csv.gz", index=False,
        compression={"method": "gzip", "compresslevel": 1, "mtime": 0},
    )
    pd.DataFrame(records).to_csv(out / "fold_manifest.csv", index=False)
    daily = daily_diagnostics(predictions)
    daily.to_csv(out / "daily_diagnostics.csv", index=False)
    metrics = period_metrics(predictions, daily)
    metrics.to_csv(out / "metrics.csv", index=False)
    comparison, comparison_daily = compare_old_predictions(predictions, old_root)
    comparison.to_csv(out / "comparison.csv", index=False)
    comparison_daily.to_csv(out / "comparison_daily.csv", index=False)
    code_paths = [Path(__file__), PROJECT / "factor_timing.py", PROJECT / "test_temporal_model.py"]
    manifest = {
        "started_at_utc": started, "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": sys.orig_argv, "data_root": str(root.resolve()), "output_root": str(out.resolve()),
        "environment": {
            "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__, "joblib": joblib.__version__,
            "threads": {key: os.environ.get(key) for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"]},
        },
        "factor_columns": FACTORS, "model_parameters": make_model().get_params(),
        "prediction_rows": len(predictions), "prediction_rows_without_label": int(predictions.forward_5d.isna().sum()),
        "prediction_dates": int(predictions.date.nunique()), "prediction_keys_unique": True,
        "fold_count": len(folds), "full_signal_coverage_passed": True,
        "assumptions": {
            "calendar": "Synthetic Monday-Friday trading days; no exchange-holiday calendar",
            "features": "factor_03/factor_12 latest released snapshot with available_on <= signal close; other 18 at reference close",
            "training_cutoff": "Previous synthetic business-day close before prediction period; train date <= cutoff AND label exit_date <= cutoff",
            "training_rows": "All historical signal-date listed members with mature, non-missing five-day labels; no random sample",
            "validation": "One model trained before 2022 and fixed throughout 2022; no parameter selection",
            "oos": "Quarterly expanding windows 2023Q1 through 2024Q4; prior realised outcomes may enter later training",
            "early_stopping": "Disabled explicitly; 55 iterations; no random inner validation and no tuning",
            "signal_membership": "Only current is_listed and is_member; no future label availability or future delisting filter",
            "missing_features": "HGB native NaN handling; no backward fill; retain missing latest snapshots",
            "prediction_tail": "Includes final 2024-12-31 signal and other rows without future endpoints; execution layer must set its tradable horizon",
            "label": "open(t+6 business days) / open(t+1 business day) - 1; known at exit-date open",
            "diagnostic_scope": "Pearson/RMSE/RankIC and top/bottom decile spread on evaluable labels only; rerank that universe; not portfolio daily returns",
            "comparison": "Old original, old PIT, and temporal PIT on common labelled 2023-24 date/asset keys; old arms NOT temporal OOS",
            "intervention": "Changes chronological evaluation, all-history training size, and disables random early stopping together; not an isolated estimate of one component",
        },
        "inputs_sha256": input_hashes,
        "code_sha256": {str(path.resolve()): sha256(path) for path in code_paths},
        "outputs_sha256": {str(path.relative_to(out)): sha256(path) for path in sorted(out.rglob("*")) if path.is_file()},
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(metrics.to_string(index=False), flush=True)
    print(comparison.to_string(index=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--old-root", type=Path, default=PROJECT / "outputs" / "model_rerun")
    parser.add_argument("--out", type=Path, default=PROJECT / "outputs" / "temporal_oos")
    args = parser.parse_args()
    run(args.root, args.old_root, args.out)
