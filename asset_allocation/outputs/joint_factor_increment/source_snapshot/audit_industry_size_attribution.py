"""Independently audit the frozen industry/size study without altering its results.

The audit does not import the study or its statistics. Daily partial correlations
use Frisch-Waugh group demeaning and a scalar size projection; controls baselines
are independently refitted from pandas dummy columns. Verification is write-once.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


PROJECT = Path(__file__).resolve().parent
KEYS = ["date", "asset_id"]
MODELS = ["Ridge", "HGB"]
KINDS = ["industry", "size", "industry_size"]
ERRORS: dict[str, float] = {}
CHECKS: dict[str, int] = {}


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def read_json(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def object_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def numeric(actual, expected, category, atol=2e-11):
    left, right = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    require(left.shape == right.shape, f"{category}: shapes differ {left.shape}, {right.shape}")
    require(np.array_equal(np.isnan(left), np.isnan(right)), f"{category}: undefined values differ")
    valid = ~np.isnan(left)
    require(np.isfinite(left[valid]).all() and np.isfinite(right[valid]).all(), f"{category}: infinity")
    error = float(np.max(np.abs(left[valid] - right[valid]))) if valid.any() else 0.0
    ERRORS[category] = max(ERRORS.get(category, 0.0), error)
    require(error <= atol, f"{category}: maximum absolute error {error} exceeds {atol}")


def key_digest(frame):
    return hashlib.sha256(frame[KEYS].to_csv(index=False).encode()).hexdigest()


def verify_hashes(root, manifest):
    for relative, expected in manifest["outputs_sha256"].items():
        require(digest(root / relative) == expected, f"Output hash mismatch: {root / relative}")
    CHECKS["output_hash_records"] = CHECKS.get("output_hash_records", 0) + len(manifest["outputs_sha256"])


def load_csv(path):
    frame = pd.read_csv(path)
    for column in ["date", "entry_date", "exit_date"]:
        if column in frame:
            frame[column] = pd.to_datetime(frame[column])
    return frame


def assert_keys(frame, expected, label):
    require(not frame.duplicated(KEYS).any(), f"{label}: duplicate keys")
    require(frame[KEYS].reset_index(drop=True).equals(expected[KEYS].reset_index(drop=True)),
            f"{label}: ordered sample keys differ")


def rank(values):
    return pd.Series(np.asarray(values)).rank(method="average").to_numpy()


def correlation(x, y):
    x, y = np.asarray(x), np.asarray(y)
    x, y = x - x.mean(), y - y.mean()
    denominator = np.linalg.norm(x) * np.linalg.norm(y)
    return float(x @ y / denominator) if denominator else float("nan")


def demean(values, groups):
    counts = np.bincount(groups)
    means = np.bincount(groups, weights=values) / counts
    return values - means[groups]


def residual(values, groups, size, kind):
    """FWL: remove intercept/industry, then the remaining scalar size direction."""
    values = np.asarray(values, dtype=float)
    centered = demean(values, groups) if "industry" in kind else values - values.mean()
    if "size" in kind:
        size_residual = demean(size, groups) if "industry" in kind else size - size.mean()
        norm_squared = float(size_residual @ size_residual)
        if norm_squared > 0:
            centered = centered - size_residual * float(size_residual @ centered) / norm_squared
    return centered


def r_squared(original, residuals):
    centered = original - original.mean()
    total = float(centered @ centered)
    return 1 - float(residuals @ residuals) / total if total else float("nan")


def tail_allocation(scores, descending):
    """Allocate exactly n/10 slots by score levels, sharing a cutoff tie equally."""
    scores = np.asarray(scores)
    weights = np.zeros(len(scores))
    remaining = target = len(scores) / 10
    levels = sorted(set(scores), reverse=descending)
    for value in levels:
        mask = scores == value
        count = int(mask.sum())
        take = min(remaining, count)
        weights[mask] = take / count / target
        remaining -= take
        if remaining <= 0:
            break
    numeric(weights.sum(), 1, "tail_weight_conservation")
    return weights


def interval(series, protocol):
    series = series.sort_index()
    require(not series.index.has_duplicates, "Bootstrap has duplicate dates")
    defined = int(series.notna().sum())
    if not defined:
        return dict(n_dates=0, mean=np.nan, ci_low=np.nan, ci_high=np.nan)
    values = series.reindex(pd.bdate_range(series.index.min(), series.index.max())).to_numpy()
    config = protocol["bootstrap"]
    block, draws = config["block_days"], config["draws"]
    require(len(values) >= block, "Bootstrap interval shorter than frozen block")
    rng = np.random.default_rng(config["seed"])
    starts = rng.integers(len(values) - block + 1, size=(draws, math.ceil(len(values) / block)))
    locations = np.concatenate([starts[..., None] + np.arange(block)], axis=-1).reshape(draws, -1)
    samples = values[locations[:, :len(values)]]
    require(np.isfinite(samples).any(axis=1).all(), "All-missing bootstrap draw")
    means = np.nanmean(samples, axis=1)
    low, high = np.quantile(means, [.025, .975])
    return dict(n_dates=defined, mean=float(series.mean()), ci_low=float(low), ci_high=float(high))


def check_interval(row, series, prefix, protocol):
    for key, value in interval(series, protocol).items():
        numeric(row[f"{prefix}_{key}" if prefix else key], value, "bootstrap_intervals")


def check_partial_summary(row, part, protocol, full_interval):
    part = part.set_index("date").sort_index()
    common = part.raw_rank_ic.notna() & part.partial_rank_ic.notna()
    numeric(row.signal_dates, len(part), "partial_counts")
    numeric(row.common_valid_dates, common.sum(), "partial_counts")
    numeric(row.raw_on_common_mean, part.raw_rank_ic[common].mean(), "partial_summary")
    numeric(row.partial_on_common_mean, part.partial_rank_ic[common].mean(), "partial_summary")
    paired = part.partial_rank_ic.where(common) - part.raw_rank_ic.where(common)
    for prefix, series in [("raw_rank_ic", part.raw_rank_ic), ("partial_rank_ic", part.partial_rank_ic),
                           ("partial_minus_raw", paired)]:
        numeric(row[f"{prefix}_mean"], series.mean(), "partial_summary")
        numeric(row[f"{prefix}_n_dates"], series.notna().sum(), "partial_counts")
        if full_interval:
            check_interval(row, series, prefix, protocol)
    for status in ["defined", "zero_both_residuals", "zero_prediction_residual", "zero_return_residual"]:
        numeric(row[f"status_{status}_dates"], part.status.eq(status).sum(), "partial_counts")
    for column in ["prediction_rank_control_r_squared", "return_rank_control_r_squared"]:
        numeric(row[f"{column}_mean"], part[column].mean(), "partial_summary")


def independent_stage(stage, out, benchmark, panel, protocol):
    root = out / stage
    year = 2023 if stage == "validation" else 2024
    candidates = panel.loc[panel.date.dt.year.between(2021, year - 1)]
    evaluation = panel.loc[panel.date.dt.year.eq(year)]
    if year == 2023:
        evaluation = evaluation.loc[evaluation.exit_date.lt(pd.Timestamp("2024-01-01"))]
    boundary = evaluation.date.min()
    training = candidates.loc[candidates.exit_date.lt(boundary)].sort_values(KEYS).reset_index(drop=True)
    evaluation = evaluation.sort_values(KEYS).reset_index(drop=True)
    split = read_json(root / "split_manifest.json")
    require(split == read_json(benchmark / stage / "split_manifest.json"), f"{stage}: archived split differs")
    for role, frame in [("train", training), ("evaluation", evaluation)]:
        record = split[role]
        require(record["rows"] == len(frame) and record["signal_dates"] == frame.date.nunique(), f"{stage}: split counts")
        require(record["ordered_keys_sha256"] == key_digest(frame), f"{stage}: independent split keys")
        for field, expected in [("signal_start", frame.date.min()), ("signal_end", frame.date.max()),
                                ("min_label_entry", frame.entry_date.min()), ("max_label_exit", frame.exit_date.max())]:
            require(record[field] == str(expected.date()), f"{stage}: {field}")
    require(training.exit_date.max() < evaluation.date.min(), "Mature-label training boundary violated")
    require(not training.set_index(KEYS).index.intersection(evaluation.set_index(KEYS).index).size,
            "Training and evaluation overlap")
    categories = sorted(training.industry.unique())
    require(set(evaluation.industry).issubset(categories), "Unknown evaluation industry")
    train_dummies = pd.get_dummies(training.industry, dtype=float).reindex(columns=categories)
    eval_dummies = pd.get_dummies(evaluation.industry, dtype=float).reindex(columns=categories, fill_value=0)
    x_train = np.column_stack([train_dummies, np.log(training.market_cap)])
    x_eval = np.column_stack([eval_dummies, np.log(evaluation.market_cap)])
    training_mean = float(training.forward_5d.mean())
    saved_daily = load_csv(root / "daily_metrics.csv").set_index(["model", "variant", "date"])
    saved_partial = load_csv(root / "partial_rank_daily.csv").set_index(["model", "control", "date"])
    saved_exposure = load_csv(root / "exposure_daily.csv").set_index(["model", "date"])
    saved_industry = load_csv(root / "industry_exposure_daily.csv").set_index(["model", "date", "industry"])
    summaries = load_csv(root / "summary.csv").set_index(["model", "variant"])
    independent_partial, independent_daily = [], []
    tied_days = 0
    for model_name in MODELS:
        meta = read_json(root / f"{model_name}_CONTROLS_fit_metadata.json")
        require(meta["industry_categories"] == categories, "Training industry encoding differs")
        require(meta["feature_names"] == [f"industry__industry_{x}" for x in categories] + ["size__log_market_cap"],
                "Full one-hot feature ordering differs")
        numeric(meta["training_mean"], training_mean, "training_mean", 1e-14)
        require(meta["training_rows"] == len(training) and meta["evaluation_rows"] == len(evaluation), "Baseline sample sizes")
        if model_name == "Ridge":
            numeric(meta["x_scaler_mean"], x_train.mean(axis=0), "preprocessing")
            numeric(meta["x_scaler_var"], x_train.var(axis=0), "preprocessing")
            numeric(meta["x_scaler_scale"], x_train.std(axis=0), "preprocessing")
            scaler = StandardScaler().fit(x_train)
            estimator = Ridge(alpha=100).fit(scaler.transform(x_train), training.forward_5d)
            refit_predictions = estimator.predict(scaler.transform(x_eval))
        else:
            estimator = HistGradientBoostingRegressor(max_iter=55, learning_rate=.065, max_leaf_nodes=17,
                l2_regularization=.2, early_stopping=False, random_state=1021).fit(x_train, training.forward_5d)
            refit_predictions = estimator.predict(x_eval)
        numeric(load_csv(root / f"{model_name}_CONTROLS_predictions.csv.gz").prediction,
                refit_predictions, "independent_controls_refit", 2e-12)
        for variant in ["ALL20", "CONTROLS"]:
            path = root / f"{model_name}_{variant}_predictions.csv.gz"
            result = load_csv(path)
            assert_keys(result, evaluation, f"{stage}/{model_name}/{variant}")
            if variant == "ALL20":
                require(digest(path) == digest(benchmark / stage / f"{model_name}_predictions.csv.gz"), "ALL20 archive not byte-identical")
            for date_column in ["entry_date", "exit_date"]:
                require(result[date_column].equals(evaluation[date_column]), "Label endpoint changed")
            numeric(result.forward_5d, evaluation.forward_5d, "canonical_labels", protocol["labels_max_absolute_tolerance"])
            result["forward_5d"] = evaluation.forward_5d.to_numpy()
            result["industry"] = evaluation.industry.to_numpy()
            result["market_cap"] = evaluation.market_cap.to_numpy()
            y, p = result.forward_5d.to_numpy(), result.prediction.to_numpy()
            row = summaries.loc[(model_name, variant)]
            mse = float(np.mean((y - p) ** 2))
            baseline_mse = float(np.mean((y - training_mean) ** 2))
            for field, value in [("rows", len(result)), ("pearson", correlation(p, y)), ("rmse", math.sqrt(mse)),
                                 ("training_mean_baseline_rmse", math.sqrt(baseline_mse)),
                                 ("r2_against_training_mean", 1 - mse / baseline_mse)]:
                numeric(row[field], value, "prediction_summary")
            variant_daily = []
            for date, cross in result.groupby("date", sort=True):
                p, y = cross.prediction.to_numpy(), cross.forward_5d.to_numpy()
                rp, ry = rank(p), rank(y)
                low, high = tail_allocation(p, False), tail_allocation(p, True)
                metrics = {"n_assets": len(cross), "rank_ic": correlation(rp, ry),
                           "gross_decile_spread": float((high - low) @ y) if np.ptp(p) else np.nan,
                           "daily_rmse": math.sqrt(float(np.mean((p - y) ** 2)))}
                archived = saved_daily.loc[(model_name, variant, date)]
                for metric, value in metrics.items():
                    numeric(archived[metric], value, "daily_prediction_metrics")
                variant_daily.append({"date": date, **metrics})
                independent_daily.append({"model": model_name, "variant": variant, "date": date, **metrics})
                if variant != "ALL20":
                    continue
                logcap = np.log(cross.market_cap.to_numpy())
                rs = rank(logcap)
                groups, names = pd.factorize(cross.industry, sort=True)
                exposure = {"n_assets": len(cross), "score_logcap_spearman": correlation(rp, rs),
                            "low_log_market_cap": float(low @ logcap), "high_log_market_cap": float(high @ logcap),
                            "high_minus_low_log_market_cap": float((high-low) @ logcap),
                            "low_size_percentile": float(low @ (rs/len(rs))),
                            "high_size_percentile": float(high @ (rs/len(rs))),
                            "high_minus_low_size_percentile": float((high-low) @ (rs/len(rs)))}
                for kind in KINDS:
                    a, b = residual(rp, groups, rs, kind), residual(ry, groups, rs, kind)
                    tol = protocol["zero_residual"]["relative_tolerance"]
                    za = np.linalg.norm(a-a.mean()) <= tol * max(1.0, np.linalg.norm(rp-rp.mean()))
                    zb = np.linalg.norm(b-b.mean()) <= tol * max(1.0, np.linalg.norm(ry-ry.mean()))
                    status = "zero_both_residuals" if za and zb else "zero_prediction_residual" if za else "zero_return_residual" if zb else "defined"
                    computed = {"n_assets": len(cross), "raw_rank_ic": correlation(rp, ry),
                                "partial_rank_ic": correlation(a, b) if status == "defined" else np.nan,
                                "prediction_rank_control_r_squared": r_squared(rp, a),
                                "return_rank_control_r_squared": r_squared(ry, b)}
                    observed = saved_partial.loc[(model_name, kind, date)]
                    require(observed.status == status, f"{stage}/{model_name}/{kind}/{date}: residual status")
                    for metric, value in computed.items():
                        numeric(observed[metric], value, "independent_partial_daily")
                    independent_partial.append({"model": model_name, "control": kind, "date": date,
                                                "status": status, **computed})
                    exposure[f"score_{kind}_r_squared"] = r_squared(p, residual(p, groups, logcap, kind))
                for metric, value in exposure.items():
                    numeric(saved_exposure.loc[(model_name, date), metric], value, "daily_exposure")
                allocations = []
                for group, name in enumerate(names):
                    mask = groups == group
                    pool, lo, hi = float(mask.mean()), float(low[mask].sum()), float(high[mask].sum())
                    values = dict(pool_weight=pool, low_weight=lo, high_weight=hi,
                                  low_minus_pool_weight=lo-pool, high_minus_pool_weight=hi-pool, high_minus_low_weight=hi-lo)
                    for metric, value in values.items():
                        numeric(saved_industry.loc[(model_name, date, name), metric], value, "industry_tail_allocation")
                    allocations.append(list(values.values()))
                numeric(np.array(allocations).sum(axis=0), [1, 1, 1, 0, 0, 0], "industry_weight_conservation")
                tied_days += int(pd.Series(p).duplicated().any())
            days = pd.DataFrame(variant_daily).set_index("date")
            for metric in ["rank_ic", "gross_decile_spread", "daily_rmse"]:
                check_interval(row, days[metric], metric, protocol)
        print(f"{stage}: {model_name} independent refit, daily projections and exposures verified", flush=True)
    parts = pd.DataFrame(independent_partial)
    require(len(parts) == len(saved_partial), "Unexpected partial rows")
    for filename, quarterly in [("partial_rank_summary.csv", False), ("partial_rank_quarterly.csv", True)]:
        for _, row in load_csv(root / filename).iterrows():
            selected = parts.loc[parts.model.eq(row.model) & parts.control.eq(row.control)]
            if quarterly:
                selected = selected.loc[selected.date.dt.to_period("Q").astype(str).eq(row.quarter)]
            check_partial_summary(row, selected, protocol, full_interval=not quarterly)
    daily = pd.DataFrame(independent_daily)
    require(len(daily) == len(saved_daily), "Unexpected daily metric rows")
    for _, row in load_csv(root / "paired_baseline_comparison.csv").iterrows():
        data = daily.loc[daily.model.eq(row.model)].pivot(index="date", columns="variant", values=row.metric)
        common = data.ALL20.notna() & data.CONTROLS.notna()
        numeric(row.all20_on_common_mean, data.ALL20[common].mean(), "baseline_pairing")
        numeric(row.controls_on_common_mean, data.CONTROLS[common].mean(), "baseline_pairing")
        check_interval(row, data.ALL20-data.CONTROLS, "", protocol)
    return dict(training_rows=len(training), evaluation_rows=len(evaluation), dates=evaluation.date.nunique(),
                partial_daily_rows=len(parts), baseline_daily_rows=len(daily),
                all20_days_with_tied_scores=tied_days, controls_models_independently_refitted=2)


def audit(out):
    started = datetime.now(timezone.utc).isoformat()
    require(not (out / "verification.json").exists(), "verification.json already exists; audit does not overwrite evidence")
    root_manifest = read_json(out / "run_manifest.json")
    require(root_manifest["status"] == "completed", "Study is not completed")
    root_manifest_hash = digest(out / "run_manifest.json")
    frozen = read_json(out / "protocol.json")
    protocol = frozen["protocol"]
    protocol_hash = object_digest(protocol)
    require(protocol_hash == frozen["protocol_sha256"] == root_manifest["protocol_sha256"], "Frozen protocol mismatch")
    require(object_digest(protocol["model_parameters"]) == protocol["parameters_sha256"], "Parameters hash mismatch")
    for name, path in protocol["input_paths"].items():
        require(digest(path) == protocol["inputs_sha256"][name], f"Input changed: {name}")
    for name, expected in protocol["code_sha256"].items():
        require(digest(PROJECT / name) == expected, f"Source changed: {name}")
    require(root_manifest["code_sha256"] == protocol["code_sha256"], "Root source manifest mismatch")
    require(root_manifest["inputs_sha256"] == protocol["inputs_sha256"], "Root input manifest mismatch")
    verify_hashes(out, root_manifest)
    benchmark = Path(protocol["input_paths"]["benchmark_protocol"]).parent
    bfrozen = read_json(benchmark / "protocol.json")
    require(object_digest(bfrozen["protocol"]) == bfrozen["protocol_sha256"] == protocol["benchmark_protocol_sha256"],
            "Benchmark protocol integrity")
    for stage in ["validation", "confirmation"]:
        manifest = read_json(out / stage / "run_manifest.json")
        require(manifest["status"] == "completed" and manifest["protocol_sha256"] == protocol_hash, "Stage not completed")
        require(manifest["code_sha256"] == protocol["code_sha256"] and manifest["inputs_sha256"] == protocol["inputs_sha256"],
                "Stage provenance mismatch")
        verify_hashes(out / stage, manifest)
        bmanifest = read_json(benchmark / stage / "run_manifest.json")
        require(bmanifest["status"] == "completed" and bmanifest["protocol_sha256"] == bfrozen["protocol_sha256"], "Benchmark status")
        verify_hashes(benchmark, bmanifest)
        for name, expected in bmanifest["code_sha256"].items():
            require(digest(PROJECT / name) == expected, f"Benchmark source changed: {name}")
    panel = load_csv(protocol["input_paths"]["pit_panel"])
    risk = load_csv(protocol["input_paths"]["risk"])[KEYS + ["industry", "market_cap"]]
    require(not panel.duplicated(KEYS).any() and not risk.duplicated(KEYS).any(), "Input duplicate keys")
    require(not risk.industry.isna().any() and not risk.industry.str.strip().eq("").any(), "Missing industry")
    require(np.isfinite(risk.market_cap).all() and risk.market_cap.gt(0).all(), "Invalid market cap")
    enriched = panel.merge(risk, on=KEYS, how="left", validate="one_to_one", sort=False, indicator=True)
    require(enriched._merge.eq("both").all(), "Missing risk record")
    assert_keys(enriched, panel, "Risk merge")
    records = {stage: independent_stage(stage, out, benchmark, enriched, protocol)
               for stage in ["validation", "confirmation"]}
    # Confirm all original evidence stayed unchanged throughout this independent audit.
    verify_hashes(out, root_manifest)
    require(digest(out / "run_manifest.json") == root_manifest_hash, "Study manifest changed during audit")
    verification = {
        "status": "passed", "started_at_utc": started, "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "audit_source_sha256": digest(Path(__file__)), "study_run_manifest_sha256": root_manifest_hash,
        "protocol_sha256": protocol_hash, "command": sys.orig_argv, "stages": records,
        "verified_input_files": len(protocol["input_paths"]), "verified_study_source_files": len(protocol["code_sha256"]),
        "checks": CHECKS, "max_absolute_errors": ERRORS,
        "independent_methods": {
            "partial_rank_ic": "Average ranks, industry group demeaning and Frisch-Waugh scalar size projection; no study statistics imported",
            "controls_baseline": "pandas full dummy design and direct sklearn Ridge/HGB refits, four models",
            "tails": "Sequential score-level allocation with equal sharing of cutoff ties",
            "intervals": "Independent full-calendar 20-day 2000-draw moving-block reproduction for annual prediction, partial and paired comparisons",
            "quarterly": "Partial means, common-date counts, undefined statuses and projection R-squared checked; quarterly confidence intervals not independently recomputed",
        },
        "scope": "Historical observed-label member panel; conditional rank diagnostics, not causal attribution or executable net returns",
    }
    with (out / "verification.json").open("x") as handle:
        json.dump(verification, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(verification, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=PROJECT / "outputs" / "industry_size_attribution")
    audit(parser.parse_args().out)
