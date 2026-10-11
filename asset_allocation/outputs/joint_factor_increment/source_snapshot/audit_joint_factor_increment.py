"""Independent model, prediction-statistics and cash-ledger increment audit.

The production joint model, preprocessing, simulation and statistics helpers
are not imported. This audit reuses the existing independent ledger audit and
independent statistical audit. Verification is written once after all checks.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from audit_industry_size_attribution import correlation, interval, rank, tail_allocation
from audit_rebalance_study import (
    DEFAULT_ROOT, END, PROJECT, START, audit_run, check_fields, equal,
    prepare_context, sha256, verify_hashes,
)


KEYS = ["date", "asset_id"]
FACTORS = [f"factor_{i:02d}" for i in range(1, 21)]
DELAYED = ["factor_03", "factor_12"]
MODELS = ["Ridge", "HGB"]
ARMS = [f"{model}_{variant}" for model in MODELS for variant in ["CONTROLS", "JOINT"]]
RULES = ["NET", "BOTH", "BUDGET20"]
METRICS = ["rank_ic", "daily_rmse", "gross_decile_spread"]
BOOTSTRAP = {"bootstrap": {"block_days": 20, "draws": 2000, "seed": 1021}}
_WORKER = {}


def read_json(path):
    return json.loads(Path(path).read_text())


def key_digest(frame):
    return hashlib.sha256(frame[KEYS].to_csv(index=False).encode()).hexdigest()


def independent_panel(root, market, universe):
    """Align delayed factor snapshots by release-date search within each asset."""
    factors = pd.read_csv(root / "data" / "factors.csv.gz", parse_dates=["date"])
    availability = pd.read_csv(root / "data" / "factor_availability.csv.gz",
                               parse_dates=["date"] + [f"{x}_available_on" for x in DELAYED])
    reference = market[KEYS].set_index(KEYS, verify_integrity=True).index.sort_values()
    for frame in [factors, availability, universe]:
        pd.testing.assert_index_equal(frame.set_index(KEYS, verify_integrity=True).index.sort_values(), reference)
    assert availability.published_at.eq("after_close").all()
    factors = factors.merge(availability[KEYS + [f"{x}_available_on" for x in DELAYED]],
                            on=KEYS, validate="one_to_one").sort_values(["asset_id", "date"])
    for factor in DELAYED:
        available = f"{factor}_available_on"
        assert factors[available].notna().all() and factors[available].ge(factors.date).all()
        assert factors.groupby("asset_id")[available].diff().dropna().ge(pd.Timedelta(0)).all()
        aligned = pd.Series(np.nan, index=factors.index)
        for _, group in factors.groupby("asset_id", sort=False):
            location = np.searchsorted(group[available].to_numpy(), group.date.to_numpy(), side="right") - 1
            has_release = location >= 0
            values = np.full(len(group), np.nan)
            values[has_release] = group[factor].to_numpy()[location[has_release]]
            aligned.loc[group.index] = values
        factors[factor] = aligned
    panel = universe.loc[universe.is_listed & universe.is_member, KEYS].merge(
        factors[KEYS + FACTORS], on=KEYS, validate="one_to_one").sort_values(KEYS).reset_index(drop=True)
    panel["entry_date"], panel["exit_date"] = panel.date + BDay(1), panel.date + BDay(6)
    opening = market.set_index(KEYS, verify_integrity=True).open
    entries = pd.MultiIndex.from_arrays([panel.entry_date, panel.asset_id], names=KEYS)
    exits = pd.MultiIndex.from_arrays([panel.exit_date, panel.asset_id], names=KEYS)
    panel["forward_5d"] = opening.reindex(exits).to_numpy() / opening.reindex(entries).to_numpy() - 1
    risk = pd.read_csv(root / "data" / "risk_exposure.csv.gz", parse_dates=["date"])
    panel = panel.merge(risk[KEYS + ["industry", "market_cap"]], on=KEYS,
                        how="left", sort=False, validate="one_to_one")
    assert panel.industry.notna().all() and np.isfinite(panel.market_cap).all() and panel.market_cap.gt(0).all()
    panel["log_market_cap"] = np.log(panel.market_cap)
    return panel


def annual_split(panel, year, saved):
    candidates = panel.loc[panel.date.dt.year.between(2021, year - 1)]
    evaluation = panel.loc[panel.date.dt.year.eq(year)]
    if year == 2023:
        evaluation = evaluation.loc[evaluation.exit_date.lt(pd.Timestamp("2024-01-01"))]
    training = candidates.loc[candidates.exit_date.lt(evaluation.date.min())]
    training = training.sort_values(KEYS).reset_index(drop=True)
    evaluation = evaluation.sort_values(KEYS).reset_index(drop=True)
    assert training.exit_date.max() < evaluation.date.min()
    assert saved["stage"] == ("validation" if year == 2023 else "confirmation")
    for role, frame in [("train", training), ("evaluation", evaluation)]:
        record = saved[role]
        assert record["ordered_keys_sha256"] == key_digest(frame)
        assert record["rows"] == len(frame) and record["signal_dates"] == frame.date.nunique()
        for label, value in [("signal_start", frame.date.min()), ("signal_end", frame.date.max()),
                             ("min_label_entry", frame.entry_date.min()), ("max_label_exit", frame.exit_date.max())]:
            assert record[label] == str(value.date())
        assert record["factor_missing_rows"] == frame[FACTORS].isna().sum().astype(int).to_dict()
    assert saved["train_candidates"] == len(candidates)
    assert saved["eval_candidates"] == panel.date.dt.year.eq(year).sum()
    assert saved["purged_training_rows"] == len(candidates) - len(training)
    assert saved["purged_evaluation_rows"] == saved["eval_candidates"] - len(evaluation)
    assert saved["training_exit_strictly_before_first_evaluation_signal"] is True
    assert saved["validation_exit_strictly_before_2024"] == (year == 2023)
    return training, evaluation


def model_design(frame, categories, medians=None):
    assert set(frame.industry).issubset(categories)
    dummy = pd.get_dummies(frame.industry, dtype=float).reindex(columns=categories, fill_value=0.)
    parts = [dummy.to_numpy(), frame[["log_market_cap"]].to_numpy()]
    if medians is not None:
        parts.append(frame[FACTORS].fillna(medians).to_numpy())
    design = np.column_stack(parts)
    assert np.isfinite(design).all()
    return design


def verify_predictions(root, out, protocol, market, universe):
    full = independent_panel(root, market, universe)
    canonical = pd.read_csv(protocol["panel_path"], parse_dates=["date", "entry_date", "exit_date"])
    matched = canonical[KEYS].merge(full, on=KEYS, how="left", validate="one_to_one", sort=False)
    for column in ["entry_date", "exit_date"]:
        pd.testing.assert_series_equal(canonical[column], matched[column], check_names=False)
    equal(canonical[FACTORS + ["forward_5d"]], matched[FACTORS + ["forward_5d"]], "canonical PIT factors/labels")
    panel = matched
    source = Path(protocol["portfolio_source"])
    predictions = {arm: pd.read_csv(out / "predictions" / f"{arm}.csv.gz",
                                   parse_dates=["date", "entry_date", "exit_date", "train_cutoff"])
                   for arm in ARMS}
    members = full.loc[full.date.between(START, END)].reset_index(drop=True)
    coverage = pd.read_csv(out / "signal_coverage.csv").set_index(["predictor", "year"], verify_integrity=True)
    fits = pd.read_csv(out / "fits.csv").set_index(["predictor", "year"], verify_integrity=True)
    assert set(coverage.index) == set(fits.index) == {(arm, year) for arm in ARMS for year in [2023, 2024]}
    for arm, frame in predictions.items():
        assert not frame.duplicated(KEYS).any() and np.isfinite(frame.prediction).all()
        pd.testing.assert_frame_equal(frame[KEYS], members[KEYS])
        for column in ["entry_date", "exit_date"]:
            pd.testing.assert_series_equal(frame[column], members[column], check_names=False)
        equal(frame.forward_5d, members.forward_5d, f"{arm} full future labels")
        cutoff = pd.to_datetime(np.where(frame.date.dt.year.eq(2023), "2022-12-30", "2023-12-29"))
        assert np.array_equal(frame.train_cutoff, cutoff)
        if arm.endswith("_CONTROLS"):
            archive = pd.read_csv(source / "predictions" / f"{arm}.csv.gz",
                                  parse_dates=["date", "entry_date", "exit_date", "train_cutoff"])
            pd.testing.assert_frame_equal(frame, archive, check_exact=False, rtol=1e-12, atol=1e-14)
    records, evaluations, training_means = [], {}, {}
    for year, stage in [(2023, "validation"), (2024, "confirmation")]:
        saved_split = read_json(out / "splits" / f"{stage}.json")
        assert saved_split == read_json(out / "prediction_evaluation" / stage / "split_manifest.json")
        training, evaluation = annual_split(panel, year, saved_split)
        signals = members.loc[members.date.dt.year.eq(year)].reset_index(drop=True)
        assert training.exit_date.max() < signals.date.min()
        categories = sorted(training.industry.unique())
        medians = training[FACTORS].median()
        assert np.isfinite(medians).all()
        evaluations[year], training_means[year] = evaluation, float(training.forward_5d.mean())
        for arm in ARMS:
            name, variant = arm.split("_")
            meta = read_json(out / "models" / f"{year}_{arm}_fit_metadata.json")
            fit = fits.loc[(arm, year)]
            expected_features = ["industry", "log_market_cap"] + (FACTORS if variant == "JOINT" else [])
            assert meta["input_features"] == expected_features
            assert meta["industry_categories"] == categories
            expected_names = [f"industry__industry_{x}" for x in categories] + ["size__log_market_cap"]
            if variant == "JOINT":
                expected_names += [f"factors__{x}" for x in FACTORS]
            assert meta["feature_names"] == expected_names
            assert meta["training_rows"] == fit.training_rows == len(training)
            assert meta["training_keys_sha256"] == fit.training_keys_sha256 == key_digest(training)
            assert meta["max_training_label_exit"] == fit.max_training_label_exit == str(training.exit_date.max().date())
            equal(meta["training_mean"], training_means[year])
            yearly = predictions[arm].loc[predictions[arm].date.dt.year.eq(year)].reset_index(drop=True)
            saved = coverage.loc[(arm, year)]
            check_fields(saved, dict(signal_rows=len(yearly), signal_dates=yearly.date.nunique(),
                                      missing_labels=yearly.forward_5d.isna().sum()))
            assert saved.signal_keys_sha256 == key_digest(yearly)
            path = (out if variant == "JOINT" else source) / "models" / f"{year}_{arm}.joblib"
            if variant == "CONTROLS":
                assert Path(meta["source_model"]).resolve() == path.resolve()
                assert meta["source_model_sha256"] == sha256(path)
            model = joblib.load(path)
            preprocessing = model.named_steps["preprocessing"]
            assert list(model.feature_names_in_) == expected_features
            assert preprocessing.named_transformers_["industry"].handle_unknown == "error"
            assert list(preprocessing.named_transformers_["industry"].categories_[0]) == categories
            train_x = model_design(training, categories, medians if variant == "JOINT" else None)
            signal_x = model_design(signals, categories, medians if variant == "JOINT" else None)
            equal(preprocessing.transform(training[expected_features]), train_x, "independent training design")
            equal(preprocessing.transform(signals[expected_features]), signal_x, "independent signal design")
            if variant == "JOINT":
                equal([meta["factor_imputation_medians"][x] for x in FACTORS], medians)
                equal(preprocessing.named_transformers_["factors"].statistics_, medians)
            saved_learner = model.named_steps["model"]
            if name == "Ridge":
                assert saved_learner.alpha == 100
                scaler = StandardScaler().fit(train_x)
                for field in ["mean", "var", "scale"]:
                    equal(meta[f"x_scaler_{field}"], getattr(scaler, field + "_"))
                    equal(getattr(model.named_steps["scaler"], field + "_"), getattr(scaler, field + "_"))
                train_x, signal_x = scaler.transform(train_x), scaler.transform(signal_x)
                estimator = Ridge(alpha=100)
            else:
                parameters = dict(max_iter=55, learning_rate=.065, max_leaf_nodes=17,
                                  l2_regularization=.2, early_stopping=False, random_state=1021)
                assert all(saved_learner.get_params()[key] == value for key, value in parameters.items())
                assert saved_learner.n_iter_ == meta["fitted_iterations"] == 55
                assert not saved_learner.do_early_stopping_ and meta["early_stopping_used"] is False
                estimator = HistGradientBoostingRegressor(**parameters)
            direct = saved_learner.predict(signal_x)
            equal(yearly.prediction, direct, "saved estimator with independent design")
            record = dict(year=year, predictor=arm, training_rows=len(training),
                          training_keys_and_maturity_checked=True, preprocessing_training_only_checked=True,
                          full_member_predictions_checked=True,
                          max_independent_design_prediction_error=float(np.max(np.abs(yearly.prediction - direct))))
            if variant == "JOINT":
                print(f"Independent refit: {year} {arm}", flush=True)
                estimator.fit(train_x, training.forward_5d)
                refit = estimator.predict(signal_x)
                equal(yearly.prediction, refit, "independent JOINT refit")
                record.update(independent_refit_checked=True,
                              max_independent_refit_error=float(np.max(np.abs(yearly.prediction - refit))))
            else:
                record["identical_archived_control_predictions_checked"] = True
            records.append(record)
    return predictions, evaluations, training_means, records


def verify_prediction_statistics(out, predictions, evaluations, training_means):
    base = out / "prediction_evaluation"
    summaries = pd.read_csv(base / "summary.csv").set_index(["year", "predictor"], verify_integrity=True)
    saved_daily = pd.read_csv(base / "daily_metrics.csv", parse_dates=["date"]).set_index(
        ["year", "predictor", "date"], verify_integrity=True).sort_index()
    paired = pd.read_csv(base / "paired_comparison.csv").set_index(["year", "model", "metric"], verify_integrity=True)
    assert set(summaries.index) == {(year, arm) for year in evaluations for arm in ARMS}
    assert set(paired.index) == {(year, model, metric) for year in evaluations for model in MODELS for metric in METRICS}
    reconstructed = {}
    for year, evaluation in evaluations.items():
        for arm in ARMS:
            joined = evaluation[KEYS + ["forward_5d"]].merge(predictions[arm][KEYS + ["prediction"]],
                    on=KEYS, how="left", sort=False, validate="one_to_one")
            assert np.isfinite(joined.prediction).all()
            y, p = joined.forward_5d.to_numpy(), joined.prediction.to_numpy()
            mse, baseline = np.mean((y - p) ** 2), np.mean((y - training_means[year]) ** 2)
            summary = summaries.loc[(year, arm)]
            check_fields(summary, dict(rows=len(joined), pearson=correlation(p, y), rmse=np.sqrt(mse),
                         training_mean_baseline_rmse=np.sqrt(baseline), r2_against_training_mean=1 - mse / baseline))
            daily = []
            for date, cross in joined.groupby("date", sort=True):
                p, y = cross.prediction.to_numpy(), cross.forward_5d.to_numpy()
                weights = tail_allocation(p, True) - tail_allocation(p, False)
                record = dict(n_assets=len(cross), rank_ic=correlation(rank(p), rank(y)),
                              daily_rmse=np.sqrt(np.mean((p - y) ** 2)),
                              gross_decile_spread=float(weights @ y) if np.ptp(p) else np.nan)
                check_fields(saved_daily.loc[(year, arm, date)], record)
                daily.append(dict(date=date, **record))
            frame = pd.DataFrame(daily).set_index("date")
            assert set(saved_daily.loc[(year, arm)].index) == set(frame.index)
            reconstructed[(year, arm)] = frame
            for metric in METRICS:
                check_fields(summary, {f"{metric}_{key}": value for key, value in interval(frame[metric], BOOTSTRAP).items()})
        for model in MODELS:
            for metric in METRICS:
                joint = reconstructed[(year, f"{model}_JOINT")][metric]
                control = reconstructed[(year, f"{model}_CONTROLS")][metric]
                row = paired.loc[(year, model, metric)]
                assert row.comparison == "JOINT minus CONTROLS"
                common = joint.notna() & control.notna()
                check_fields(row, dict(joint_on_common_mean=joint.loc[common].mean(),
                                       controls_on_common_mean=control.loc[common].mean(),
                                       **interval(joint - control, BOOTSTRAP)))
    for kind in ["summary", "paired_comparison", "daily_metrics"]:
        combined = pd.read_csv(base / f"{kind}.csv")
        stages = pd.concat([pd.read_csv(base / stage / f"{kind}.csv") for stage in ["validation", "confirmation"]], ignore_index=True)
        pd.testing.assert_frame_equal(combined, stages)
    return dict(yearly_prediction_summary_rows=len(summaries), paired_prediction_rows=len(paired),
                daily_prediction_rows=len(saved_daily), independent_block_intervals_checked=True)


def verify_portfolio_pairs(out):
    paired = pd.read_csv(out / "paired_portfolio.csv", dtype={"period": str}).set_index(
        ["rule", "model", "cost_multiplier", "period", "metric"], verify_integrity=True)
    metric_names = ["daily_return_difference", "daily_log_relative_growth"]
    expected = {(rule, model, cost, period, metric) for rule in RULES for model in MODELS
                for cost in [0, 1, 2] for period in ["2023", "2024"] for metric in metric_names}
    assert set(paired.index) == expected
    for rule in RULES:
        for model in MODELS:
            for cost in [0, 1, 2]:
                frames = {variant: pd.read_csv(out / "runs" / f"{rule}__{model}_{variant}__cost{cost}" / "daily.csv",
                                              parse_dates=["date"]).set_index("date").daily_return
                          for variant in ["JOINT", "CONTROLS"]}
                pd.testing.assert_index_equal(frames["JOINT"].index, frames["CONTROLS"].index)
                differences = [frames["JOINT"] - frames["CONTROLS"],
                               np.log1p(frames["JOINT"]) - np.log1p(frames["CONTROLS"])]
                for period in ["2023", "2024"]:
                    for metric, difference in zip(metric_names, differences):
                        row = paired.loc[(rule, model, cost, period, metric)]
                        assert row.policy == ("NET" if rule == "BUDGET20" else rule)
                        assert row.predictor == f"{model}_JOINT" and row.reference_predictor == f"{model}_CONTROLS"
                        check_fields(row, interval(difference.loc[difference.index.year == int(period)], BOOTSTRAP))
    return dict(paired_portfolio_rows=len(paired), independent_block_intervals_checked=True)


def _initialize(context, periods, out):
    _WORKER.update(context=context, periods=periods, out=out)


def _audit_record(record):
    row = next(pd.DataFrame([record]).itertuples(index=False))
    return audit_run(row, _WORKER["out"] / "runs" / row.run, _WORKER["context"],
                     _WORKER["periods"], max_review_turnover=.2 if row.rule == "BUDGET20" else None)


def run(root: Path, out: Path):
    destination = out / "verification.json"
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite verification: {destination}")
    protocol, manifest = read_json(out / "protocol.json"), read_json(out / "run_manifest.json")
    verify_hashes(out, protocol, manifest)
    assert Path(protocol["root"]).resolve() == root.resolve()
    market = pd.read_csv(root / "data" / "market_daily.csv.gz", parse_dates=["date"])
    constraints = pd.read_csv(root / "data" / "trading_constraints.csv.gz", parse_dates=["date"])
    universe = pd.read_csv(root / "data" / "universe.csv.gz", parse_dates=["date"])
    predictions, evaluations, means, model_checks = verify_predictions(root, out, protocol, market, universe)
    prediction_stats = verify_prediction_statistics(out, predictions, evaluations, means)
    context = prepare_context(market, constraints, universe, predictions)
    planned = pd.DataFrame(protocol["runs"])
    assert len(planned) == 36 and not planned.run.duplicated().any()
    assert set(planned[["rule", "predictor", "cost_multiplier"]].itertuples(index=False, name=None)) == {
        (rule, arm, cost) for rule in RULES for arm in ARMS for cost in [0, 1, 2]}
    for row in planned.itertuples(index=False):
        assert row.run == f"{row.rule}__{row.predictor}__cost{int(row.cost_multiplier)}"
        assert row.policy == ("NET" if row.rule == "BUDGET20" else row.rule)
        equal(row.max_review_turnover, .2 if row.rule == "BUDGET20" else np.nan)
    summary = pd.read_csv(out / "summary.csv")
    pd.testing.assert_frame_equal(summary[planned.columns].sort_values("run").reset_index(drop=True),
                                  planned.sort_values("run").reset_index(drop=True), check_dtype=False)
    periods = pd.read_csv(out / "period_metrics.csv", dtype={"period": str}).set_index(["run", "period"], verify_integrity=True)
    assert set(periods.index) == {(name, period) for name in planned.run for period in ["FULL", "2023", "2024"]}
    for row in planned.itertuples(index=False):
        for period in ["FULL", "2023", "2024"]:
            saved = periods.loc[(row.run, period)]
            assert saved.rule == row.rule and saved.policy == row.policy and saved.predictor == row.predictor
            equal(saved.cost_multiplier, row.cost_multiplier)
            equal(saved.max_review_turnover, row.max_review_turnover)
    records = []
    with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context("spawn"),
                             initializer=_initialize, initargs=(context, periods, out)) as pool:
        futures = [pool.submit(_audit_record, record) for record in summary.to_dict("records")]
        for future in as_completed(futures):
            result = future.result()
            records.append(result)
            print(f"Independent increment audit {len(records)}/{len(futures)}: {result['run']}", flush=True)
    controls_compared = 0
    for row in planned.itertuples(index=False):
        if row.rule != "BUDGET20" and row.predictor.endswith("_CONTROLS"):
            current = pd.read_csv(out / "runs" / row.run / "daily.csv")
            archived = pd.read_csv(Path(protocol["rebalance_source"]) / "runs" / row.run / "daily.csv")
            pd.testing.assert_frame_equal(current, archived, check_exact=False, rtol=1e-9, atol=1e-10)
            controls_compared += 1
    portfolio_stats = verify_portfolio_pairs(out)
    verify_hashes(out, protocol, manifest)
    result = dict(status="passed", verified_at_utc=datetime.now(timezone.utc).isoformat(),
                  input_code_output_hashes_checked=True, model_checks=model_checks,
                  prediction_statistics=prediction_stats, portfolio_statistics=portfolio_stats,
                  audited_runs=sorted(records, key=lambda x: x["run"]),
                  archived_unbudgeted_control_accounts_compared=controls_compared,
                  scope="PIT inputs, mature-label splits, saved preprocessing, joint-model refits, predictions, "
                        "prediction statistics, 36 independent cash/position/plan accounts, review budgets, "
                        "annual/full account metrics and 72 paired portfolio intervals. Separate risk-exposure "
                        "and budget-diagnostics summary tables are outside this audit.",
                  run_manifest_sha256=sha256(out / "run_manifest.json"), audit_code_sha256=sha256(Path(__file__)))
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"Verified {len(records)} accounts and model/statistical increment checks; saved {destination}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out", type=Path, default=PROJECT / "outputs" / "joint_factor_increment")
    args = parser.parse_args()
    run(args.root, args.out)
