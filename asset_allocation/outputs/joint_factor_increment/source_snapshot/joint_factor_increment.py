"""Direct historical increment of adding PIT factors to industry and size.

Freeze one model design and three trading rules before fitting.  The budgeted
rule caps close-planned turnover, not realized opening turnover or risk exposure.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing
import os
from pathlib import Path
import platform
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn

from compare_models import PROJECT, jsonable, moving_block_interval, object_hash, sha256, utc_now, write_json
from factor_timing import DEFAULT_DATA
from industry_size_attribution import make_control_model, validate_risk_values
from joint_factor_model import ARMS, MODELS, evaluate_predictions, fit_joint_predictions, make_joint_model
from portfolio_model_comparison import COMPRESSION, END, START, period_statistics
from rebalance_ledger import simulate_rebalance
from rebalance_study import exposure_diagnostics


RULES = {"NET": ("NET", None), "BOTH": ("BOTH", None), "BUDGET20": ("NET", .20)}
COSTS = (0, 1, 2)
SCOPE = "Direct nested feature comparison on previously inspected 2023/2024 synthetic history; no parameter selection or independent new sample."
_WORKER = {}


def run_specs():
    return [dict(run=f"{rule}__{arm}__cost{cost}", rule=rule, policy=policy,
                 predictor=arm, cost_multiplier=cost, max_review_turnover=budget)
            for rule, (policy, budget) in RULES.items() for arm in ARMS for cost in COSTS]


def build_protocol(root, panel_path, portfolio_source, rebalance_source):
    source_manifest = json.loads((portfolio_source / "run_manifest.json").read_text())
    rebalance_manifest = json.loads((rebalance_source / "run_manifest.json").read_text())
    if source_manifest["status"] != "completed" or rebalance_manifest["status"] != "completed":
        raise ValueError("Prerequisite studies have not completed")
    paths = [root / "data" / f"{name}.csv.gz" for name in [
        "market_daily", "factors", "universe", "factor_availability", "trading_constraints", "risk_exposure"]]
    paths += [root / "data" / "data_dictionary.md", panel_path, panel_path.parent / "run_manifest.json"]
    for path in paths:
        expected = source_manifest["inputs_sha256"][str(path.resolve())]
        if sha256(path) != expected:
            raise ValueError(f"Input differs from the archived complete-universe study: {path}")
    for source, manifest, relatives in [
        (portfolio_source, source_manifest, ["protocol.json", "fits.csv"]
         + [f"models/{year}_{name}_CONTROLS.joblib" for year in (2023, 2024) for name in MODELS]
         + [f"predictions/{name}_CONTROLS.csv.gz" for name in MODELS]),
        (rebalance_source, rebalance_manifest, ["protocol.json"]
         + [f"runs/{rule}__{name}_CONTROLS__cost{cost}/daily.csv"
            for rule in ("NET", "BOTH") for name in MODELS for cost in COSTS]),
    ]:
        for relative in relatives:
            path = source / relative
            if sha256(path) != manifest["outputs_sha256"][relative]:
                raise ValueError(f"Archived result changed: {path}")
            paths.append(path)
        paths.append(source / "run_manifest.json")
        verification = source / "verification.json"
        if json.loads(verification.read_text())["status"] != "passed":
            raise ValueError(f"Prerequisite audit did not pass: {verification}")
        paths.append(verification)
    code_names = ["joint_factor_increment.py", "test_joint_factor_increment.py", "joint_factor_model.py", "test_joint_factor_model.py",
                  "audit_joint_factor_increment.py", "rebalance_ledger.py", "test_rebalance_ledger.py",
                  "audit_rebalance_study.py", "audit_industry_size_attribution.py", "rebalance_study.py", "portfolio_ledger.py",
                  "portfolio_model_comparison.py", "industry_size_attribution.py", "compare_models.py",
                  "factor_timing.py", "temporal_model.py"]
    return {
        "frozen_at_utc": utc_now(), "scope": SCOPE, "root": str(root.resolve()),
        "panel_path": str(panel_path.resolve()), "portfolio_source": str(portfolio_source.resolve()),
        "rebalance_source": str(rebalance_source.resolve()), "runs": run_specs(),
        "primary_predictor": "Ridge", "secondary_predictor": "HGB", "primary_period": "2024",
        "primary_cost_multiplier": 1, "descriptive_periods": ["2023", "FULL"],
        "primary_prediction_metric": "paired mean daily RankIC of JOINT minus CONTROLS",
        "primary_portfolio_comparison": "BUDGET20 Ridge_JOINT minus Ridge_CONTROLS at cost1 in 2024",
        "secondary_portfolio_rules": ["NET", "BOTH"],
        "inputs": "CONTROLS = industry dummies + natural log market_cap; JOINT appends all 20 PIT factors",
        "training": "Same mature annual training keys as archived controls: 2021-2022 for 2023; 2021-2023 for 2024; no tuning, factor selection, or outcome-dependent signal exclusion",
        "preprocessing": "Industry vocabulary from training; JOINT factor missing values filled with training medians; Ridge scales the full design using training statistics; labels never imputed",
        "model_parameters": {f"{name}_{variant}": jsonable((make_joint_model(name) if variant == "JOINT" else make_control_model(name)).get_params(deep=True))
                             for name in MODELS for variant in ("CONTROLS", "JOINT")},
        "signal_universe": "All listed members on the signal date, including unobservable future labels",
        "evaluation_universe": "Canonical common labelled temporal holdout; 2023 label exits precede 2024; trading does not use this filter",
        "risk_availability": "Same-date industry and cap assumed known after signal close; no independent publication timestamps supplied",
        "risk_constraints": "Common long-only unlevered cash accounts, five independent sleeves, top-decile equal-weight ideals, symmetric cutoff ties, no borrowing; no imposed industry/size neutrality or equal realized risk",
        "rules": {
            "NET": "Unchanged top-decile equal-weight targets; target minus actual shares",
            "BOTH": "Unchanged top10% entry/top20% actual-holding retention; +/-10% relative weight bands",
            "BUDGET20": "NET ideal; at each sleeve review interpolate from actual close weights toward ideal so sum(abs(effective - actual)) <= 0.20 of sleeve close NAV; one fixed budget, no search",
        },
        "budget_scope": "Close-planned two-sided turnover per five-day sleeve review; initialization is capped; terminal zero-target liquidation is exempt. Opening gaps, cost-inclusive buy scaling and blocked-order retries can change realized turnover; realized turnover is reported, not asserted equal or capped.",
        "execution": "Freeze signal-close weights and NAV; next open min(signal NAV, opening NAV) budget; sells first, common fee-inclusive buy scale, blocked slots retained as cash, no borrowing or cross-sleeve netting",
        "costs_permissions": "Previous-business-day fee + half spread + standard-order impact; previous-day can_buy/can_sell; actual flags audit only; costs 0/1/2 each follow their own holdings",
        "missing_quote_terminal": "Preserve existing zero-recovery writeoff on first missing quote; stop entries six business days before 2024-12-31; normal sleeve terminal reviews attempt liquidation; blocked positions marked at close",
        "risk_diagnostics": "Daily cash, concentration, breadth, industry allocation versus member equal weight, and weighted log cap; same rules/budget do not imply equal realized exposures",
        "uncertainty": "Daily paired arithmetic return and log relative growth, 20 consecutive business-day blocks, 2000 draws, seed1021; 2023/2024 separate; reference intervals without multiple-comparison adjustment",
        "inputs_sha256": {str(path.resolve()): sha256(path) for path in paths},
        "code_sha256": {str((PROJECT / name).resolve()): sha256(PROJECT / name) for name in code_names},
    }


def _initialize(root, out, rebalance_source):
    root, out = Path(root), Path(out)
    market = pd.read_csv(root / "data" / "market_daily.csv.gz", usecols=["date", "asset_id", "open", "close"], parse_dates=["date"])
    constraints = pd.read_csv(root / "data" / "trading_constraints.csv.gz", parse_dates=["date"])
    predictions = {arm: pd.read_csv(out / "predictions" / f"{arm}.csv.gz", usecols=["date", "asset_id", "prediction"], parse_dates=["date"]) for arm in ARMS}
    risk = pd.read_csv(root / "data" / "risk_exposure.csv.gz", usecols=["date", "asset_id", "industry", "market_cap"], parse_dates=["date"])
    risk.set_index(["date", "asset_id"], verify_integrity=True)
    validate_risk_values(risk)
    risk["log_market_cap"] = np.log(risk.market_cap)
    members = predictions["Ridge_CONTROLS"][["date", "asset_id"]].merge(risk, on=["date", "asset_id"], validate="one_to_one")
    if len(members) != len(predictions["Ridge_CONTROLS"]):
        raise AssertionError("Missing member risk exposure")
    baseline = members.groupby("date").log_market_cap.agg(["mean", "std"])
    industry_baseline = members.groupby(["date", "industry"]).size().unstack(fill_value=0)
    industry_baseline = industry_baseline.div(industry_baseline.sum(axis=1), axis=0)
    scenarios = {}
    for cost in COSTS:
        frame = constraints.copy()
        columns = ["fee_bps_one_way", "spread_bps", "impact_bps_proxy"]
        frame[columns] = frame[columns] * cost
        scenarios[cost] = frame
    _WORKER.update(market=market, predictions=predictions, scenarios=scenarios, risk=risk,
                   risk_baseline=baseline, industry_baseline=industry_baseline,
                   out=out, rebalance_source=Path(rebalance_source))


def risk_diagnostics(daily, positions):
    dates = pd.DatetimeIndex(daily.date, name="date")
    nav = daily.set_index("date").nav
    holdings = positions.groupby(["date", "asset_id"], as_index=False).market_value.sum()
    joined = holdings.merge(_WORKER["risk"], on=["date", "asset_id"], how="left", validate="one_to_one")
    if joined[["industry", "market_cap"]].isna().any().any():
        raise AssertionError("Missing actual-held-asset risk exposure")
    joined["weighted_log_cap"] = joined.market_value * joined.log_market_cap
    equity = joined.groupby("date").market_value.sum().reindex(dates, fill_value=0.)
    log_cap = joined.groupby("date").weighted_log_cap.sum().reindex(dates).div(equity.where(equity > 0))
    baseline = _WORKER["risk_baseline"].reindex(dates)
    if baseline.isna().any().any() or (baseline["std"] <= 0).any():
        raise AssertionError("Missing or degenerate member size baseline")
    industries = _WORKER["industry_baseline"].columns.union(pd.Index(joined.industry.unique()))
    industry_values = joined.groupby(["date", "industry"]).market_value.sum().unstack(fill_value=0.).reindex(index=dates, columns=industries, fill_value=0.)
    allocation = industry_values.div(equity.where(equity > 0), axis=0)
    member_allocation = _WORKER["industry_baseline"].reindex(index=dates, columns=industries, fill_value=0.)
    active = allocation - member_allocation
    diagnostics = pd.DataFrame({
        "date": dates, "equity_fraction": equity.div(nav).to_numpy(),
        "weighted_log_cap": log_cap.to_numpy(), "member_mean_log_cap": baseline["mean"].to_numpy(),
        "size_z_equity": ((log_cap - baseline["mean"]) / baseline["std"]).to_numpy(),
        "industry_active_l1_equity": active.abs().sum(axis=1, min_count=1).to_numpy(),
        "max_abs_industry_active_equity": active.abs().max(axis=1).to_numpy(),
    })
    industry_rows = pd.DataFrame({
        "weight_nav": industry_values.div(nav, axis=0).stack(future_stack=True),
        "weight_equity": allocation.stack(future_stack=True),
        "member_equal_weight": member_allocation.stack(future_stack=True),
    }).rename_axis(["date", "industry"]).reset_index()
    return diagnostics, industry_rows


def plan_diagnostics(plans, reviews, max_review_turnover):
    records = []
    grouped = {(date, sleeve): frame for (date, sleeve), frame in plans.groupby(["signal_date", "sleeve"])}
    for review in reviews.itertuples(index=False):
        frame = grouped.get((review.signal_date, review.sleeve))
        if frame is None:
            desired = planned = 0.
        else:
            if review.signal_nav <= 0:
                raise AssertionError("Plan diagnostic requires positive sleeve NAV")
            actual = frame.signal_shares * frame.signal_close / review.signal_nav
            desired = float((frame.target_weight - actual).abs().sum())
            planned = float((frame.effective_weight - actual).abs().sum())
        capped = max_review_turnover is not None and not review.terminal
        if capped and planned > max_review_turnover + 1e-12:
            raise AssertionError("Close-planned turnover exceeds frozen budget")
        records.append({"signal_date": review.signal_date, "sleeve": review.sleeve,
                        "terminal": review.terminal, "ideal_gross_turnover": desired,
                        "planned_gross_turnover": planned,
                        "budget_applies": capped,
                        "budget_binding": capped and desired > max_review_turnover + 1e-12})
    return pd.DataFrame(records)


def _execute(spec):
    result = simulate_rebalance(_WORKER["predictions"][spec["predictor"]], _WORKER["market"],
                                _WORKER["scenarios"][spec["cost_multiplier"]], start_signal=START,
                                end_date=END, policy=spec["policy"],
                                max_review_turnover=spec["max_review_turnover"])
    folder = _WORKER["out"] / "runs" / spec["run"]
    folder.mkdir()
    exposures = exposure_diagnostics(result["daily"], result["positions"])
    risks, industry = risk_diagnostics(result["daily"], result["positions"])
    plans = plan_diagnostics(result["plans"], result["reviews"], spec["max_review_turnover"])
    for filename, frame in [("exposure_daily", exposures), ("risk_exposure_daily", risks),
                            ("industry_weights_daily", industry), ("budget_diagnostics", plans),
                            ("daily", result["daily"]), ("reviews", result["reviews"])]:
        frame.to_csv(folder / f"{filename}.csv", mode="x", index=False)
    for kind in ["orders", "positions", "plans"]:
        result[kind].to_csv(folder / f"{kind}.csv.gz", mode="x", index=False, compression=COMPRESSION)
    reference_error = None
    if spec["predictor"].endswith("_CONTROLS") and spec["rule"] in ("NET", "BOTH"):
        archived = pd.read_csv(_WORKER["rebalance_source"] / "runs" / spec["run"] / "daily.csv", parse_dates=["date"])
        pd.testing.assert_frame_equal(result["daily"], archived, check_dtype=False, rtol=1e-10, atol=1e-12)
        reference_error = float((result["daily"].nav - archived.nav).abs().max())
    records = []
    for period in ["FULL", "2023", "2024"]:
        stats = period_statistics(result["daily"], result["orders"], period)
        exposure_part = exposures if period == "FULL" else exposures.loc[exposures.date.dt.year.eq(int(period))]
        risk_part = risks if period == "FULL" else risks.loc[risks.date.dt.year.eq(int(period))]
        plan_part = plans if period == "FULL" else plans.loc[plans.signal_date.dt.year.eq(int(period))]
        stats.update(mean_asset_count=float(exposure_part.asset_count.mean()),
                     mean_hhi=float(exposure_part.hhi.mean()), max_asset_weight=float(exposure_part.max_asset_weight.max()),
                     mean_gross_exposure=float(exposure_part.gross_exposure.mean()),
                     turnover_per_mean_exposure=float(stats["mean_daily_turnover"] / exposure_part.gross_exposure.mean()),
                     mean_weighted_log_cap=float(risk_part.weighted_log_cap.mean()),
                     mean_size_z_equity=float(risk_part.size_z_equity.mean()),
                     mean_industry_active_l1_equity=float(risk_part.industry_active_l1_equity.mean()),
                     mean_planned_gross_turnover=float(plan_part.loc[~plan_part.terminal, "planned_gross_turnover"].mean()),
                     budget_binding_reviews=int(plan_part.budget_binding.sum()),
                     reference_max_nav_difference=reference_error)
        records.append({**spec, **stats})
    summary = {**result["summary"], **records[0]}
    write_json(folder / "summary.json", jsonable(summary))
    return summary, records


def portfolio_pairs(out):
    records = []
    for rule, (policy, _) in RULES.items():
        for name in MODELS:
            for cost in COSTS:
                daily = {variant: pd.read_csv(out / "runs" / f"{rule}__{name}_{variant}__cost{cost}" / "daily.csv", parse_dates=["date"]).set_index("date").daily_return
                         for variant in ("CONTROLS", "JOINT")}
                left, right = daily["JOINT"], daily["CONTROLS"]
                if not left.index.equals(right.index):
                    raise AssertionError("Paired portfolio dates differ")
                for period in ["2023", "2024"]:
                    for metric, differences in [("daily_return_difference", left - right),
                                                ("daily_log_relative_growth", np.log1p(left) - np.log1p(right))]:
                        records.append({"rule": rule, "policy": policy, "model": name,
                                        "predictor": name + "_JOINT", "reference_predictor": name + "_CONTROLS",
                                        "period": period, "cost_multiplier": cost, "metric": metric,
                                        **moving_block_interval(differences.loc[differences.index.year == int(period)])})
    return pd.DataFrame(records)


def make_figures(periods, out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7), sharey=True, layout="constrained")
    x = np.arange(len(RULES))
    for axis, name in zip(axes, MODELS):
        for offset, variant, color in [(-.18, "CONTROLS", "#64748b"), (.18, "JOINT", "#2563eb")]:
            selected = periods.loc[periods.period.eq("2024") & periods.cost_multiplier.eq(1) & periods.predictor.eq(name + "_" + variant)].set_index("rule").loc[list(RULES)]
            bars = axis.bar(x + offset, 100 * selected.total_return, .34, color=color,
                            label="Industry + size" if variant == "CONTROLS" else "+ 20 PIT factors")
            axis.bar_label(bars, fmt="%.2f%%", fontsize=9, padding=3)
        axis.set_title(name)
        axis.set_xticks(x, ["Net", "Double buffer", "20% plan budget"])
        axis.axhline(0, color="#475569", lw=.8)
        axis.grid(axis="y", alpha=.2)
    axes[0].set_ylabel("2024 return after supplied costs (%)")
    axes[1].legend(fontsize=9)
    fig.suptitle("Direct increment | common rules, different realized exposures")
    fig.savefig(out / "joint_increment_2024.png", dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
    for axis, name in zip(axes, MODELS):
        for variant, color in [("CONTROLS", "#64748b"), ("JOINT", "#2563eb")]:
            frame = pd.read_csv(out / "runs" / f"BUDGET20__{name}_{variant}__cost1" / "daily.csv", parse_dates=["date"])
            opening = frame.loc[frame.date.dt.year.eq(2023), "nav"].iloc[-1]
            frame = frame.loc[frame.date.dt.year.eq(2024)]
            axis.plot(frame.date, frame.nav / opening, color=color,
                      label="Industry + size" if variant == "CONTROLS" else "+ 20 PIT factors")
        axis.set_title(f"{name} | fixed close-plan budget")
        axis.grid(alpha=.2)
        axis.tick_params(axis="x", rotation=25)
    axes[0].set_ylabel("NAV / 2023 year-end NAV")
    axes[1].legend(fontsize=9)
    fig.savefig(out / "budget_nav_2024.png", dpi=170)
    plt.close(fig)


def run(root, panel_path, portfolio_source, rebalance_source, out, workers):
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite study: {out}")
    if workers not in (1, 2):
        raise ValueError("Use one or two account workers")
    started = utc_now()
    protocol = build_protocol(root, panel_path, portfolio_source, rebalance_source)
    out.mkdir(parents=True)
    write_json(out / "protocol.json", protocol)
    (out / "source_snapshot").mkdir()
    for path in protocol["code_sha256"]:
        shutil.copyfile(path, out / "source_snapshot" / Path(path).name)
    predictions = fit_joint_predictions(root, panel_path, portfolio_source, out)
    evaluate_predictions(predictions, panel_path, out)
    del predictions
    (out / "runs").mkdir()
    summaries, periods = [], []
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                             initializer=_initialize, initargs=(str(root), str(out), str(rebalance_source))) as pool:
        futures = {pool.submit(_execute, spec): spec for spec in protocol["runs"]}
        for future in as_completed(futures):
            summary, rows = future.result()
            summaries.append(summary)
            periods.extend(rows)
            print(f"Completed {summary['run']}: final NAV={summary['final_nav']:.6f}", flush=True)
    summary = pd.DataFrame(summaries).sort_values("run").reset_index(drop=True)
    periods = pd.DataFrame(periods).sort_values(["run", "period"]).reset_index(drop=True)
    summary.to_csv(out / "summary.csv", mode="x", index=False)
    periods.to_csv(out / "period_metrics.csv", mode="x", index=False)
    portfolio_pairs(out).to_csv(out / "paired_portfolio.csv", mode="x", index=False)
    make_figures(periods, out)
    for field in ("inputs_sha256", "code_sha256"):
        for path, expected in protocol[field].items():
            if sha256(Path(path)) != expected:
                raise AssertionError(f"Source changed during study: {path}")
    write_json(out / "run_manifest.json", {
        "status": "completed", "started_at_utc": started, "finished_at_utc": utc_now(),
        "scope": SCOPE, "command": sys.orig_argv, "protocol_sha256": object_hash(protocol),
        "joint_model_fits": 4, "archived_control_models_verified": 4, "ledger_runs": len(summaries),
        "inputs_sha256": protocol["inputs_sha256"], "code_sha256": protocol["code_sha256"],
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                        "sklearn": sklearn.__version__, "workers": workers,
                        "threads": {key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}},
        "outputs_sha256": {str(path.relative_to(out)): sha256(path) for path in sorted(out.rglob("*")) if path.is_file()},
    })
    print(periods.loc[periods.period.eq("2024") & periods.cost_multiplier.eq(1),
                      ["rule", "predictor", "total_return", "mean_daily_turnover", "mean_cash_fraction"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--panel", type=Path, default=PROJECT / "outputs" / "factor_timing" / "pit_research_panel.csv.gz")
    parser.add_argument("--portfolio-source", type=Path, default=PROJECT / "outputs" / "portfolio_model_comparison")
    parser.add_argument("--rebalance-source", type=Path, default=PROJECT / "outputs" / "rebalance_study")
    parser.add_argument("--out", type=Path, default=PROJECT / "outputs" / "joint_factor_increment")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    run(args.root, args.panel, args.portfolio_source, args.rebalance_source, args.out, args.workers)
