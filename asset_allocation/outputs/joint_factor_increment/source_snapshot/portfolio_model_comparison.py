"""Frozen annual predictors in a common five-day, long-only cash ledger.

The full signal-date member universe is predicted, including unknown outcomes.
2023 is descriptive development history; 2024 is previously inspected history.
No model, portfolio rule or cost assumption is selected from these results.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import sys
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn

from compare_models import (
    FACTORS, KEYS, PROJECT, jsonable, key_hash, make_models,
    moving_block_interval, object_hash, sha256, temporal_split, utc_now, write_json,
)
from factor_timing import DEFAULT_DATA
from industry_size_attribution import make_control_model, merge_risk
from portfolio_ledger import simulate
from temporal_model import load_signal_panel


START = pd.Timestamp("2023-01-02")
END = pd.Timestamp("2024-12-31")
ARMS = ["Ridge_ALL20", "HGB_ALL20", "Ridge_CONTROLS", "HGB_CONTROLS"]
PREDICTORS = ARMS + ["EqualWeight"]
COSTS = [0, 1, 2]
COMPRESSION = {"method": "gzip", "compresslevel": 1, "mtime": 0}
SCOPE = "Previously inspected synthetic history, not an untouched sample or verified opening execution."


def select_complete_signals(panel: pd.DataFrame, year: int) -> pd.DataFrame:
    signals = panel.loc[panel.date.dt.year == year].sort_values(KEYS).reset_index(drop=True)
    if signals.empty or signals[KEYS].isna().any().any():
        raise ValueError("Empty signal year or missing signal keys")
    signals.set_index(KEYS, verify_integrity=True)
    return signals


def verify_archived(result: pd.DataFrame, archived: pd.DataFrame) -> float:
    for frame in [result, archived]:
        if frame[KEYS].isna().any().any() or frame.duplicated(KEYS).any():
            raise AssertionError("Non-unique or missing prediction keys")
        if not np.isfinite(frame.prediction).all():
            raise AssertionError("Nonfinite prediction")
    joined = archived[KEYS + ["prediction"]].merge(
        result[KEYS + ["prediction"]], on=KEYS, how="left",
        suffixes=("_old", "_new"), validate="one_to_one",
    )
    if joined.prediction_new.isna().any():
        raise AssertionError("Complete signals omit an archived key")
    np.testing.assert_allclose(joined.prediction_new, joined.prediction_old, atol=1e-10, rtol=1e-10)
    return float(np.abs(joined.prediction_new - joined.prediction_old).max())


def period_statistics(daily: pd.DataFrame, orders: pd.DataFrame, period: str) -> dict:
    daily = daily.sort_values("date").reset_index(drop=True)
    mask = pd.Series(True, index=daily.index) if period == "FULL" else daily.date.dt.year.eq(int(period))
    part = daily.loc[mask]
    if part.empty:
        raise ValueError(f"No observations for period {period}")
    initial = float(daily.nav.iloc[part.index[0] - 1]) if part.index[0] > 0 else 1.
    final = float(part.nav.iloc[-1])
    returns = part.daily_return.dropna()
    if initial <= 0 or returns.empty:
        raise ValueError("Period must have positive opening capital and observed daily returns")
    volatility = float(returns.std(ddof=1)) if len(returns) >= 2 else None
    path = np.r_[initial, part.nav.to_numpy()]
    selected_orders = orders.loc[orders.execution_date.between(part.date.iloc[0], part.date.iloc[-1])]
    return {
        "period": period, "initial_nav": initial, "final_nav": final,
        "n_return_days": len(returns), "total_return": final / initial - 1,
        "annualized_return": (final / initial) ** (252 / len(returns)) - 1,
        "annualized_volatility": volatility * np.sqrt(252) if volatility is not None else None,
        "sharpe": float(returns.mean() / volatility * np.sqrt(252)) if volatility else None,
        "max_drawdown": float((path / np.maximum.accumulate(path) - 1).min()),
        "mean_daily_turnover": float(part.loc[part.daily_return.notna(), "gross_turnover"].mean()),
        "total_fee": float(part.fee.sum()), "total_spread_cost": float(part.spread_cost.sum()),
        "total_impact_cost": float(part.impact_cost.sum()), "total_writeoff": float(part.writeoff.sum()),
        "filled_orders": int(selected_orders.filled_shares.gt(0).sum()),
        "actual_flag_conflict_orders": int(selected_orders.actual_flag_conflict.sum()),
        "mean_cash_fraction": float((part.cash / part.nav).mean()),
    }


def frozen_protocol(root: Path, benchmark: Path, attribution: Path, panel_path: Path) -> dict:
    paths = [root / "data" / f"{name}.csv.gz" for name in [
        "market_daily", "factors", "universe", "factor_availability", "trading_constraints", "risk_exposure",
    ]] + [root / "data" / "data_dictionary.md", panel_path, panel_path.parent / "run_manifest.json",
          benchmark / "protocol.json", benchmark / "selection.json", attribution / "protocol.json"]
    timing = json.loads((panel_path.parent / "run_manifest.json").read_text())
    if sha256(panel_path) != timing["outputs_sha256"][panel_path.name]:
        raise ValueError("Canonical PIT panel changed")
    models = make_models()
    previous = json.loads((benchmark / "protocol.json").read_text())["protocol"]
    previous_controls = json.loads((attribution / "protocol.json").read_text())["protocol"]
    parameters = {}
    for name in ["Ridge", "HGB"]:
        params = jsonable(models[name].get_params(deep=True))
        if params != previous["model_parameters"][name]:
            raise ValueError(f"{name}: parameters differ from annual benchmark")
        parameters[name + "_ALL20"] = params
        parameters[name + "_CONTROLS"] = jsonable(make_control_model(name).get_params(deep=True))
        if parameters[name + "_CONTROLS"] != previous_controls["model_parameters"][name]:
            raise ValueError(f"{name}: parameters differ from industry/size benchmark")
    for stage in ["validation", "confirmation"]:
        for folder, filenames, prefix in [
            (benchmark, ["split_manifest.json", "Ridge_predictions.csv.gz", "HGB_predictions.csv.gz"], stage + "/"),
            (attribution, ["split_manifest.json", "Ridge_CONTROLS_predictions.csv.gz", "HGB_CONTROLS_predictions.csv.gz"], ""),
        ]:
            manifest_path = folder / stage / "run_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            if manifest["status"] != "completed":
                raise ValueError(f"Incomplete prerequisite {manifest_path}")
            for name in filenames:
                path = folder / stage / name
                if sha256(path) != manifest["outputs_sha256"][prefix + name]:
                    raise ValueError(f"Changed prerequisite {path}")
                paths.append(path)
            paths.append(manifest_path)
    code = [PROJECT / filename for filename in [
        "portfolio_model_comparison.py", "portfolio_ledger.py", "compare_models.py",
        "industry_size_attribution.py", "temporal_model.py", "factor_timing.py",
        "test_portfolio_model_comparison.py", "audit_portfolio_model_comparison.py",
    ]]
    return {
        "frozen_at_utc": utc_now(), "scope": SCOPE, "predictors": PREDICTORS,
        "primary_period": "2024", "descriptive_periods": ["2023", "FULL"],
        "primary_comparisons": ["Ridge_ALL20 vs HGB_ALL20", "ALL20 vs same-learner CONTROLS", "each vs EqualWeight"],
        "training": "Annual refit; canonical archived training rows and preprocessing; mature labels strictly before signal-year start; no tuning",
        "signal_universe": "All currently listed members; never exclude unknown future labels",
        "start_signal": str(START.date()), "end_date": str(END.date()),
        "last_entry_signal": str((END - pd.offsets.BDay(6)).date()),
        "selection": "Long-only prediction >= same-day 90% quantile, all ties retained; equal weights among selected names",
        "benchmark": "EqualWeight uses every current member with the identical five-day rotation and costs",
        "capital": "Initial NAV 1, five independent sleeves initially 0.2, no borrowing, fractional shares, no cash interest",
        "execution": "Signal close sets names and fee-inclusive budget; next open; normal exit t+6; blocked buy cash not redistributed; overdue sells retried",
        "netting": "No netting, including same-day same-asset sell/rebuy within a sleeve; full scheduled liquidation and re-entry",
        "constraints": "Previous business-day can_buy/can_sell; current final flags audited only",
        "cost_multipliers": COSTS, "primary_cost_multiplier": 1,
        "cost_formula": "multiplier * previous-day (fee_bps_one_way + spread_bps/2 + impact_bps_proxy), each actual side",
        "missing_quote": "Zero recovery at first missing quote, separately reported valuation assumption",
        "terminal": "Stop new signals six business days before end; remaining unsellable positions marked at actual close",
        "risk_availability": "Industry and market_cap assumed known after the signal-day close; no separate publication timestamps",
        "metrics": "Daily close NAV, 252-day annualization, zero risk-free rate; annual returns and drawdown include prior year-end NAV; 2024 carries 2023 positions",
        "uncertainty": "Paired daily arithmetic return difference and log relative growth; moving 20-day blocks, 2000 draws; reference intervals, no multiplicity correction",
        "model_parameters": parameters,
        "inputs_sha256": {str(p.resolve()): sha256(p) for p in paths},
        "code_sha256": {str(p.resolve()): sha256(p) for p in code},
    }


def fit_predictions(root: Path, benchmark: Path, attribution: Path, panel_path: Path, out: Path):
    canonical = pd.read_csv(panel_path, parse_dates=["date", "entry_date", "exit_date"])
    full = load_signal_panel(root)
    risk = pd.read_csv(root / "data" / "risk_exposure.csv.gz", parse_dates=["date"])
    full_risk, canonical_risk = merge_risk(full, risk), merge_risk(canonical, risk)
    (out / "models").mkdir()
    (out / "predictions").mkdir()
    parts = {arm: [] for arm in ARMS}
    fits, coverage, selections = [], [], []
    for year, stage in [(2023, "validation"), (2024, "confirmation")]:
        train, _, split = temporal_split(canonical, stage)
        if split != json.loads((benchmark / stage / "split_manifest.json").read_text()):
            raise AssertionError("Annual training/evaluation split differs from archived benchmark")
        risk_train, _, risk_split = temporal_split(canonical_risk, stage)
        if risk_split != split:
            raise AssertionError("Control and all-factor training samples differ")
        signals = select_complete_signals(full, year)
        risk_signals = select_complete_signals(full_risk, year)
        models = make_models()
        for name in ["Ridge", "HGB"]:
            for variant in ["ALL20", "CONTROLS"]:
                arm = name + "_" + variant
                if variant == "ALL20":
                    model = models[name]
                    x_train, x_signals = train[FACTORS], signals[FACTORS]
                    archive_path = benchmark / stage / f"{name}_predictions.csv.gz"
                else:
                    model = make_control_model(name)
                    x_train = risk_train[["industry", "log_market_cap"]]
                    x_signals = risk_signals[["industry", "log_market_cap"]]
                    archive_path = attribution / stage / f"{name}_CONTROLS_predictions.csv.gz"
                print(f"Fitting {year} {arm}: {len(train)} mature rows -> {len(signals)} full signals", flush=True)
                tick = time.perf_counter()
                model.fit(x_train, train.forward_5d)
                scores = model.predict(x_signals)
                if not np.isfinite(scores).all():
                    raise AssertionError("Nonfinite full-universe prediction")
                result = signals[KEYS + ["entry_date", "exit_date", "forward_5d"]].assign(
                    prediction=scores, train_cutoff=pd.Timestamp(year, 1, 1) - pd.offsets.BDay(1),
                )
                archive = pd.read_csv(archive_path, parse_dates=["date"])
                difference = verify_archived(result, archive)
                joblib.dump(model, out / "models" / f"{year}_{arm}.joblib")
                parts[arm].append(result)
                fits.append({"year": year, "predictor": arm, "training_rows": len(train),
                             "training_keys_sha256": key_hash(train),
                             "max_training_label_exit": str(train.exit_date.max().date()),
                             "first_signal": str(result.date.min().date()),
                             "fit_predict_seconds": time.perf_counter() - tick,
                             "archived_rows": len(archive), "max_archived_prediction_difference": difference})
                coverage.append({"year": year, "predictor": arm, "signal_rows": len(result),
                                 "signal_dates": result.date.nunique(), "missing_labels": int(result.forward_5d.isna().sum()),
                                 "archived_rows": len(archive), "added_rows": len(result) - len(archive),
                                 "signal_keys_sha256": key_hash(result)})
                for date, rows in result.groupby("date", sort=False):
                    chosen = rows.prediction.ge(rows.prediction.quantile(.9))
                    selections.append({"date": date, "predictor": arm, "universe_count": len(rows),
                                       "selected_count": int(chosen.sum()),
                                       "selected_fraction": float(chosen.mean()),
                                       "selected_missing_labels": int(rows.loc[chosen, "forward_5d"].isna().sum())})
    predictions = {}
    for arm, frames in parts.items():
        result = pd.concat(frames, ignore_index=True).sort_values(KEYS).reset_index(drop=True)
        if not result[KEYS].equals(full.loc[full.date.dt.year.between(2023, 2024), KEYS].reset_index(drop=True)):
            raise AssertionError("Full signal coverage changed")
        result.to_csv(out / "predictions" / f"{arm}.csv.gz", index=False, compression=COMPRESSION)
        predictions[arm] = result
    predictions["EqualWeight"] = predictions["Ridge_ALL20"][KEYS].assign(prediction=0.)
    pd.DataFrame(fits).to_csv(out / "fits.csv", index=False)
    pd.DataFrame(coverage).to_csv(out / "signal_coverage.csv", index=False)
    pd.DataFrame(selections).to_csv(out / "selection_diagnostics.csv", index=False)
    return predictions


def make_figures(dailies: dict, periods: pd.DataFrame, out: Path):
    labels = {"Ridge_ALL20": "Ridge: 20 factors", "HGB_ALL20": "HGB: 20 factors",
              "Ridge_CONTROLS": "Ridge: industry + size", "HGB_CONTROLS": "HGB: industry + size",
              "EqualWeight": "Member equal weight"}
    colors = ["#2563eb", "#d97706", "#60a5fa", "#fbbf24", "#6b7280"]
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, layout="constrained")
    for arm, color in zip(PREDICTORS, colors):
        frame = dailies[f"{arm}__cost1"]
        nav = frame.nav.to_numpy()
        axes[0].plot(frame.date, nav, label=labels[arm], color=color, lw=1.6)
        axes[1].plot(frame.date, 100 * (nav / np.maximum.accumulate(nav) - 1), color=color, lw=1.3)
    axes[0].set_title("Annual models | supplied costs + lagged execution proxies | synthetic history")
    axes[0].set_ylabel("Net asset value (initial = 1)")
    axes[0].legend(fontsize=8, ncols=2)
    axes[1].set_ylabel("Drawdown (%)")
    for ax in axes:
        ax.grid(alpha=.18)
    fig.savefig(out / "nav_drawdown.png", dpi=170)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
    for i, (arm, color) in enumerate(zip(PREDICTORS, colors)):
        frame = periods.loc[(periods.period == "2024") & (periods.predictor == arm)].sort_values("cost_multiplier")
        ax.bar(np.arange(3) + (i - 2) * .15, 100 * frame.total_return, width=.145, label=labels[arm], color=color)
    ax.set_xticks(np.arange(3), ["0 x costs", "1 x supplied costs", "2 x supplied costs"])
    ax.set_ylabel("2024 return (%)")
    ax.set_title("Cost sensitivity | same lagged permissions and five-day full rotation")
    ax.axhline(0, color="#374151", lw=.8)
    ax.legend(fontsize=8, ncols=2)
    ax.grid(axis="y", alpha=.18)
    fig.savefig(out / "cost_sensitivity_2024.png", dpi=170)
    plt.close(fig)


def run(root: Path, benchmark: Path, attribution: Path, panel_path: Path, out: Path):
    started = utc_now()
    if out.exists():
        raise FileExistsError(f"Refusing to replace study directory: {out}")
    protocol = frozen_protocol(root, benchmark, attribution, panel_path)
    out.mkdir(parents=True)
    write_json(out / "protocol.json", protocol)
    predictions = fit_predictions(root, benchmark, attribution, panel_path, out)
    market = pd.read_csv(root / "data" / "market_daily.csv.gz", parse_dates=["date"])
    constraints = pd.read_csv(root / "data" / "trading_constraints.csv.gz", parse_dates=["date"])
    (out / "runs").mkdir()
    summaries, periods, dailies = [], [], {}
    for multiplier in COSTS:
        scenario = constraints.copy()
        columns = ["fee_bps_one_way", "spread_bps", "impact_bps_proxy"]
        scenario[columns] = scenario[columns] * multiplier
        for arm in PREDICTORS:
            name = f"{arm}__cost{multiplier}"
            print(f"Ledger {name}", flush=True)
            result = simulate(predictions[arm], market, scenario, start_signal=START, end_date=END,
                              selection="equal_weight" if arm == "EqualWeight" else "top_decile",
                              cost_mode="base", constraint_mode="lagged")
            folder = out / "runs" / name
            folder.mkdir()
            result["daily"].to_csv(folder / "daily.csv", index=False)
            for kind in ["orders", "positions"]:
                result[kind].to_csv(folder / f"{kind}.csv.gz", index=False, compression=COMPRESSION)
            write_json(folder / "summary.json", jsonable(result["summary"]))
            identity = {"run": name, "predictor": arm, "cost_multiplier": multiplier}
            summaries.append({**identity, **result["summary"],
                              "filled_orders": int(result["orders"].filled_shares.gt(0).sum()),
                              "mean_cash_fraction": float((result["daily"].cash / result["daily"].nav).mean())})
            for period in ["FULL", "2023", "2024"]:
                periods.append({**identity, **period_statistics(result["daily"], result["orders"], period)})
            dailies[name] = result["daily"]
            print(f"  final NAV={result['summary']['final_nav']:.6f}", flush=True)
    summary, period_frame = pd.DataFrame(summaries), pd.DataFrame(periods)
    indexed = period_frame.set_index(["predictor", "cost_multiplier", "period"], verify_integrity=True)
    period_frame["relative_terminal_wealth_vs_equalweight"] = [
        (1 + r.total_return) / (1 + indexed.loc[("EqualWeight", r.cost_multiplier, r.period), "total_return"]) - 1
        for r in period_frame.itertuples()]
    period_frame["relative_terminal_wealth_vs_matching_controls"] = [
        ((1 + r.total_return) / (1 + indexed.loc[(r.predictor.replace("ALL20", "CONTROLS"), r.cost_multiplier, r.period), "total_return"]) - 1)
        if r.predictor.endswith("ALL20") else np.nan for r in period_frame.itertuples()]
    summary.to_csv(out / "summary.csv", index=False)
    period_frame.to_csv(out / "period_metrics.csv", index=False)
    comparisons = [("Ridge_ALL20", "HGB_ALL20"), ("Ridge_ALL20", "Ridge_CONTROLS"),
                   ("HGB_ALL20", "HGB_CONTROLS")] + [(arm, "EqualWeight") for arm in ARMS]
    paired = []
    for multiplier in COSTS:
        for candidate, reference in comparisons:
            left = dailies[f"{candidate}__cost{multiplier}"].set_index("date").daily_return
            right = dailies[f"{reference}__cost{multiplier}"].set_index("date").daily_return
            if not left.index.equals(right.index):
                raise AssertionError("Portfolio comparison dates differ")
            for period in ["FULL", "2023", "2024"]:
                mask = np.ones(len(left), dtype=bool) if period == "FULL" else left.index.year == int(period)
                for metric, diff in [("daily_return_difference", left - right),
                                     ("daily_log_relative_growth", np.log1p(left) - np.log1p(right))]:
                    paired.append({"candidate": candidate, "reference": reference,
                                   "cost_multiplier": multiplier, "period": period, "metric": metric,
                                   **moving_block_interval(diff.loc[mask])})
    pd.DataFrame(paired).to_csv(out / "paired_comparisons.csv", index=False)
    make_figures(dailies, period_frame, out)
    for field in ["inputs_sha256", "code_sha256"]:
        for path, expected in protocol[field].items():
            if sha256(Path(path)) != expected:
                raise AssertionError(f"Source changed during run: {path}")
    write_json(out / "run_manifest.json", {
        "status": "completed", "started_at_utc": started, "finished_at_utc": utc_now(),
        "command": sys.orig_argv, "scope": SCOPE, "protocol_sha256": object_hash(protocol),
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                        "sklearn": sklearn.__version__, "threads": {k: os.environ.get(k) for k in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]}},
        "annual_model_fits": 8, "ledger_runs": len(summaries),
        "inputs_sha256": protocol["inputs_sha256"], "code_sha256": protocol["code_sha256"],
        "outputs_sha256": {str(p.relative_to(out)): sha256(p) for p in sorted(out.rglob("*")) if p.is_file()},
    })
    print(period_frame.loc[(period_frame.period == "2024") & (period_frame.cost_multiplier == 1),
                           ["predictor", "total_return", "sharpe", "max_drawdown", "mean_daily_turnover"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--benchmark", type=Path, default=PROJECT / "outputs" / "model_benchmark")
    parser.add_argument("--attribution", type=Path, default=PROJECT / "outputs" / "industry_size_attribution")
    parser.add_argument("--panel", type=Path, default=PROJECT / "outputs" / "factor_timing" / "pit_research_panel.csv.gz")
    parser.add_argument("--out", type=Path, default=PROJECT / "outputs" / "portfolio_model_comparison")
    args = parser.parse_args()
    run(args.root, args.benchmark, args.attribution, args.panel, args.out)
