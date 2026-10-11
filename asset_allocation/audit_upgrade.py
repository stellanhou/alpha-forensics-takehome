"""Independently reconcile saved orders, cash, inventory and daily NAV."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay

from factor_timing import DEFAULT_DATA, PROJECT, sha256


def close_enough(actual, expected):
    np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-10)


def verify_manifest(folder: Path, filename: str, output_root: Path | None = None):
    manifest = json.loads((folder / filename).read_text())
    base = output_root or folder
    for relative, digest in manifest["outputs_sha256"].items():
        if sha256(base / relative) != digest:
            raise AssertionError(f"Saved output changed: {base / relative}")
    return manifest


def run(root: Path, portfolio: Path, temporal: Path, output: Path):
    for line in (root / "SHA256_PUBLIC.txt").read_text().splitlines():
        digest, relative = line.split(maxsplit=1)
        if sha256(root / relative) != digest:
            raise AssertionError(f"Original package changed: {relative}")
    verify_manifest(PROJECT / "outputs" / "factor_timing", "run_manifest.json")
    verify_manifest(PROJECT / "outputs" / "model_rerun" / "pit", "baseline_run_manifest.json",
                    PROJECT / "outputs" / "model_rerun")
    temporal_manifest = verify_manifest(temporal, "run_manifest.json")
    portfolio_manifest = verify_manifest(portfolio, "run_manifest.json")
    folds = pd.read_csv(temporal / "fold_manifest.csv", parse_dates=[
        "train_cutoff", "train_label_exit_max", "prediction_date_min"
    ])
    assert (folds.train_label_exit_max <= folds.train_cutoff).all()
    assert (folds.train_cutoff < folds.prediction_date_min).all()
    assert folds.training_prediction_key_overlap.eq(0).all()
    assert folds.fitted_iterations.eq(55).all() and not folds.early_stopping_used.any()

    market = pd.read_csv(root / "data" / "market_daily.csv.gz", parse_dates=["date"])
    constraints = pd.read_csv(root / "data" / "trading_constraints.csv.gz", parse_dates=["date"])
    prices = market[["date", "asset_id", "open"]].rename(columns={"date": "execution_date", "open": "actual_open"})
    previous = constraints.rename(columns={"date": "signal_date"})
    actual = constraints[["date", "asset_id", "can_buy", "can_sell"]].rename(
        columns={"date": "execution_date", "can_buy": "actual_buy", "can_sell": "actual_sell"}
    )
    close_prices = market.pivot(index="date", columns="asset_id", values="close")
    summaries = pd.read_csv(portfolio / "portfolio_comparison.csv")
    records = []
    for row in summaries.itertuples():
        print(f"Independent ledger audit: {row.run}", flush=True)
        folder = portfolio / row.run
        daily = pd.read_csv(folder / "daily.csv", parse_dates=["date"]).set_index("date")
        orders = pd.read_csv(folder / "orders.csv.gz", parse_dates=["signal_date", "execution_date", "scheduled_exit_date"])
        buys = orders.side.eq("buy")
        sells = orders.side.eq("sell")
        filled = orders.filled_shares.gt(0)
        assert (orders.loc[buys, "execution_date"] == orders.loc[buys, "signal_date"] + BDay(1)).all()
        assert (orders.loc[buys, "scheduled_exit_date"] == orders.loc[buys, "execution_date"] + BDay(5)).all()
        assert (orders.loc[sells & filled, "execution_date"] >= orders.loc[sells & filled, "scheduled_exit_date"]).all()
        assert orders.loc[buys & filled, "requested_budget"].le(orders.loc[buys & filled, "planned_budget"] + 1e-12).all()
        cost = orders.fee + orders.spread_cost + orders.impact_cost
        cash_flow = np.where(sells, orders.filled_notional, 0.) - np.where(buys, orders.filled_notional, 0.) - cost
        cash_changes = pd.Series(cash_flow, index=orders.index).groupby(orders.execution_date).sum().reindex(daily.index, fill_value=0)
        expected_cash = 1 + cash_changes.cumsum()
        close_enough(daily.cash, expected_cash)
        share_flow = np.where(buys, orders.filled_shares, 0.) - np.where(sells, orders.filled_shares, 0.)
        share_flow -= np.where(orders.side.eq("writeoff"), orders.requested_shares, 0.)
        flow = orders[["execution_date", "sleeve", "asset_id"]].assign(share_flow=share_flow)
        inventory = flow.pivot_table(index="execution_date", columns=["sleeve", "asset_id"], values="share_flow", aggfunc="sum", fill_value=0)
        inventory = inventory.reindex(daily.index, fill_value=0).cumsum()
        assert inventory.to_numpy().min() >= -1e-10
        valuation = close_prices.reindex(index=daily.index, columns=inventory.columns.get_level_values("asset_id"))
        # Missing asset quotes may be zero-valued ONLY after an explicit writeoff
        # has removed all its shares; no price forward fill is used here.
        missing = valuation.isna().to_numpy()
        assert np.max(np.abs(inventory.to_numpy()[missing]), initial=0) <= 1e-10
        expected_holdings = (inventory.to_numpy() * valuation.fillna(0).to_numpy()).sum(axis=1)
        close_enough(daily.holdings_value, expected_holdings)
        reconstructed_nav = expected_cash.to_numpy() + expected_holdings
        close_enough(daily.nav, reconstructed_nav)
        close_enough(daily.daily_return.iloc[1:], daily.nav.to_numpy()[1:] / daily.nav.to_numpy()[:-1] - 1)
        assert daily.filter(regex="_cash$").to_numpy().min() >= -1e-12

        trades = orders.loc[filled].merge(prices, on=["execution_date", "asset_id"], validate="many_to_one")
        close_enough(trades.fill_price, trades.actual_open)
        close_enough(trades.filled_notional, trades.filled_shares * trades.actual_open)
        trades = trades.merge(previous, on=["signal_date", "asset_id"], validate="many_to_one")
        if row.constraint_mode == "lagged":
            assert trades.loc[trades.side.eq("buy"), "can_buy"].all()
            assert trades.loc[trades.side.eq("sell"), "can_sell"].all()
        multiplier = 0 if row.cost_mode == "zero" else 1
        close_enough(trades.fee, multiplier * trades.filled_notional * trades.fee_bps_one_way / 10000)
        close_enough(trades.spread_cost, multiplier * trades.filled_notional * trades.spread_bps / 20000)
        close_enough(trades.impact_cost, multiplier * trades.filled_notional * trades.impact_bps_proxy / 10000)
        trades = trades.merge(actual, on=["execution_date", "asset_id"], validate="many_to_one")
        assert len(trades) == int(filled.sum()), "Market/constraint audit joins lost filled orders"
        conflict = np.where(trades.side.eq("buy"), ~trades.actual_buy, ~trades.actual_sell)
        assert np.array_equal(conflict, trades.actual_flag_conflict.to_numpy())
        close_enough(trades.loc[conflict, "filled_notional"].sum(), row.actual_flag_conflict_notional)
        assert int(conflict.sum()) == row.actual_flag_conflict_orders
        returns = daily.daily_return.iloc[1:]
        close_enough(row.total_return, daily.nav.iloc[-1] - 1)
        close_enough(row.annualized_return, daily.nav.iloc[-1] ** (252 / len(returns)) - 1)
        close_enough(row.sharpe, returns.mean() / returns.std(ddof=1) * np.sqrt(252))
        close_enough(row.max_drawdown, (daily.nav / daily.nav.cummax() - 1).min())
        close_enough(row.total_writeoff, orders.writeoff_amount.sum())
        records.append({
            "run": row.run, "dates": len(daily), "order_attempts": len(orders),
            "filled_orders": int(filled.sum()), "writeoff_events": int(orders.side.eq("writeoff").sum()),
            "max_cash_reconciliation_error": float(np.max(np.abs(daily.cash - expected_cash))),
            "max_nav_reconciliation_error": float(np.max(np.abs(daily.nav.to_numpy() - reconstructed_nav))),
            "costs_dates_prices_permissions_metrics_checked": True,
        })
    result = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "original_package_unchanged": True, "prior_results_unchanged": True,
        "temporal_fold_boundaries_passed": True,
        "temporal_predictions_include_unlabelled_signals": temporal_manifest["prediction_rows_without_label"],
        "audited_runs": records,
        "temporal_manifest_sha256": sha256(temporal / "run_manifest.json"),
        "portfolio_manifest_sha256": sha256(portfolio / "run_manifest.json"),
        "audit_code_sha256": sha256(Path(__file__)),
        "portfolio_protocol": portfolio_manifest["protocol"],
    }
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--portfolio", type=Path, default=PROJECT / "outputs" / "portfolio_upgrade")
    parser.add_argument("--temporal", type=Path, default=PROJECT / "outputs" / "temporal_oos")
    parser.add_argument("--output", type=Path, default=PROJECT / "outputs" / "upgrade_audit.json")
    args = parser.parse_args()
    run(args.root, args.portfolio, args.temporal, args.output)
