"""Independently reconstruct the frozen model-comparison portfolio accounts.

No portfolio simulation or portfolio-metric implementation is imported. Raw
prices, constraints, member keys, and saved order attempts are the audit inputs.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay


PROJECT = Path(__file__).resolve().parent
DEFAULT_ROOT = (PROJECT.parent / "Lab404-CUFEL-E" / "03-资产配置策略方向"
                / "Alpha_Forensics_Candidate_v1")
PREDICTORS = ["Ridge_ALL20", "HGB_ALL20", "Ridge_CONTROLS", "HGB_CONTROLS"]
KEYS = ["date", "asset_id"]
START, END = pd.Timestamp("2023-01-02"), pd.Timestamp("2024-12-31")
COST_COLUMNS = ["fee", "spread_cost", "impact_cost"]


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def equal(actual, expected) -> None:
    np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-10, equal_nan=True)


def verify_hashes(mapping: dict, base: Path) -> None:
    for name, expected in mapping.items():
        path = base / name
        if sha256(path) != expected:
            raise AssertionError(f"Changed artifact: {path}")


def verify_saved_hashes(out: Path, protocol: dict, manifest: dict) -> None:
    assert manifest["status"] == "completed"
    assert manifest["inputs_sha256"] == protocol["inputs_sha256"]
    assert manifest["code_sha256"] == protocol["code_sha256"]
    canonical = json.dumps(protocol, sort_keys=True, allow_nan=False).encode()
    assert manifest["protocol_sha256"] == hashlib.sha256(canonical).hexdigest()
    verify_hashes(protocol["inputs_sha256"], out)
    verify_hashes(protocol["code_sha256"], out)
    verify_hashes(manifest["outputs_sha256"], out)


def independent_metrics(daily: pd.DataFrame, orders: pd.DataFrame, period: str) -> dict:
    if period == "FULL":
        selected = daily
        initial = 1.0
    else:
        year = int(period)
        selected = daily.loc[daily.index.year == year]
        previous = daily.loc[daily.index < selected.index.min(), "nav"]
        initial = float(previous.iloc[-1]) if len(previous) else 1.0
    returns = selected.daily_return.dropna()
    order_slice = orders.loc[orders.execution_date.isin(selected.index)]
    filled = order_slice.filled_shares.gt(0)
    final = float(selected.nav.iloc[-1])
    volatility = float(returns.std(ddof=1))
    peaks = np.maximum.accumulate(np.r_[initial, selected.nav.to_numpy()])[1:]
    return {
        "initial_nav": initial, "final_nav": final, "n_return_days": len(returns),
        "total_return": final / initial - 1,
        "annualized_return": (final / initial) ** (252 / len(returns)) - 1,
        "annualized_volatility": volatility * np.sqrt(252),
        "sharpe": returns.mean() / volatility * np.sqrt(252) if volatility else np.nan,
        "max_drawdown": min(0., float((selected.nav.to_numpy() / peaks - 1).min())),
        "mean_daily_turnover": float(selected.loc[returns.index, "gross_turnover"].mean()),
        "total_fee": float(order_slice.fee.sum()),
        "total_spread_cost": float(order_slice.spread_cost.sum()),
        "total_impact_cost": float(order_slice.impact_cost.sum()),
        "total_writeoff": float(order_slice.writeoff_amount.sum()),
        "filled_orders": int(filled.sum()),
        "actual_flag_conflict_orders": int(order_slice.actual_flag_conflict.sum()),
        "mean_cash_fraction": float((selected.cash / selected.nav).mean()),
    }


def check_fields(saved, expected: dict) -> None:
    for key, value in expected.items():
        equal(np.nan if saved[key] is None else saved[key], value)


def audit_run(row, out, dates, market, constraints, expected_buys, periods):
    folder = out / "runs" / row.run
    daily = pd.read_csv(folder / "daily.csv", parse_dates=["date"]).set_index("date")
    orders = pd.read_csv(folder / "orders.csv.gz", parse_dates=[
        "signal_date", "execution_date", "scheduled_exit_date"])
    positions = pd.read_csv(folder / "positions.csv.gz", parse_dates=[
        "date", "entry_signal_date", "entry_date", "scheduled_exit_date"])
    saved_summary = json.loads((folder / "summary.json").read_text())
    pd.testing.assert_index_equal(daily.index, dates, check_names=False)
    assert np.isfinite(daily.nav).all() and (daily.nav > 0).all()
    assert orders.execution_date.isin(dates).all()
    assert (orders.signal_date + BDay(1) == orders.execution_date).all()
    assert orders.side.isin(["buy", "sell", "writeoff"]).all()
    buys, sells = orders.side.eq("buy"), orders.side.eq("sell")
    filled = orders.filled_shares.gt(0)
    assert orders.filled_shares.ge(0).all() and orders.filled_notional.ge(0).all()
    assert orders.loc[filled, "reason"].eq("filled").all()
    assert not orders.loc[~filled, "reason"].eq("filled").any()
    equal(orders.loc[~filled, ["filled_notional"] + COST_COLUMNS], 0)
    assert orders.loc[~filled, "actual_flag_conflict"].eq(False).all()
    actual_buys = orders.loc[buys, ["signal_date", "sleeve", "asset_id"]]
    assert not actual_buys.duplicated().any()
    pd.testing.assert_frame_equal(
        actual_buys.sort_values(["signal_date", "asset_id"]).reset_index(drop=True),
        expected_buys.sort_values(["signal_date", "asset_id"]).reset_index(drop=True))
    assert (orders.loc[buys, "scheduled_exit_date"] == orders.loc[buys, "execution_date"] + BDay(5)).all()
    assert (orders.loc[sells, "execution_date"] >= orders.loc[sells, "scheduled_exit_date"]).all()

    # Cash is reconstructed independently for each of the five sleeves.
    cost = orders[COST_COLUMNS].sum(axis=1)
    flow = np.where(sells, orders.filled_notional, 0) - np.where(buys, orders.filled_notional, 0) - cost
    cash_flow = orders[["execution_date", "sleeve"]].assign(flow=flow).pivot_table(
        index="execution_date", columns="sleeve", values="flow", aggfunc="sum", fill_value=0)
    cash_flow = cash_flow.reindex(index=dates, columns=range(5), fill_value=0)
    cash = .2 + cash_flow.cumsum()
    assert cash.to_numpy().min() >= -1e-10
    equal(daily.cash, cash.sum(axis=1))
    share_flow = np.where(buys, orders.filled_shares, 0) - np.where(sells, orders.filled_shares, 0)
    share_flow -= np.where(orders.side.eq("writeoff"), orders.requested_shares, 0)
    inventory = orders[["execution_date", "sleeve", "asset_id"]].assign(flow=share_flow).pivot_table(
        index="execution_date", columns=["sleeve", "asset_id"], values="flow", aggfunc="sum", fill_value=0)
    inventory = inventory.reindex(dates, fill_value=0).cumsum()
    assert inventory.to_numpy().min() >= -1e-10
    close_prices = market.pivot(index="date", columns="asset_id", values="close")
    valuation = close_prices.reindex(index=dates, columns=inventory.columns.get_level_values("asset_id"))
    missing = valuation.isna().to_numpy()
    assert np.max(np.abs(inventory.to_numpy()[missing]), initial=0) <= 1e-10
    market_value = inventory.to_numpy() * valuation.fillna(0).to_numpy()
    equal(daily.holdings_value, market_value.sum(axis=1))
    reconstructed_nav = cash.sum(axis=1).to_numpy() + market_value.sum(axis=1)
    equal(daily.nav, reconstructed_nav)
    assert pd.isna(daily.daily_return.iloc[0])
    equal(daily.daily_return.iloc[1:], reconstructed_nav[1:] / reconstructed_nav[:-1] - 1)
    for sleeve in range(5):
        sleeve_value = market_value[:, inventory.columns.get_level_values("sleeve") == sleeve].sum(axis=1)
        equal(daily[f"sleeve_{sleeve}_cash"], cash[sleeve])
        equal(daily[f"sleeve_{sleeve}_market_value"], sleeve_value)
        equal(daily[f"sleeve_{sleeve}_nav"], cash[sleeve] + sleeve_value)

    # Saved positions must reproduce every date/asset/sleeve inventory, not just NAV.
    assert not positions.duplicated(["date", "sleeve", "asset_id"]).any()
    recorded_inventory = positions.pivot(index="date", columns=["sleeve", "asset_id"], values="shares")
    assert recorded_inventory.columns.isin(inventory.columns).all()
    recorded_inventory = recorded_inventory.reindex(index=dates, columns=inventory.columns).fillna(0)
    equal(recorded_inventory, inventory)
    valued_positions = positions.merge(market[KEYS + ["close"]], on=KEYS,
                                      how="left", validate="many_to_one", suffixes=("", "_raw"))
    assert valued_positions.close_raw.notna().all()
    equal(valued_positions.close, valued_positions.close_raw)
    equal(valued_positions.market_value, valued_positions.shares * valued_positions.close_raw)

    # The day before execution supplies permission and cost; open supplies price.
    trade = orders.loc[filled].merge(
        market[["date", "asset_id", "open"]].rename(columns={"date": "execution_date"}),
        on=["execution_date", "asset_id"], how="left", validate="many_to_one")
    assert trade.open.notna().all()
    equal(trade.fill_price, trade.open)
    equal(trade.filled_notional, trade.filled_shares * trade.open)
    trade = trade.merge(constraints.rename(columns={"date": "signal_date"}),
                        on=["signal_date", "asset_id"], how="left", validate="many_to_one")
    assert trade.loc[trade.side.eq("buy"), "can_buy"].eq(True).all()
    assert trade.loc[trade.side.eq("sell"), "can_sell"].eq(True).all()
    for column, proxy, divisor in [("fee", "fee_bps_one_way", 10000),
                                   ("spread_cost", "spread_bps", 20000),
                                   ("impact_cost", "impact_bps_proxy", 10000)]:
        equal(trade[column], trade.filled_notional * trade[proxy] / divisor * row.cost_multiplier)
    actual = constraints[["date", "asset_id", "can_buy", "can_sell"]].rename(columns={
        "date": "execution_date", "can_buy": "actual_buy", "can_sell": "actual_sell"})
    trade = trade.merge(actual, on=["execution_date", "asset_id"], how="left", validate="many_to_one")
    conflicts = np.where(trade.side.eq("buy"), ~trade.actual_buy, ~trade.actual_sell)
    assert np.array_equal(trade.actual_flag_conflict.to_numpy(), conflicts)

    # Budget planning uses the signal close and scales to opening cash after exits.
    buy_orders = orders.loc[buys].copy()
    counts = buy_orders.groupby("signal_date").size()
    for signal_day, group in buy_orders.groupby("signal_date", sort=False):
        sleeve = int(group.sleeve.iloc[0])
        day = signal_day + BDay(1)
        planned = daily.loc[signal_day, f"sleeve_{sleeve}_nav"]
        opening_sells = orders.loc[sells & orders.execution_date.eq(day) & orders.sleeve.eq(sleeve)]
        available = cash.loc[signal_day, sleeve] + opening_sells.filled_notional.sum() - opening_sells[COST_COLUMNS].to_numpy().sum()
        equal(group.planned_budget, planned / counts[signal_day])
        equal(group.requested_budget, min(planned, available) / counts[signal_day])

    writeoffs = orders.loc[orders.side.eq("writeoff")]
    market_last = market.groupby("asset_id").date.max()
    for event in writeoffs.itertuples():
        assert market_last[event.asset_id] < event.execution_date
        before = event.execution_date - BDay(1)
        equal(event.requested_shares, inventory.loc[before, (event.sleeve, event.asset_id)])
        equal(event.writeoff_amount, event.requested_shares * close_prices.loc[before, event.asset_id])
        assert pd.isna(event.fill_price)
    assert orders.loc[~orders.side.eq("writeoff"), "writeoff_amount"].eq(0).all()

    for order_field, daily_field in [("filled_notional", "gross_traded_notional"),
                                     ("fee", "fee"), ("spread_cost", "spread_cost"),
                                     ("impact_cost", "impact_cost"), ("writeoff_amount", "writeoff")]:
        expected = orders.groupby("execution_date")[order_field].sum().reindex(dates, fill_value=0)
        equal(daily[daily_field], expected)
    conflict_daily = trade.loc[trade.actual_flag_conflict].groupby("execution_date").filled_notional.agg(["size", "sum"]).reindex(dates, fill_value=0)
    equal(daily.actual_flag_conflict_count, conflict_daily["size"])
    equal(daily.actual_flag_conflict_notional, conflict_daily["sum"])
    equal(daily.gross_turnover, daily.gross_traded_notional / np.r_[1., reconstructed_nav[:-1]])
    equal(daily.open_position_count, positions.groupby("date").size().reindex(dates, fill_value=0))
    common = independent_metrics(daily, orders, "FULL")
    check_fields(row._asdict(), common)
    check_fields(saved_summary, {key: value for key, value in common.items() if key not in ["filled_orders", "mean_cash_fraction"]})
    for period in ["FULL", "2023", "2024"]:
        check_fields(periods.loc[(row.run, period)], independent_metrics(daily, orders, period))
    extra = {
        "cumulative_gross_turnover": daily.gross_turnover.sum(),
        "gross_traded_notional": orders.filled_notional.sum(),
        "actual_flag_conflict_notional": trade.loc[conflicts, "filled_notional"].sum(),
        "end_open_positions": int(daily.open_position_count.iloc[-1]),
        "end_holdings_value": daily.holdings_value.iloc[-1], "end_cash": daily.cash.iloc[-1],
    }
    check_fields(row._asdict(), extra)
    check_fields(saved_summary, extra)
    paired = trade.pivot_table(index=["execution_date", "sleeve", "asset_id"],
                              columns="side", values="filled_notional", aggfunc="sum", fill_value=0)
    overlap = float(paired[["buy", "sell"]].min(axis=1).sum() * 2)
    return {
        "run": row.run, "order_attempts": len(orders), "filled_orders": int(filled.sum()),
        "writeoff_events": len(writeoffs), "signal_list_checked": True,
        "cash_inventory_nav_prices_costs_permissions_checked": True,
        "full_and_annual_metrics_checked": True,
        "max_cash_reconciliation_error": float(np.max(np.abs(daily.cash - cash.sum(axis=1)))),
        "max_nav_reconciliation_error": float(np.max(np.abs(daily.nav - reconstructed_nav))),
        "same_sleeve_same_day_offsettable_two_way_notional": overlap,
        "offsettable_share_of_gross_notional": overlap / orders.filled_notional.sum(),
        "offsetting_diagnostic_note": "Ex-post overlap only; not a netted-strategy return estimate",
    }


def run(root: Path, out: Path) -> None:
    destination = out / "verification.json"
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite verification: {destination}")
    protocol = json.loads((out / "protocol.json").read_text())
    manifest = json.loads((out / "run_manifest.json").read_text())
    verify_saved_hashes(out, protocol, manifest)
    dates = pd.bdate_range(START, END, name="date")
    market = pd.read_csv(root / "data" / "market_daily.csv.gz", parse_dates=["date"])
    constraints = pd.read_csv(root / "data" / "trading_constraints.csv.gz", parse_dates=["date"])
    universe = pd.read_csv(root / "data" / "universe.csv.gz", parse_dates=["date"])
    members = universe.loc[universe.is_listed & universe.is_member & universe.date.between(START, END), KEYS].sort_values(KEYS).reset_index(drop=True)
    assert not members.duplicated(KEYS).any()
    lists, coverage = {}, []
    saved_coverage = pd.read_csv(out / "signal_coverage.csv").set_index(["predictor", "year"], verify_integrity=True)
    fits = pd.read_csv(out / "fits.csv", parse_dates=["max_training_label_exit", "first_signal"]).set_index(["predictor", "year"], verify_integrity=True)
    expected_fits = {(name, year) for name in PREDICTORS for year in [2023, 2024]}
    assert set(saved_coverage.index) == set(fits.index) == expected_fits
    opens = market.set_index(KEYS, verify_integrity=True).open
    signal_days = dates[dates <= END - BDay(6)]
    for predictor in PREDICTORS:
        frame = pd.read_csv(out / "predictions" / f"{predictor}.csv.gz", parse_dates=["date", "train_cutoff"])
        frame = frame.sort_values(KEYS).reset_index(drop=True)
        pd.testing.assert_frame_equal(frame[KEYS], members)
        assert np.isfinite(frame.prediction).all()
        assert (frame.train_cutoff < frame.date).all()
        expected_cutoff = np.where(frame.date.dt.year.eq(2023), pd.Timestamp("2022-12-30"), pd.Timestamp("2023-12-29"))
        assert np.array_equal(frame.train_cutoff.to_numpy(), pd.to_datetime(expected_cutoff).to_numpy())
        entry_keys = pd.MultiIndex.from_arrays([frame.date + BDay(1), frame.asset_id], names=KEYS)
        exit_keys = pd.MultiIndex.from_arrays([frame.date + BDay(6), frame.asset_id], names=KEYS)
        labels = opens.reindex(exit_keys).to_numpy() / opens.reindex(entry_keys).to_numpy() - 1
        equal(frame.forward_5d, labels)
        for year in [2023, 2024]:
            yearly = frame.loc[frame.date.dt.year.eq(year)]
            saved = saved_coverage.loc[(predictor, year)]
            equal(saved.signal_rows, len(yearly))
            equal(saved.signal_dates, yearly.date.nunique())
            equal(saved.missing_labels, yearly.forward_5d.isna().sum())
            fit = fits.loc[(predictor, year)]
            assert fit.max_training_label_exit <= yearly.train_cutoff.iloc[0] < fit.first_signal
            assert fit.first_signal == yearly.date.min()
            assert fit.max_archived_prediction_difference <= 1e-10
            stage = "validation" if year == 2023 else "confirmation"
            archive_name = (predictor.replace("_ALL20", "") + "_predictions.csv.gz")
            archives = [Path(path) for path in protocol["inputs_sha256"]
                        if Path(path).name == archive_name and Path(path).parent.name == stage]
            assert len(archives) == 1
            archived = pd.read_csv(archives[0], parse_dates=["date"])
            match = archived[KEYS + ["prediction"]].merge(yearly[KEYS + ["prediction"]],
                    on=KEYS, how="left", validate="one_to_one", suffixes=("_old", "_new"))
            assert match.prediction_new.notna().all()
            equal(match.prediction_old, match.prediction_new)
            equal(saved.archived_rows, len(archived))
            equal(saved.added_rows, len(yearly) - len(archived))
        chosen = frame.loc[frame.date.isin(signal_days)].copy()
        threshold = chosen.groupby("date").prediction.transform(lambda x: x.quantile(.9))
        chosen = chosen.loc[chosen.prediction.ge(threshold), KEYS].rename(columns={"date": "signal_date"})
        chosen["sleeve"] = dates.get_indexer(chosen.signal_date) % 5
        lists[predictor] = chosen[["signal_date", "sleeve", "asset_id"]]
        coverage.append({"predictor": predictor, "rows": len(frame),
                         "dates": int(frame.date.nunique()),
                         "rows_without_future_label": int(frame.forward_5d.isna().sum())})
    equal_weight = members.loc[members.date.isin(signal_days)].rename(columns={"date": "signal_date"})
    equal_weight["sleeve"] = dates.get_indexer(equal_weight.signal_date) % 5
    lists["EqualWeight"] = equal_weight[["signal_date", "sleeve", "asset_id"]]
    summary = pd.read_csv(out / "summary.csv")
    expected_runs = {f"{predictor}__cost{multiplier}" for predictor in PREDICTORS + ["EqualWeight"] for multiplier in [0, 1, 2]}
    assert set(summary.run) == expected_runs and len(summary) == len(expected_runs)
    periods = pd.read_csv(out / "period_metrics.csv", dtype={"period": str}).set_index(["run", "period"], verify_integrity=True)
    assert set(periods.index) == {(name, period) for name in expected_runs for period in ["FULL", "2023", "2024"]}
    records = []
    for row in summary.itertuples(index=False):
        assert row.run == f"{row.predictor}__cost{int(row.cost_multiplier)}"
        assert row.cost_mode == "base" and row.constraint_mode == "lagged"
        assert row.selection == ("equal_weight" if row.predictor == "EqualWeight" else "top_decile")
        print(f"Independent portfolio audit: {row.run}", flush=True)
        records.append(audit_run(row, out, dates, market, constraints,
                                 lists[row.predictor], periods))
    for (name, period), values in periods.iterrows():
        predictor, cost = name.split("__")
        equal(values.relative_terminal_wealth_vs_equalweight,
              (1 + values.total_return) / (1 + periods.loc[(f"EqualWeight__{cost}", period), "total_return"]) - 1)
        if not predictor.endswith("ALL20"):
            assert pd.isna(values.relative_terminal_wealth_vs_matching_controls)
        else:
            controls = predictor.split("_")[0] + "_CONTROLS"
            equal(values.relative_terminal_wealth_vs_matching_controls,
                  (1 + values.total_return) / (1 + periods.loc[(f"{controls}__{cost}", period), "total_return"]) - 1)
    # Recheck after all reads, so a concurrent edit cannot silently pass the audit.
    verify_saved_hashes(out, protocol, manifest)
    result = {
        "verified_at_utc": datetime.now(timezone.utc).isoformat(), "status": "passed",
        "input_code_output_hashes_checked": True, "complete_member_signal_coverage": coverage,
        "audited_runs": records, "run_manifest_sha256": sha256(out / "run_manifest.json"),
        "audit_code_sha256": sha256(Path(__file__)),
    }
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"Verified {len(records)} runs; saved {destination}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out", type=Path, default=PROJECT / "outputs" / "portfolio_model_comparison")
    args = parser.parse_args()
    run(args.root, args.out)
