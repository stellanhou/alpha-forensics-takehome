"""Independent accounting and execution-contract audit of the rebalance study.

The audit imports neither the trading engine nor its policy/metric functions.
Raw market arrays and score cross sections are prepared once for all runs.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
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
START, END = pd.Timestamp("2023-01-02"), pd.Timestamp("2024-12-31")
KEYS = ["date", "asset_id"]
COSTS = ["fee", "spread_cost", "impact_cost"]
PREDICTORS = ["Ridge_ALL20", "HGB_ALL20", "Ridge_CONTROLS", "HGB_CONTROLS"]
POLICIES = ["NET", "RANK", "WEIGHT", "BOTH", "EMA50", "TAPER", "SCORE", "DAILY_NET"]
ATOL = 1e-10
_AUDIT_WORKER = {}


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def equal(actual, expected, label="") -> None:
    np.testing.assert_allclose(np.asarray(actual, dtype=float), np.asarray(expected, dtype=float), rtol=1e-9, atol=ATOL,
                               equal_nan=True, err_msg=label)


def check_fields(saved, expected: dict) -> None:
    for key, value in expected.items():
        equal(np.nan if saved[key] is None else saved[key], value, key)


def verify_hashes(out: Path, protocol: dict, manifest: dict) -> None:
    assert manifest["status"] == "completed"
    assert manifest["inputs_sha256"] == protocol["inputs_sha256"]
    assert manifest["code_sha256"] == protocol["code_sha256"]
    canonical = json.dumps(protocol, sort_keys=True, allow_nan=False).encode()
    assert manifest["protocol_sha256"] == hashlib.sha256(canonical).hexdigest()
    for mapping in [protocol["inputs_sha256"], protocol["code_sha256"], manifest["outputs_sha256"]]:
        for name, digest in mapping.items():
            if sha256(out / name) != digest:
                raise AssertionError(f"Changed artifact: {out / name}")


@dataclass
class MarketContext:
    dates: pd.DatetimeIndex
    assets: pd.Index
    opening: np.ndarray
    closing: np.ndarray
    prior_buy: np.ndarray
    prior_sell: np.ndarray
    prior_cost: np.ndarray
    actual_buy: np.ndarray
    actual_sell: np.ndarray
    members: np.ndarray
    scores: dict[str, list[pd.Series]]


def prepare_context(market: pd.DataFrame, constraints: pd.DataFrame,
                    universe: pd.DataFrame, predictions: dict[str, pd.DataFrame],
                    start=START, end=END) -> MarketContext:
    """Prepare once; also exposed for small independent audit fixtures."""
    dates = pd.bdate_range(start, end, name="date")
    assets = pd.Index(sorted(market.asset_id.unique()), name="asset_id")
    for frame in [market, constraints, universe]:
        assert not frame.duplicated(KEYS).any()
    grid = pd.MultiIndex.from_product([dates, assets], names=KEYS)
    previous_grid = pd.MultiIndex.from_product([dates - BDay(1), assets], names=KEYS)
    prices = market.set_index(KEYS)
    conditions = constraints.set_index(KEYS)
    membership = universe.set_index(KEYS)
    shape = (len(dates), len(assets))

    def values(frame, column, index):
        return frame[column].reindex(index).to_numpy().reshape(shape)

    opening, closing = (values(prices, key, grid).astype(float) for key in ["open", "close"])
    members = (membership.is_listed & membership.is_member).reindex(grid, fill_value=False).to_numpy().reshape(shape)
    scores = {}
    expected_keys = grid[members.ravel()]
    for name, frame in predictions.items():
        indexed = frame.set_index(KEYS, verify_integrity=True).sort_index()
        pd.testing.assert_index_equal(indexed.index, expected_keys)
        assert np.isfinite(indexed.prediction).all()
        scores[name] = [indexed.xs(day).prediction for day in dates]
    scores["EqualWeight"] = [pd.Series(0., index=assets[members[d]]) for d in range(len(dates))]
    prior_cost = np.stack([values(conditions, column, previous_grid).astype(float) / divisor
                           for column, divisor in [("fee_bps_one_way", 10000),
                                                   ("spread_bps", 20000),
                                                   ("impact_bps_proxy", 10000)]], axis=2)
    return MarketContext(dates, assets, opening, closing,
                         values(conditions, "can_buy", previous_grid),
                         values(conditions, "can_sell", previous_grid), prior_cost,
                         values(conditions, "can_buy", grid),
                         values(conditions, "can_sell", grid), members, scores)


def raw_policy_weights(policy: str, scores: pd.Series, held: set,
                       previous_target: dict, equal_weight=False) -> dict:
    """Implement the written selection contract without engine helpers."""
    if equal_weight:
        base = {asset: 1 / len(scores) for asset in scores.index}
    else:
        candidates = scores.loc[scores.ge(scores.quantile(.9))]
        if policy in ["RANK", "BOTH"]:
            kept = {asset for asset in held if asset in scores.index and scores[asset] >= scores.quantile(.8)}
            needed = int(np.ceil(len(scores) * .1))
            for score in sorted(candidates.unique(), reverse=True):
                if len(kept) >= needed:
                    break
                kept.update(candidates.index[candidates.eq(score)])
            candidates = scores.loc[sorted(kept)]
        if policy == "TAPER":
            raw = len(candidates) + 1 - candidates.rank(ascending=False, method="average")
            base = (raw / raw.sum()).to_dict()
        elif policy == "SCORE":
            if scores.nunique() == 1:
                return {}
            positive = (candidates - scores.mean()).clip(lower=0)
            cap = 2 / len(candidates)
            # Water-filling by remaining capacity; zero excess scores stay cash.
            base = {asset: 0. for asset in candidates.index}
            active = positive.loc[positive.gt(0)].copy()
            remaining = min(1., len(active) * cap)
            while len(active) and remaining > 0:
                proposal = remaining * active / active.sum()
                capped = proposal.ge(cap)
                if not capped.any():
                    base.update(proposal.to_dict())
                    break
                for asset in active.index[capped]:
                    base[asset] = cap
                remaining -= float(capped.sum()) * cap
                active = active.loc[~capped]
            base = {asset: weight for asset, weight in base.items() if weight > 0}
        else:
            base = {asset: 1 / len(candidates) for asset in candidates.index}
    if policy == "EMA50":
        eligible = set(scores.index)
        return {asset: .5 * previous_target.get(asset, 0.) + .5 * base.get(asset, 0.)
                for asset in sorted((set(base) | set(previous_target)) & eligible)
                if previous_target.get(asset, 0.) + base.get(asset, 0.) > 0}
    return base


def independent_metrics(daily: pd.DataFrame, orders: pd.DataFrame, exposure: pd.DataFrame,
                        period: str) -> dict:
    selected = daily if period == "FULL" else daily.loc[daily.index.year == int(period)]
    previous = daily.loc[daily.index < selected.index[0], "nav"]
    initial = float(previous.iloc[-1]) if len(previous) else 1.
    final = float(selected.nav.iloc[-1])
    returns = selected.daily_return.dropna()
    volatility = float(returns.std(ddof=1))
    path = np.r_[initial, selected.nav.to_numpy()]
    in_period = orders.loc[orders.execution_date.isin(selected.index)]
    measured = exposure.loc[selected.index]
    mean_exposure = float(measured.gross_exposure.mean())
    mean_turnover = float(selected.loc[returns.index, "gross_turnover"].mean())
    return {
        "initial_nav": initial, "final_nav": final, "n_return_days": len(returns),
        "total_return": final / initial - 1,
        "annualized_return": (final / initial) ** (252 / len(returns)) - 1,
        "annualized_volatility": volatility * np.sqrt(252),
        "sharpe": float(returns.mean() / volatility * np.sqrt(252)) if volatility else np.nan,
        "max_drawdown": float((path / np.maximum.accumulate(path) - 1).min()),
        "mean_daily_turnover": mean_turnover,
        "total_fee": float(in_period.fee.sum()), "total_spread_cost": float(in_period.spread_cost.sum()),
        "total_impact_cost": float(in_period.impact_cost.sum()),
        "total_writeoff": float(in_period.writeoff_amount.sum()),
        "filled_orders": int(in_period.filled_shares.gt(0).sum()),
        "actual_flag_conflict_orders": int(in_period.actual_flag_conflict.sum()),
        "mean_cash_fraction": float((selected.cash / selected.nav).mean()),
        "mean_asset_count": float(measured.asset_count.mean()),
        "mean_hhi": float(measured.hhi.mean()),
        "max_asset_weight": float(measured.max_asset_weight.max()),
        "mean_gross_exposure": mean_exposure,
        "turnover_per_mean_exposure": mean_turnover / mean_exposure if mean_exposure else np.nan,
    }


def reconstruct_accounts(daily, orders, positions, context, sleeves, multiplier):
    """Only recorded signed flows update inventory and cash in this audit."""
    days, assets = len(context.dates), len(context.assets)
    oi = context.dates.get_indexer(orders.execution_date)
    ai = context.assets.get_indexer(orders.asset_id)
    si = orders.sleeve.to_numpy(dtype=int)
    assert (oi >= 0).all() and (ai >= 0).all() and ((si >= 0) & (si < sleeves)).all()
    buys, sells = orders.side.eq("buy").to_numpy(), orders.side.eq("sell").to_numpy()
    writeoffs = orders.side.eq("writeoff").to_numpy()
    assert (buys | sells | writeoffs).all()
    filled = orders.filled_shares.gt(0).to_numpy()
    assert orders.filled_shares.ge(0).all() and orders.filled_notional.ge(0).all()
    assert orders.loc[filled, "reason"].eq("filled").all()
    assert not orders.loc[~filled, "reason"].eq("filled").any()
    equal(orders.loc[~filled, ["filled_notional"] + COSTS], 0)
    assert not orders.loc[~filled, "actual_flag_conflict"].any()
    assert (orders.signal_date + BDay(1) == orders.execution_date).all()
    assert orders.scheduled_exit_date.isna().all()
    equal(orders.loc[filled, "fill_price"], context.opening[oi[filled], ai[filled]])
    equal(orders.loc[filled, "filled_notional"], orders.loc[filled, "filled_shares"] * context.opening[oi[filled], ai[filled]])
    assert np.all(context.prior_buy[oi[filled & buys], ai[filled & buys]] == True)
    assert np.all(context.prior_sell[oi[filled & sells], ai[filled & sells]] == True)
    expected_costs = orders.loc[filled, "filled_notional"].to_numpy()[:, None] * context.prior_cost[oi[filled], ai[filled]] * multiplier
    equal(orders.loc[filled, COSTS], expected_costs)
    actual_allowed = np.where(buys[filled], context.actual_buy[oi[filled], ai[filled]], context.actual_sell[oi[filled], ai[filled]])
    assert np.array_equal(orders.loc[filled, "actual_flag_conflict"].to_numpy(), actual_allowed == False)
    equal(orders.loc[~writeoffs, "writeoff_amount"], 0)
    share_flow = orders.filled_shares.to_numpy() * (buys.astype(int) - sells.astype(int))
    share_flow[writeoffs] = -orders.loc[writeoffs, "requested_shares"].to_numpy()
    flows = np.zeros((days, sleeves, assets))
    np.add.at(flows, (oi, si, ai), share_flow)
    inventory = flows.cumsum(axis=0)
    assert inventory.min(initial=0) >= -ATOL
    cash_delta = orders.filled_notional.to_numpy() * (sells.astype(int) - buys.astype(int)) - orders[COSTS].sum(axis=1).to_numpy()
    cash_flows = np.zeros((days, sleeves))
    np.add.at(cash_flows, (oi, si), cash_delta)
    cash = 1 / sleeves + cash_flows.cumsum(axis=0)
    assert cash.min(initial=0) >= -ATOL
    missing = np.isnan(context.closing)
    assert np.max(np.abs(inventory * missing[:, None, :]), initial=0) <= ATOL
    holdings_value = inventory * np.nan_to_num(context.closing)[:, None, :]
    nav = cash.sum(axis=1) + holdings_value.sum(axis=(1, 2))
    equal(daily.cash, cash.sum(axis=1))
    equal(daily.holdings_value, holdings_value.sum(axis=(1, 2)))
    equal(daily.nav, nav)
    assert np.isfinite(nav).all() and (nav > 0).all()
    assert pd.isna(daily.daily_return.iloc[0])
    equal(daily.daily_return.iloc[1:], nav[1:] / nav[:-1] - 1)
    for sleeve in range(sleeves):
        equal(daily[f"sleeve_{sleeve}_cash"], cash[:, sleeve])
        equal(daily[f"sleeve_{sleeve}_market_value"], holdings_value[:, sleeve].sum(axis=1))
        equal(daily[f"sleeve_{sleeve}_nav"], cash[:, sleeve] + holdings_value[:, sleeve].sum(axis=1))
    assert not positions.duplicated(["date", "sleeve", "asset_id"]).any()
    pi, pa = context.dates.get_indexer(positions.date), context.assets.get_indexer(positions.asset_id)
    ps = positions.sleeve.to_numpy(dtype=int)
    assert (pi >= 0).all() and (pa >= 0).all() and ((ps >= 0) & (ps < sleeves)).all()
    assert positions.shares.gt(0).all()
    reported_inventory = np.zeros_like(inventory)
    np.add.at(reported_inventory, (pi, ps, pa), positions.shares.to_numpy())
    equal(reported_inventory, inventory)
    equal(positions.close, context.closing[pi, pa])
    equal(positions.market_value, positions.shares.to_numpy() * context.closing[pi, pa])
    equal(daily.open_position_count, np.bincount(pi, minlength=days))
    for field, output in [("filled_notional", "gross_traded_notional"), ("fee", "fee"),
                           ("spread_cost", "spread_cost"), ("impact_cost", "impact_cost"),
                           ("writeoff_amount", "writeoff")]:
        equal(daily[output], np.bincount(oi, weights=orders[field], minlength=days))
    conflict = orders.actual_flag_conflict.to_numpy(dtype=bool)
    equal(daily.actual_flag_conflict_count, np.bincount(oi[conflict], minlength=days))
    equal(daily.actual_flag_conflict_notional, np.bincount(oi[conflict], weights=orders.loc[conflict, "filled_notional"], minlength=days))
    equal(daily.gross_turnover, daily.gross_traded_notional / np.r_[1., nav[:-1]])
    assert daily.cash_roundoff_adjustment.ge(0).all() and daily.cash_roundoff_adjustment.le(1e-12).all()
    # Exposure aggregates the same asset across sleeves before measuring HHI.
    asset_values = holdings_value.sum(axis=1)
    weights = asset_values / nav[:, None]
    counts = np.zeros((days, assets), dtype=bool)
    counts[pi, pa] = True
    exposure = pd.DataFrame({"asset_count": counts.sum(axis=1),
                             "hhi": (weights ** 2).sum(axis=1),
                             "max_asset_weight": weights.max(axis=1),
                             "gross_exposure": weights.sum(axis=1)}, index=context.dates)
    return reported_inventory, cash, exposure


def audit_plans(row, plans, reviews, orders, inventory, cash, context):
    """Rebuild close decisions and compare quantity intents with recorded orders."""
    sleeves = inventory.shape[1]
    last_signal = context.dates[-1] - BDay(6)
    signal_indices = [i for i, day in enumerate(context.dates[:-1])
                      if row.policy != "DAILY_NET" or day <= last_signal
                      or day == context.dates[-1] - BDay(1)]
    expected_reviews = pd.DataFrame({
        "signal_date": context.dates[signal_indices],
        "execution_date": context.dates[np.asarray(signal_indices) + 1],
        "sleeve": np.asarray(signal_indices) % sleeves,
        "terminal": context.dates[signal_indices] > last_signal,
    })
    assert not reviews.duplicated(["execution_date", "sleeve"]).any()
    pd.testing.assert_frame_equal(reviews[expected_reviews.columns].reset_index(drop=True), expected_reviews)
    assert not plans.duplicated(["execution_date", "sleeve", "asset_id"]).any()
    assert plans.execution_date.isin(reviews.execution_date).all()
    assert (plans.signal_date + BDay(1) == plans.execution_date).all()
    assert not orders.duplicated(["execution_date", "sleeve", "asset_id", "side"]).any()
    plan_groups = {day: group.set_index("asset_id") for day, group in plans.groupby("execution_date", sort=False)}
    review_groups = {review.execution_date: review for review in reviews.itertuples(index=False)}
    order_groups = {day: list(group.itertuples(index=False)) for day, group in orders.groupby("execution_date", sort=False)}
    asset_indices = {asset: i for i, asset in enumerate(context.assets)}
    targets, previous_targets = [{} for _ in range(sleeves)], [{} for _ in range(sleeves)]
    target_dates = [pd.NaT] * sleeves
    ordinary_count = terminal_count = 0
    for d, day in enumerate(context.dates):
        start_holdings = inventory[d - 1].copy() if d else np.zeros_like(inventory[0])
        previous_cash = cash[d - 1].copy() if d else np.full(sleeves, 1 / sleeves)
        today = order_groups.get(day, [])
        lookup = {(order.sleeve, order.asset_id, order.side): order for order in today}
        expected_writeoffs = {(sleeve, context.assets[a], "writeoff")
                              for sleeve, a in zip(*np.where((start_holdings > 0) & np.isnan(context.opening[d])[None, :]))}
        assert {key for key in lookup if key[2] == "writeoff"} == expected_writeoffs
        for sleeve, asset, _ in expected_writeoffs:
            a = asset_indices[asset]
            order = lookup[(sleeve, asset, "writeoff")]
            equal(order.requested_shares, start_holdings[sleeve, a])
            equal(order.current_quantity, start_holdings[sleeve, a])
            equal(order.desired_trade_quantity, -start_holdings[sleeve, a])
            equal(order.writeoff_amount, start_holdings[sleeve, a] * context.closing[d - 1, a])
            equal(order.requested_notional, order.writeoff_amount)
            assert order.plan_signal_date == target_dates[sleeve]
            assert pd.isna(order.fill_price) and np.isnan(context.opening[d:, a]).all()
            assert order.reason == "missing_market_writeoff"
            start_holdings[sleeve, a] = 0.

        review = review_groups.get(day)
        plan = None
        if review is not None:
            sleeve = int(review.sleeve)
            signal_holdings = inventory[d - 1, sleeve]
            held = set(context.assets[signal_holdings > 0])
            signal_nav = previous_cash[sleeve] + np.nansum(signal_holdings * context.closing[d - 1])
            ideal = {} if review.terminal else raw_policy_weights(
                row.policy, context.scores[row.predictor][d - 1], held,
                previous_targets[sleeve], row.predictor == "EqualWeight")
            previous_targets[sleeve] = ideal
            # A no-position/no-target review is deliberately still required.
            plan = plan_groups.get(day, pd.DataFrame(columns=plans.columns).set_index("asset_id"))
            assert set(plan.index) == held | set(ideal)
            if len(plan):
                assert plan.sleeve.eq(sleeve).all() and plan.terminal.eq(review.terminal).all()
                assert plan.signal_date.eq(review.signal_date).all()
                equal(plan.signal_sleeve_nav, signal_nav)
            opening_nav = previous_cash[sleeve] + np.nansum(start_holdings[sleeve] * context.opening[d])
            budget = min(signal_nav, opening_nav)
            equal(review.signal_nav, signal_nav)
            equal(review.opening_nav, opening_nav)
            equal(review.execution_budget, budget)
            equal(review.desired_asset_count, len(ideal))
            new_targets = {}
            for asset, values in plan.iterrows():
                a = asset_indices[asset]
                shares, close = signal_holdings[a], context.closing[d - 1, a]
                wanted = ideal.get(asset, 0.)
                effective, freeze = wanted, False
                if row.policy in ["WEIGHT", "BOTH"] and wanted > 0 and shares > 0:
                    actual = shares * close / float(review.signal_nav)
                    lower, upper = wanted * .9, wanted * 1.1
                    # Saved decimal values may round an exact boundary by an ulp.
                    boundary = abs(actual - lower) <= 1e-14 or abs(actual - upper) <= 1e-14
                    if lower + 1e-14 < actual < upper - 1e-14:
                        effective, freeze = actual, True
                    elif boundary:
                        freeze = bool(values.freeze_quantity)
                        effective = actual if freeze else (lower if abs(actual - lower) <= 1e-14 else upper)
                    else:
                        effective = lower if actual < lower else upper
                equal(values.target_weight, wanted)
                equal(values.effective_weight, effective)
                equal(values.signal_shares, shares)
                equal(values.signal_close, close)
                assert bool(values.freeze_quantity) == freeze
                equal(values.frozen_quantity, shares if freeze else np.nan)
                if effective == 0:
                    target = 0.
                elif np.isnan(context.opening[d, a]):
                    target = np.nan
                elif freeze:
                    target = shares
                else:
                    target = budget * effective / context.opening[d, a]
                new_targets[asset] = target
            targets[sleeve], target_dates[sleeve] = new_targets, review.signal_date
            terminal_count += bool(review.terminal)
            ordinary_count += not review.terminal

        after_sells = start_holdings.copy()
        post_sell_cash = previous_cash.copy()
        actual_sells = {(order.sleeve, order.asset_id) for order in today if order.side == "sell"}
        required_sells = set()
        for sleeve, a in zip(*np.where(start_holdings > 0)):
            asset = context.assets[a]
            target = targets[sleeve][asset]
            assert np.isfinite(target) and target >= 0
            delta = start_holdings[sleeve, a] - target
            if delta > ATOL:
                required_sells.add((sleeve, asset))
        assert required_sells <= actual_sells
        for order in [order for order in today if order.side == "sell"]:
            sleeve, asset = int(order.sleeve), order.asset_id
            a = asset_indices[asset]
            current, target = start_holdings[sleeve, a], targets[sleeve][asset]
            delta = current - target
            assert delta >= -ATOL and order.requested_shares > 0
            equal(order.current_quantity, current)
            equal(order.target_quantity, target)
            equal(order.desired_trade_quantity, -delta)
            equal(order.requested_shares, delta)
            equal(order.requested_notional, delta * context.opening[d, a])
            assert order.plan_signal_date == target_dates[sleeve]
            allowed = context.prior_sell[d, a] == True
            assert order.reason == ("filled" if allowed else "lagged_sell_blocked")
            equal(order.filled_shares, delta if allowed else 0.)
            after_sells[sleeve, a] -= order.filled_shares
            post_sell_cash[sleeve] += order.filled_notional - order.fee - order.spread_cost - order.impact_cost

        buy_orders = [order for order in today if order.side == "buy"]
        if review is None:
            assert not buy_orders
            continue
        sleeve = int(review.sleeve)
        assert all(order.sleeve == sleeve for order in buy_orders)
        required_buys = set()
        for asset, values in plan.iterrows():
            a = asset_indices[asset]
            if values.effective_weight > 0 and (np.isnan(context.opening[d, a])
                    or targets[sleeve][asset] - after_sells[sleeve, a] > ATOL):
                required_buys.add(asset)
        assert required_buys <= {order.asset_id for order in buy_orders}
        assert not ({order.asset_id for order in buy_orders} &
                    {asset for target_sleeve, asset in actual_sells if target_sleeve == sleeve})
        for order in buy_orders:
            asset, a = order.asset_id, asset_indices[order.asset_id]
            assert asset in plan.index and plan.loc[asset, "effective_weight"] > 0
            missing = np.isnan(context.opening[d, a])
            target, current = targets[sleeve][asset], after_sells[sleeve, a]
            delta = np.nan if missing else target - current
            notional = budget * plan.loc[asset, "effective_weight"] if missing else delta * context.opening[d, a]
            equal(order.target_quantity, target)
            equal(order.current_quantity, current)
            equal(order.desired_trade_quantity, delta)
            equal(order.requested_shares, delta)
            equal(order.requested_notional, notional)
            equal(order.planned_budget, review.signal_nav * plan.loc[asset, "effective_weight"])
            equal(order.requested_budget, notional * (1 + context.prior_cost[d, a].sum() * row.cost_multiplier))
            assert order.plan_signal_date == review.signal_date
        demand = sum(order.requested_budget for order in buy_orders)
        available = post_sell_cash[sleeve]
        equal(review.post_sell_cash, available)
        equal(review.requested_buy_cost, demand)
        # The cash amount is independently reconciled above. Use its recorded
        # decimal for the ratio: a different summation order at cash ~= 0 can
        # amplify a sub-ulp monetary discrepancy into a material ratio change.
        scale = min(1., review.post_sell_cash / demand) if demand > 0 else 1.
        assert scale >= -1e-12
        scale = max(0., scale)
        equal(review.buy_scale, scale)
        for order in buy_orders:
            a = asset_indices[order.asset_id]
            equal(order.buy_scale, scale)
            reason = ("missing_execution_market" if np.isnan(context.opening[d, a]) else
                      "lagged_buy_blocked" if context.prior_buy[d, a] == False else
                      "zero_budget" if scale == 0 else "filled")
            assert order.reason == reason
            equal(order.filled_shares, order.requested_shares * scale if reason == "filled" else 0.)
            equal(order.filled_notional, order.requested_notional * scale if reason == "filled" else 0.)
    assert terminal_count == sleeves
    return {"ordinary_reviews": ordinary_count, "terminal_reviews": terminal_count,
            "close_policy_weights_and_bands_checked": True,
            "opening_targets_net_orders_and_cash_scale_checked": True,
            "blocked_budgets_not_redistributed": True,
            "new_plans_replace_previous_sell_intents": True,
            "terminal_schedule_and_retry_orders_checked": True}


def audit_run(row, folder: Path, context: MarketContext, periods=None) -> dict:
    """Audit one completed run; shared context is safe to reuse across runs."""
    daily = pd.read_csv(folder / "daily.csv", parse_dates=["date"]).set_index("date")
    orders = pd.read_csv(folder / "orders.csv.gz", parse_dates=[
        "signal_date", "execution_date", "plan_signal_date", "scheduled_exit_date"])
    positions = pd.read_csv(folder / "positions.csv.gz", parse_dates=["date"])
    # The flat-score SCORE contract produces header-only order/position files.
    # Give their numeric and boolean columns the same types as populated files.
    numeric_orders = ["sleeve", "planned_budget", "requested_budget", "requested_notional",
                      "requested_shares", "filled_shares", "fill_price", "filled_notional",
                      "fee", "spread_cost", "impact_cost", "writeoff_amount", "target_quantity",
                      "current_quantity", "desired_trade_quantity", "buy_scale"]
    orders[numeric_orders] = orders[numeric_orders].astype(float)
    orders["actual_flag_conflict"] = orders.actual_flag_conflict.astype(bool)
    positions[["sleeve", "shares", "close", "market_value"]] = positions[["sleeve", "shares", "close", "market_value"]].astype(float)
    plans = pd.read_csv(folder / "plans.csv.gz", parse_dates=["signal_date", "execution_date"])
    reviews = pd.read_csv(folder / "reviews.csv", parse_dates=["signal_date", "execution_date"])
    summary = json.loads((folder / "summary.json").read_text())
    pd.testing.assert_index_equal(daily.index, context.dates, check_names=False)
    sleeves = 1 if row.policy == "DAILY_NET" else 5
    assert summary["n_sleeves"] == sleeves and summary["policy"] == row.policy
    assert summary["constraint_mode"] == "lagged"
    assert summary["selection"] == ("equal_weight" if row.predictor == "EqualWeight" else "top_decile")
    inventory, cash, exposure = reconstruct_accounts(daily, orders, positions, context,
                                                     sleeves, row.cost_multiplier)
    saved_exposure = pd.read_csv(folder / "exposure_daily.csv", parse_dates=["date"]).set_index("date")
    pd.testing.assert_index_equal(saved_exposure.index, context.dates, check_names=False)
    equal(saved_exposure[exposure.columns], exposure)
    planning = audit_plans(row, plans, reviews, orders, inventory, cash, context)
    common = independent_metrics(daily, orders, exposure, "FULL")
    engine_fields = {key: value for key, value in common.items() if key in summary}
    check_fields(summary, engine_fields)
    check_fields(row._asdict(), engine_fields)
    extra = {"cumulative_gross_turnover": daily.gross_turnover.sum(),
             "gross_traded_notional": orders.filled_notional.sum(),
             "cash_roundoff_adjustment": daily.cash_roundoff_adjustment.sum(),
             "actual_flag_conflict_notional": orders.loc[orders.actual_flag_conflict, "filled_notional"].sum(),
             "end_open_positions": int(daily.open_position_count.iloc[-1]),
             "end_holdings_value": daily.holdings_value.iloc[-1], "end_cash": daily.cash.iloc[-1]}
    check_fields(summary, extra)
    check_fields(row._asdict(), extra)
    checked_periods = []
    if periods is not None:
        for period in ["FULL", "2023", "2024"]:
            check_fields(periods.loc[(row.run, period)], independent_metrics(daily, orders, exposure, period))
            checked_periods.append(period)
    reconstructed_nav = cash.sum(axis=1) + (inventory * np.nan_to_num(context.closing)[:, None, :]).sum(axis=(1, 2))
    return {"run": row.run, "status": "passed", "order_attempts": len(orders),
            "filled_orders": int(orders.filled_shares.gt(0).sum()),
            "writeoff_events": int(orders.side.eq("writeoff").sum()),
            "cash_inventory_positions_nav_checked": True,
            "opening_prices_lagged_costs_permissions_conflicts_checked": True,
            "period_metrics_checked": checked_periods, **planning,
            "max_cash_reconciliation_error": float(np.max(np.abs(daily.cash - cash.sum(axis=1)))),
            "max_nav_reconciliation_error": float(np.max(np.abs(daily.nav - reconstructed_nav)))}


def verify_comparison_references(out: Path, protocol: dict, periods: pd.DataFrame) -> None:
    comparison = pd.read_csv(out / "comparison.csv", dtype={"period": str})
    aliases = {"RANK": "NET", "TAPER": "NET", "SCORE": "NET", "BOTH": "WEIGHT"}
    assert {item["policy"]: item["actual_policy"] for item in protocol["aliases"]} == aliases
    expected = periods.reset_index().assign(source_kind="new", actual_run=lambda x: x.run, is_alias=False)
    frames = [expected]
    for alias, actual in aliases.items():
        subset = expected.loc[expected.predictor.eq("EqualWeight") & expected.policy.eq(actual)].copy()
        subset["policy"] = alias
        subset["run"] = [f"{alias}__EqualWeight__cost{int(cost)}" for cost in subset.cost_multiplier]
        subset["is_alias"], subset["source_kind"] = True, "alias"
        frames.append(subset)
    old = pd.read_csv(Path(protocol["source"]) / "period_metrics.csv", dtype={"period": str})
    old["run"] = [f"FULL__{r.predictor}__cost{int(r.cost_multiplier)}" for r in old.itertuples()]
    old["actual_run"], old["policy"], old["source_kind"], old["is_alias"] = old.run, "FULL", "archived_full", False
    frames.append(old)
    expected = pd.concat(frames, ignore_index=True).sort_values(["run", "period"]).reset_index(drop=True)
    assert not comparison.duplicated(["run", "period"]).any()
    comparison = comparison.sort_values(["run", "period"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(comparison[expected.columns], expected, check_dtype=False,
                                  check_exact=False, rtol=1e-9, atol=ATOL)


def _initialize_audit(context, periods, out):
    _AUDIT_WORKER.update(context=context, periods=periods, out=out)


def _audit_record(record):
    row = next(pd.DataFrame([record]).itertuples(index=False))
    return audit_run(row, _AUDIT_WORKER["out"] / "runs" / row.run,
                     _AUDIT_WORKER["context"], _AUDIT_WORKER["periods"])


def run(root: Path, out: Path) -> None:
    destination = out / "verification.json"
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite verification: {destination}")
    protocol = json.loads((out / "protocol.json").read_text())
    manifest = json.loads((out / "run_manifest.json").read_text())
    verify_hashes(out, protocol, manifest)
    predictions = {}
    for predictor in PREDICTORS:
        paths = [Path(path) for path in protocol["inputs_sha256"] if Path(path).name == f"{predictor}.csv.gz"]
        assert len(paths) == 1
        predictions[predictor] = pd.read_csv(paths[0], parse_dates=["date"])
    for filename in ["market_daily.csv.gz", "trading_constraints.csv.gz", "universe.csv.gz"]:
        digests = {digest for path, digest in protocol["inputs_sha256"].items() if Path(path).name == filename}
        assert digests == {sha256(root / "data" / filename)}
    market = pd.read_csv(root / "data" / "market_daily.csv.gz", parse_dates=["date"])
    constraints = pd.read_csv(root / "data" / "trading_constraints.csv.gz", parse_dates=["date"])
    universe = pd.read_csv(root / "data" / "universe.csv.gz", parse_dates=["date"])
    context = prepare_context(market, constraints, universe, predictions)
    summary = pd.read_csv(out / "summary.csv")
    planned = pd.DataFrame(protocol["runs"])
    assert len(planned) == 108 and not planned.run.duplicated().any()
    expected_specs = {(policy, predictor, cost) for policy in POLICIES
                      for predictor in PREDICTORS + ["EqualWeight"] for cost in [0, 1, 2]
                      if predictor != "EqualWeight" or policy in ["NET", "WEIGHT", "EMA50", "DAILY_NET"]}
    assert set(planned[["policy", "predictor", "cost_multiplier"]].itertuples(index=False, name=None)) == expected_specs
    assert planned.run.tolist() == [f"{r.policy}__{r.predictor}__cost{int(r.cost_multiplier)}" for r in planned.itertuples()]
    pd.testing.assert_frame_equal(
        summary[planned.columns].sort_values("run").reset_index(drop=True),
        planned.sort_values("run").reset_index(drop=True), check_dtype=False)
    assert set(protocol["policies"]) == set(POLICIES)
    assert set(protocol["cost_multipliers"]) == {0, 1, 2}
    periods = pd.read_csv(out / "period_metrics.csv", dtype={"period": str}).set_index(["run", "period"], verify_integrity=True)
    assert set(periods.index) == {(name, period) for name in planned.run for period in ["FULL", "2023", "2024"]}
    records = []
    with ProcessPoolExecutor(max_workers=2, initializer=_initialize_audit,
                             initargs=(context, periods, out)) as pool:
        futures = [pool.submit(_audit_record, record) for record in summary.to_dict("records")]
        for future in as_completed(futures):
            result = future.result()
            records.append(result)
            print(f"Independent rebalance audit {len(records)}/{len(futures)}: {result['run']}", flush=True)
    records.sort(key=lambda row: row["run"])
    verify_comparison_references(out, protocol, periods)
    verify_hashes(out, protocol, manifest)
    result = {"status": "passed", "verified_at_utc": datetime.now(timezone.utc).isoformat(),
              "input_code_output_hashes_checked": True, "audited_runs": records,
              "archived_full_and_equal_weight_aliases_checked": True,
              "paired_intervals_scope": "Paired bootstrap intervals are outside this accounting/execution audit",
              "run_manifest_sha256": sha256(out / "run_manifest.json"),
              "audit_code_sha256": sha256(Path(__file__))}
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"Verified {len(records)} runs; saved {destination}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out", type=Path, default=PROJECT / "outputs" / "rebalance_study")
    args = parser.parse_args()
    run(args.root, args.out)
