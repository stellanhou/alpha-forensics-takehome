"""Close-planned, next-open net rebalancing of independent long-only sleeves.

Prices determine share quantities only after names, weights and a close NAV
budget have been frozen.  Permission and cost inputs are always lagged one
business day.  Failed buys expire; excess positions are sold again until a
new plan replaces the old quantity target.  Cash roundoff in [-1e-12, 0) is
explicitly reported, and there is no implicit minimum trade size.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay

from portfolio_ledger import ORDER_COLUMNS as OLD_ORDER_COLUMNS, _validate_inputs


POLICIES = ("NET", "RANK", "WEIGHT", "BOTH", "EMA50", "TAPER", "SCORE", "DAILY_NET")
ORDER_COLUMNS = OLD_ORDER_COLUMNS + [
    "plan_signal_date", "target_quantity", "current_quantity",
    "desired_trade_quantity", "buy_scale",
]
POSITION_COLUMNS = ["date", "sleeve", "asset_id", "shares", "close", "market_value"]
PLAN_COLUMNS = [
    "signal_date", "execution_date", "sleeve", "asset_id", "target_weight",
    "effective_weight", "freeze_quantity", "frozen_quantity", "signal_shares",
    "signal_close", "signal_sleeve_nav", "terminal",
]
REVIEW_COLUMNS = [
    "signal_date", "execution_date", "sleeve", "signal_nav", "opening_nav",
    "execution_budget", "buy_scale", "requested_buy_cost", "post_sell_cash",
    "terminal", "desired_asset_count",
]


def target_weights(
    scores: pd.Series,
    held: dict[str, float],
    policy: str,
    previous_weights: dict[str, float],
    selection: str = "top_decile",
) -> dict[str, float]:
    """Construct positive ideal weights using only the current member scores.

``held`` contains actual share quantities, not previous intended positions.
WEIGHT's relative bands are applied by the ledger after ideal weights exist.
Equal-weight members support only NET, WEIGHT, EMA50 and DAILY_NET; equivalent
benchmarks for the other policies must be deduplicated by the study caller.
    """
    if policy not in POLICIES:
        raise ValueError(f"unknown rebalance policy: {policy}")
    if selection not in {"top_decile", "equal_weight"}:
        raise ValueError("selection must be top_decile or equal_weight")
    if selection == "equal_weight" and policy not in {"NET", "WEIGHT", "EMA50", "DAILY_NET"}:
        raise ValueError("equal_weight has no distinct version of this policy")
    if (scores.empty or scores.index.has_duplicates or scores.index.isna().any()
            or not np.isfinite(scores.to_numpy(dtype=float)).all()):
        raise ValueError("scores must contain unique members and finite values")
    if any(not np.isfinite(v) or v < 0 for v in held.values()):
        raise ValueError("held quantities must be finite and nonnegative")
    if any(not np.isfinite(v) or v < 0 for v in previous_weights.values()):
        raise ValueError("previous weights must be finite and nonnegative")

    entry = scores if selection == "equal_weight" else scores[scores >= scores.quantile(.9)]
    if policy in {"RANK", "BOTH"}:
        retention_cutoff = scores.quantile(.8)
        retained = {
            asset for asset, shares in held.items()
            if shares > 0 and asset in scores.index and scores[asset] >= retention_cutoff
        }
        target_count = int(np.ceil(.1 * len(scores)))
        # The final tied score group is included in full; IDs do not choose winners.
        for value in sorted(entry.unique(), reverse=True):
            if len(retained) >= target_count:
                break
            retained.update(entry.index[entry.eq(value)])
        selected = scores.loc[sorted(retained)]
    else:
        selected = entry

    weights = {asset: 1. / len(selected) for asset in selected.index}
    if policy == "TAPER":
        raw = len(selected) + 1 - selected.rank(method="average", ascending=False)
        weights = (raw / raw.sum()).to_dict()
    elif policy == "SCORE":
        if scores.nunique() == 1:
            return {}
        raw = (selected - scores.mean()).clip(lower=0)
        active = raw[raw > 0].to_dict()
        cap = 2. / len(selected)
        remaining = min(1., cap * len(active))
        weights = {}
        while active:
            total = sum(active.values())
            proposed = {asset: remaining * value / total for asset, value in active.items()}
            capped = [asset for asset, value in proposed.items() if value > cap]
            if not capped:
                weights.update(proposed)
                break
            for asset in capped:
                weights[asset] = cap
                remaining -= cap
                del active[asset]
    elif policy == "EMA50":
        weights = {
            asset: .5 * weights.get(asset, 0.) + .5 * previous_weights.get(asset, 0.)
            for asset in set(weights) | (set(previous_weights) & set(scores.index))
        }
    if any(not np.isfinite(v) or v < 0 for v in weights.values()):
        raise AssertionError("constructed weights must be finite and nonnegative")
    return {asset: float(weights[asset]) for asset in sorted(weights) if weights[asset] > 0}


@dataclass
class Holding:
    shares: float
    last_close: float


@dataclass
class Plan:
    signal_date: pd.Timestamp
    sleeve: int
    signal_nav: float
    terminal: bool
    rows: list[dict]


def simulate_rebalance(
    predictions: pd.DataFrame,
    market: pd.DataFrame,
    constraints: pd.DataFrame,
    *,
    start_signal,
    end_date,
    policy: str,
    selection: str = "top_decile",
) -> dict:
    """Run the fixed policy and return daily, orders, positions, plans, reviews.

The caller applies the cost scenario to the supplied constraints beforehand.
All policies use five initial 0.2 sleeves, except DAILY_NET's single 1.0 sleeve.
New-entry signals stop six business days before the final valuation date.
Five-sleeve policies then give each sleeve a zero-target close plan on its next
review.  DAILY_NET holds its final ordinary target until a zero-target plan at
the penultimate close.  Terminal liquidation remains permission constrained.
    """
    if policy not in POLICIES:
        raise ValueError(f"unknown rebalance policy: {policy}")
    if selection not in {"top_decile", "equal_weight"}:
        raise ValueError("selection must be top_decile or equal_weight")
    if selection == "equal_weight" and policy not in {"NET", "WEIGHT", "EMA50", "DAILY_NET"}:
        raise ValueError("equal_weight has no distinct version of this policy")
    start, end = pd.Timestamp(start_signal), pd.Timestamp(end_date)
    if (start.tzinfo is not None or end.tzinfo is not None
            or start != start.normalize() or end != end.normalize()
            or start.dayofweek >= 5 or end.dayofweek >= 5 or end < start):
        raise ValueError("start_signal and end_date must be ordered business dates")
    predictions, market, constraints, dates = _validate_inputs(
        predictions, market, constraints, start, end,
    )
    market_by_date = {
        date: dict(zip(rows.asset_id, rows[["open", "close"]].itertuples(index=False, name=None)))
        for date, rows in market.groupby("date", sort=False)
    }
    constraint_by_date = {
        date: dict(zip(rows.asset_id, rows.iloc[:, 2:].itertuples(index=False, name=None)))
        for date, rows in constraints.groupby("date", sort=False)
    }
    scores_by_date = {
        date: rows.set_index("asset_id").prediction
        for date, rows in predictions.groupby("date", sort=False)
    }
    n_sleeves = 1 if policy == "DAILY_NET" else 5
    last_signal = end - BDay(6)
    cash = np.full(n_sleeves, 1. / n_sleeves)
    holdings: list[dict[str, Holding]] = [{} for _ in range(n_sleeves)]
    active_targets: list[dict[str, float]] = [{} for _ in range(n_sleeves)]
    active_dates = [pd.NaT for _ in range(n_sleeves)]
    previous_weights: list[dict[str, float]] = [{} for _ in range(n_sleeves)]
    orders, positions, daily, plans, reviews = [], [], [], [], []
    pending: Plan | None = None
    previous_nav = 1.

    def record(**values):
        row = dict(
            planned_budget=np.nan, requested_budget=np.nan, requested_notional=0.,
            requested_shares=0., filled_shares=0., fill_price=np.nan,
            filled_notional=0., fee=0., spread_cost=0., impact_cost=0.,
            writeoff_amount=0., actual_flag_conflict=False,
            scheduled_exit_date=pd.NaT, target_quantity=np.nan,
            current_quantity=0., desired_trade_quantity=0., buy_scale=np.nan,
        )
        row.update(values)
        orders.append(row)

    def known(day, asset):
        return constraint_by_date[day - BDay(1)][asset]

    def rates(permission):
        return permission[2] / 10000., permission[3] / 20000., permission[4] / 10000.

    for index, day in enumerate(dates):
        quotes = market_by_date[day]
        actual_flags = constraint_by_date[day]
        day_order_start = len(orders)

        # Permanent disappearance is discovered today and recovered at zero.
        for sleeve, lots in enumerate(holdings):
            for asset, lot in list(lots.items()):
                if asset not in quotes:
                    record(
                        signal_date=day - BDay(1), execution_date=day,
                        plan_signal_date=active_dates[sleeve], sleeve=sleeve,
                        asset_id=asset, side="writeoff",
                        current_quantity=lot.shares, desired_trade_quantity=-lot.shares,
                        requested_notional=lot.shares * lot.last_close,
                        requested_shares=lot.shares,
                        writeoff_amount=lot.shares * lot.last_close,
                        reason="missing_market_writeoff",
                    )
                    del lots[asset]

        execution_budget = opening_nav = None
        pending_rows = {}
        if pending is not None:
            if pending.signal_date + BDay(1) != day:
                raise AssertionError("plan must execute at the next business open")
            sleeve = pending.sleeve
            opening_nav = cash[sleeve] + sum(
                lot.shares * quotes[asset][0] for asset, lot in holdings[sleeve].items()
            )
            execution_budget = min(pending.signal_nav, opening_nav)
            pending_rows = {row["asset_id"]: row for row in pending.rows}
            new_targets = {}
            for asset, row in pending_rows.items():
                if row["effective_weight"] == 0:
                    new_targets[asset] = 0.
                elif asset not in quotes:
                    # No price-derived quantity exists for the reserved slot.
                    new_targets[asset] = np.nan
                elif row["freeze_quantity"]:
                    new_targets[asset] = row["frozen_quantity"]
                else:
                    new_targets[asset] = execution_budget * row["effective_weight"] / quotes[asset][0]
            # Replace old sell intent before any execution of this sleeve.
            active_targets[sleeve] = new_targets
            active_dates[sleeve] = pending.signal_date

        for sleeve, lots in enumerate(holdings):
            for asset, lot in list(lots.items()):
                quantity = active_targets[sleeve][asset]
                if not np.isfinite(quantity) or quantity < 0:
                    raise AssertionError("quoted holdings require a finite nonnegative target")
                if lot.shares <= quantity:
                    continue
                requested = lot.shares - quantity
                permission = known(day, asset)
                open_price = quotes[asset][0]
                notional = requested * open_price
                details = dict(
                    signal_date=day - BDay(1), execution_date=day,
                    plan_signal_date=active_dates[sleeve], sleeve=sleeve,
                    asset_id=asset, side="sell", target_quantity=quantity,
                    current_quantity=lot.shares, desired_trade_quantity=-requested,
                    requested_notional=notional, requested_shares=requested,
                )
                if not permission[1]:
                    record(**details, reason="lagged_sell_blocked")
                    continue
                fee, spread, impact = (notional * rate for rate in rates(permission))
                cash[sleeve] += notional - fee - spread - impact
                record(
                    **details, filled_shares=requested, fill_price=open_price,
                    filled_notional=notional, fee=fee, spread_cost=spread,
                    impact_cost=impact, reason="filled",
                    actual_flag_conflict=not bool(actual_flags[asset][1]),
                )
                if quantity == 0:
                    del lots[asset]
                else:
                    # Set the exact target to avoid a roundoff-only reverse buy.
                    lot.shares = quantity

        if pending is not None:
            sleeve = pending.sleeve
            demands = []
            for asset, row in pending_rows.items():
                if row["effective_weight"] == 0:
                    continue
                permission = known(day, asset)
                cost_rates = rates(permission)
                current = holdings[sleeve][asset].shares if asset in holdings[sleeve] else 0.
                target = active_targets[sleeve][asset]
                if asset not in quotes:
                    shares = np.nan
                    notional = execution_budget * row["effective_weight"]
                else:
                    if target <= current:
                        continue
                    shares = target - current
                    notional = shares * quotes[asset][0]
                demand = notional * (1 + sum(cost_rates))
                demands.append((asset, row, permission, cost_rates, current, target, shares, notional, demand))
            requested_cost = sum(demand[-1] for demand in demands)
            post_sell_cash = float(cash[sleeve])
            buy_scale = min(1., post_sell_cash / requested_cost) if requested_cost > 0 else 1.
            if buy_scale < -1e-12:
                raise AssertionError("cash cannot be borrowed to buy")
            # An accumulated cash subtraction may be a few ulps below zero.
            buy_scale = max(0., buy_scale)
            for asset, row, permission, cost_rates, current, target, shares, notional, demand in demands:
                details = dict(
                    signal_date=day - BDay(1), execution_date=day,
                    plan_signal_date=pending.signal_date, sleeve=sleeve,
                    asset_id=asset, side="buy", target_quantity=target,
                    current_quantity=current, desired_trade_quantity=shares,
                    planned_budget=pending.signal_nav * row["effective_weight"],
                    requested_budget=demand, requested_notional=notional,
                    requested_shares=shares, buy_scale=buy_scale,
                )
                if asset not in quotes:
                    record(**details, reason="missing_execution_market")
                    continue
                if not permission[0]:
                    record(**details, reason="lagged_buy_blocked")
                    continue
                if buy_scale == 0:
                    record(**details, reason="zero_budget")
                    continue
                filled_shares, filled_notional = shares * buy_scale, notional * buy_scale
                fee, spread, impact = (filled_notional * rate for rate in cost_rates)
                cash[sleeve] -= filled_notional + fee + spread + impact
                updated_shares = target if buy_scale == 1. else current + filled_shares
                holdings[sleeve][asset] = Holding(updated_shares, quotes[asset][1])
                record(
                    **details, filled_shares=filled_shares, fill_price=quotes[asset][0],
                    filled_notional=filled_notional, fee=fee, spread_cost=spread,
                    impact_cost=impact, reason="filled",
                    actual_flag_conflict=not bool(actual_flags[asset][0]),
                )
            reviews.append(dict(
                signal_date=pending.signal_date, execution_date=day, sleeve=sleeve,
                signal_nav=pending.signal_nav, opening_nav=opening_nav,
                execution_budget=execution_budget, buy_scale=buy_scale,
                requested_buy_cost=requested_cost, post_sell_cash=post_sell_cash,
                terminal=pending.terminal,
                desired_asset_count=sum(row["target_weight"] > 0 for row in pending.rows),
            ))
            pending = None

        if cash.min() < -1e-12:
            raise AssertionError("cash cannot be borrowed")
        roundoff = float(-cash[cash < 0].sum())
        cash[cash < 0] = 0.
        market_values = np.zeros(n_sleeves)
        for sleeve, lots in enumerate(holdings):
            for asset, lot in lots.items():
                close_price = quotes[asset][1]
                value = lot.shares * close_price
                market_values[sleeve] += value
                lot.last_close = close_price
                positions.append(dict(
                    date=day, sleeve=sleeve, asset_id=asset, shares=lot.shares,
                    close=close_price, market_value=value,
                ))
        sleeve_nav = cash + market_values
        nav = float(sleeve_nav.sum())
        if not np.isfinite(nav) or nav < 0:
            raise AssertionError("NAV must be finite and nonnegative")
        day_orders = orders[day_order_start:]
        totals = {
            field: float(sum(order[field] for order in day_orders))
            for field in ["filled_notional", "fee", "spread_cost", "impact_cost", "writeoff_amount"]
        }
        conflict_orders = [order for order in day_orders if order["actual_flag_conflict"]]
        daily_return = (nav / previous_nav - 1) if previous_nav > 0 else 0.
        if previous_nav == 0 and (nav != 0 or totals["filled_notional"] != 0):
            raise AssertionError("zero-NAV portfolio cannot regain capital")
        row = dict(
            date=day, nav=nav, daily_return=np.nan if index == 0 else daily_return,
            cash=float(cash.sum()), holdings_value=float(market_values.sum()),
            gross_traded_notional=totals["filled_notional"],
            gross_turnover=totals["filled_notional"] / previous_nav if previous_nav > 0 else 0.,
            fee=totals["fee"], spread_cost=totals["spread_cost"],
            impact_cost=totals["impact_cost"], writeoff=totals["writeoff_amount"],
            cash_roundoff_adjustment=roundoff,
            actual_flag_conflict_count=len(conflict_orders),
            actual_flag_conflict_notional=float(sum(order["filled_notional"] for order in conflict_orders)),
            open_position_count=sum(len(lots) for lots in holdings),
        )
        for sleeve in range(n_sleeves):
            row[f"sleeve_{sleeve}_cash"] = float(cash[sleeve])
            row[f"sleeve_{sleeve}_market_value"] = float(market_values[sleeve])
            row[f"sleeve_{sleeve}_nav"] = float(sleeve_nav[sleeve])
        daily.append(row)
        previous_nav = nav

        if day >= end:
            continue
        if policy == "DAILY_NET" and last_signal < day < end - BDay(1):
            continue
        sleeve = index % n_sleeves
        terminal = day > last_signal
        ideal = {} if terminal else target_weights(
            scores_by_date[day],
            {asset: lot.shares for asset, lot in holdings[sleeve].items()},
            policy, previous_weights[sleeve], selection,
        )
        previous_weights[sleeve] = ideal
        plan_rows = []
        for asset in sorted(set(ideal) | set(holdings[sleeve])):
            shares = holdings[sleeve][asset].shares if asset in holdings[sleeve] else 0.
            weight = ideal.get(asset, 0.)
            effective, freeze = weight, False
            if policy in {"WEIGHT", "BOTH"} and weight > 0 and shares > 0:
                actual_weight = shares * quotes[asset][1] / sleeve_nav[sleeve]
                lower, upper = .9 * weight, 1.1 * weight
                if lower <= actual_weight <= upper:
                    effective, freeze = actual_weight, True
                else:
                    effective = lower if actual_weight < lower else upper
            plan_row = dict(
                signal_date=day, execution_date=day + BDay(1), sleeve=sleeve,
                asset_id=asset, target_weight=weight, effective_weight=effective,
                freeze_quantity=freeze, frozen_quantity=shares if freeze else np.nan,
                signal_shares=shares, signal_close=quotes[asset][1],
                signal_sleeve_nav=float(sleeve_nav[sleeve]), terminal=terminal,
            )
            plan_rows.append(plan_row)
        plans.extend(plan_rows)
        pending = Plan(day, sleeve, float(sleeve_nav[sleeve]), terminal, plan_rows)

    daily_frame = pd.DataFrame(daily)
    returns = daily_frame.daily_return.dropna()
    volatility = float(returns.std(ddof=1)) if len(returns) >= 2 else None
    summary = dict(
        policy=policy, selection=selection, constraint_mode="lagged", n_sleeves=n_sleeves,
        start_signal=str(start.date()), end_date=str(end.date()),
        last_allowed_signal=str(last_signal.date()), initial_nav=1., final_nav=previous_nav,
        n_return_days=len(returns), total_return=previous_nav - 1,
        annualized_return=previous_nav ** (252 / len(returns)) - 1 if len(returns) else None,
        annualized_volatility=volatility * np.sqrt(252) if volatility is not None else None,
        sharpe=float(returns.mean() / volatility * np.sqrt(252)) if volatility else None,
        max_drawdown=float((daily_frame.nav / daily_frame.nav.cummax() - 1).min()),
        cumulative_gross_turnover=float(daily_frame.gross_turnover.sum()),
        mean_daily_turnover=float(daily_frame.gross_turnover.iloc[1:].mean()) if len(returns) else None,
        gross_traded_notional=float(daily_frame.gross_traded_notional.sum()),
        total_fee=float(daily_frame.fee.sum()), total_spread_cost=float(daily_frame.spread_cost.sum()),
        total_impact_cost=float(daily_frame.impact_cost.sum()),
        total_writeoff=float(daily_frame.writeoff.sum()),
        cash_roundoff_adjustment=float(daily_frame.cash_roundoff_adjustment.sum()),
        actual_flag_conflict_orders=int(daily_frame.actual_flag_conflict_count.sum()),
        actual_flag_conflict_notional=float(daily_frame.actual_flag_conflict_notional.sum()),
        end_open_positions=int(daily_frame.open_position_count.iloc[-1]),
        end_holdings_value=float(daily_frame.holdings_value.iloc[-1]),
        end_cash=float(daily_frame.cash.iloc[-1]),
    )
    return dict(
        daily=daily_frame, orders=pd.DataFrame(orders, columns=ORDER_COLUMNS),
        positions=pd.DataFrame(positions, columns=POSITION_COLUMNS),
        plans=pd.DataFrame(plans, columns=PLAN_COLUMNS),
        reviews=pd.DataFrame(reviews, columns=REVIEW_COLUMNS), summary=summary,
    )
