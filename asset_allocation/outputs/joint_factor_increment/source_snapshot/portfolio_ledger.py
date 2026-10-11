"""Five staggered, self-financing long-only sleeves marked to daily closes.

Signals and inclusive buy budgets are fixed at the previous close.  Execution
uses the next open and (optionally) previous-day permission/cost proxies.  The
same-day final permission flags are audit observations, never execution inputs.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay


KEYS = ["date", "asset_id"]
ORDER_COLUMNS = [
    "signal_date", "execution_date", "sleeve", "asset_id", "side",
    "planned_budget", "requested_budget", "requested_notional", "requested_shares",
    "filled_shares", "fill_price", "filled_notional", "fee", "spread_cost",
    "impact_cost", "writeoff_amount", "reason", "actual_flag_conflict",
    "scheduled_exit_date",
]
POSITION_COLUMNS = [
    "date", "sleeve", "asset_id", "shares", "close", "market_value",
    "entry_signal_date", "entry_date", "scheduled_exit_date",
]


@dataclass
class Lot:
    shares: float
    entry_signal_date: pd.Timestamp
    entry_date: pd.Timestamp
    scheduled_exit_date: pd.Timestamp
    last_close: float


def _frame(frame: pd.DataFrame, columns: list[str], name: str) -> pd.DataFrame:
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"{name}: missing columns {sorted(missing)}")
    frame = frame[columns].copy()
    frame["date"] = pd.to_datetime(frame.date)
    if frame.date.dt.tz is not None or not frame.date.eq(frame.date.dt.normalize()).all():
        raise ValueError(f"{name}: expected timezone-naive dates at midnight")
    if frame[KEYS].isna().any().any() or frame.duplicated(KEYS).any():
        raise ValueError(f"{name}: null or duplicate date/asset keys")
    if frame.date.dt.dayofweek.ge(5).any():
        raise ValueError(f"{name}: dates must use the synthetic business-day calendar")
    return frame.sort_values(KEYS).reset_index(drop=True)


def _validate_inputs(predictions, market, constraints, start, end):
    predictions = _frame(predictions, KEYS + ["prediction"], "predictions")
    market = _frame(market, KEYS + ["open", "close"], "market")
    constraints = _frame(
        constraints,
        KEYS + ["can_buy", "can_sell", "fee_bps_one_way", "spread_bps", "impact_bps_proxy"],
        "constraints",
    )
    if not np.isfinite(predictions.prediction.to_numpy(dtype=float)).all():
        raise ValueError("predictions: nonfinite predictions")
    prices = market[["open", "close"]].to_numpy(dtype=float)
    if not (np.isfinite(prices).all() and (prices > 0).all()):
        raise ValueError("market: prices must be finite and strictly positive")
    flags = constraints[["can_buy", "can_sell"]]
    if flags.isna().any().any() or not flags.isin([True, False]).all().all():
        raise ValueError("constraints: expected boolean permission flags")
    costs = constraints[["fee_bps_one_way", "spread_bps", "impact_bps_proxy"]].to_numpy(dtype=float)
    if not (np.isfinite(costs).all() and (costs >= 0).all()):
        raise ValueError("constraints: cost proxies must be finite and nonnegative")
    if ((costs[:, 0] + costs[:, 1] / 2 + costs[:, 2]) >= 10000).any():
        raise ValueError("constraints: one-way cost rate must be below one")
    market_keys = pd.MultiIndex.from_frame(market[KEYS])
    if not market_keys.equals(pd.MultiIndex.from_frame(constraints[KEYS])):
        raise ValueError("constraints: keys differ from market")
    if not pd.MultiIndex.from_frame(predictions[KEYS]).isin(market_keys).all():
        raise ValueError("predictions: signal-date market data missing")
    dates = pd.bdate_range(start, end)
    if not dates.isin(market.date.unique()).all():
        raise ValueError("market: missing complete market date")
    # An absent row may denote an asset's permanent exit, but not an internal
    # data gap followed by reappearance.  This check does not inform selection.
    for asset, rows in market.groupby("asset_id", sort=False):
        if len(rows) != len(pd.bdate_range(rows.date.iloc[0], rows.date.iloc[-1])):
            raise ValueError(f"market: internal date gap for {asset}")
    signal_dates = dates[dates <= end - BDay(6)]
    if not signal_dates.isin(predictions.date.unique()).all():
        raise ValueError("predictions: missing scheduled signal date")
    return predictions, market, constraints, dates


def simulate(
    predictions: pd.DataFrame,
    market: pd.DataFrame,
    constraints: pd.DataFrame,
    *,
    start_signal,
    end_date,
    selection: str = "top_decile",
    cost_mode: str = "base",
    constraint_mode: str = "lagged",
) -> dict:
    """Return daily NAV, all order attempts, close positions, and NAV metrics.

    ``top_decile`` uses prediction >= each date's 0.9 quantile (ties retained).
    ``equal_weight`` uses all predicted members on that signal date.  Missing
    forward labels never remove a signal.  Five independent sleeves begin with
    0.2 cash each; returns are reinvested within their originating sleeve.

    A sleeve is signalled every fifth business day.  At the following open,
    first liquidate lots whose scheduled five-day holding period has elapsed;
    an unsold lot blocks only a new buy of the same asset and retains its
    original expiry.  Other names use fixed equal allocations of available
    cash, without redistributing blocked names' budgets.  Failed new buy
    orders expire immediately.  Overdue sells are retried each subsequent day.
    Entry signals stop at end_date minus six business days, a fixed portfolio
    end convention.  Existing positions are never forcibly sold at the end.
    """
    if selection not in {"top_decile", "equal_weight"}:
        raise ValueError("selection must be top_decile or equal_weight")
    if cost_mode not in {"zero", "base"}:
        raise ValueError("cost_mode must be zero or base")
    if constraint_mode not in {"none", "lagged"}:
        raise ValueError("constraint_mode must be none or lagged")
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
    signals = {}
    for date, rows in predictions.groupby("date", sort=False):
        if start <= date <= end - BDay(6):
            selected = rows if selection == "equal_weight" else rows.loc[
                rows.prediction >= rows.prediction.quantile(.9)
            ]
            signals[date] = tuple(selected.asset_id)

    cash = np.full(5, .2, dtype=float)
    holdings: list[dict] = [{} for _ in range(5)]
    orders, positions, daily = [], [], []
    pending = None
    previous_nav = 1.

    def record(**values):
        row = dict(
            planned_budget=np.nan, requested_budget=np.nan, requested_notional=0.,
            requested_shares=0., filled_shares=0., fill_price=np.nan,
            filled_notional=0., fee=0., spread_cost=0., impact_cost=0.,
            writeoff_amount=0., actual_flag_conflict=False,
        )
        row.update(values)
        orders.append(row)

    def known(day, asset):
        # Exact previous-business-day lookup; a gap is a broken input invariant.
        return constraint_by_date[day - BDay(1)][asset]

    def rates(row):
        if cost_mode == "zero":
            return (0., 0., 0.)
        return row[2] / 10000., row[3] / 20000., row[4] / 10000.

    for index, day in enumerate(dates):
        quotes = market_by_date[day]
        actual_flags = constraint_by_date[day]
        day_order_start = len(orders)

        # A permanent exit is discovered on the first missing date, with zero
        # recovery and no fictitious fill.  Last close value is the loss audit.
        for sleeve, lots in enumerate(holdings):
            for asset, lot in list(lots.items()):
                if asset not in quotes:
                    record(
                        signal_date=day - BDay(1), execution_date=day,
                        sleeve=sleeve, asset_id=asset, side="writeoff",
                        requested_notional=lot.shares * lot.last_close,
                        requested_shares=lot.shares,
                        writeoff_amount=lot.shares * lot.last_close,
                        reason="missing_market_writeoff",
                        scheduled_exit_date=lot.scheduled_exit_date,
                    )
                    del lots[asset]

        # All expired lots, including previously blocked exits, are attempted.
        for sleeve, lots in enumerate(holdings):
            for asset, lot in list(lots.items()):
                if lot.scheduled_exit_date > day:
                    continue
                permission = known(day, asset)
                open_price = quotes[asset][0]
                notional = lot.shares * open_price
                details = dict(
                    signal_date=day - BDay(1), execution_date=day,
                    sleeve=sleeve, asset_id=asset, side="sell",
                    requested_notional=notional, requested_shares=lot.shares,
                    scheduled_exit_date=lot.scheduled_exit_date,
                )
                if constraint_mode == "lagged" and not permission[1]:
                    record(**details, reason="lagged_sell_blocked")
                    continue
                fee, spread, impact = (notional * rate for rate in rates(permission))
                cash[sleeve] += notional - fee - spread - impact
                record(
                    **details, filled_shares=lot.shares, fill_price=open_price,
                    filled_notional=notional, fee=fee, spread_cost=spread,
                    impact_cost=impact, reason="filled",
                    actual_flag_conflict=not bool(actual_flags[asset][1]),
                )
                del lots[asset]

        if pending is not None:
            signal_day, sleeve, names, planned_budget = pending
            if signal_day + BDay(1) != day:
                raise AssertionError("buy plan must execute at the next business open")
            # This fixed per-name denominator includes blocked/unquoted names.
            total_budget = min(planned_budget, cash[sleeve])
            allocation = total_budget / len(names)
            for asset in names:
                permission = known(day, asset)
                fee_rate, spread_rate, impact_rate = rates(permission)
                notional = allocation / (1 + fee_rate + spread_rate + impact_rate)
                details = dict(
                    signal_date=signal_day, execution_date=day,
                    sleeve=sleeve, asset_id=asset, side="buy",
                    planned_budget=planned_budget / len(names), requested_budget=allocation,
                    requested_notional=notional, scheduled_exit_date=day + BDay(5),
                )
                if asset in holdings[sleeve]:
                    record(**details, reason="existing_unsold_position")
                    continue
                if asset not in quotes:
                    record(**details, reason="missing_execution_market")
                    continue
                if constraint_mode == "lagged" and not permission[0]:
                    record(**details, reason="lagged_buy_blocked")
                    continue
                if allocation == 0:
                    record(**details, reason="zero_budget")
                    continue
                open_price, close_price = quotes[asset]
                shares = notional / open_price
                fee, spread, impact = (
                    notional * fee_rate, notional * spread_rate, notional * impact_rate,
                )
                cash[sleeve] -= notional + fee + spread + impact
                holdings[sleeve][asset] = Lot(
                    shares, signal_day, day, day + BDay(5), close_price,
                )
                record(
                    **details, requested_shares=shares, filled_shares=shares,
                    fill_price=open_price, filled_notional=notional, fee=fee,
                    spread_cost=spread, impact_cost=impact, reason="filled",
                    actual_flag_conflict=not bool(actual_flags[asset][0]),
                )
            pending = None

        if cash.min() < -1e-12:
            raise AssertionError("cash cannot be borrowed")
        # Roundoff from summing fractionally allocated, fee-inclusive orders.
        cash[cash < 0] = 0.
        market_values = np.zeros(5)
        for sleeve, lots in enumerate(holdings):
            for asset, lot in lots.items():
                close_price = quotes[asset][1]
                value = lot.shares * close_price
                market_values[sleeve] += value
                lot.last_close = close_price
                positions.append(dict(
                    date=day, sleeve=sleeve, asset_id=asset, shares=lot.shares,
                    close=close_price, market_value=value,
                    entry_signal_date=lot.entry_signal_date, entry_date=lot.entry_date,
                    scheduled_exit_date=lot.scheduled_exit_date,
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
        # A zero-NAV portfolio is absorbing: no debt or cash injection is allowed.
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
            actual_flag_conflict_count=len(conflict_orders),
            actual_flag_conflict_notional=float(sum(order["filled_notional"] for order in conflict_orders)),
            open_position_count=sum(len(lots) for lots in holdings),
        )
        for sleeve in range(5):
            row[f"sleeve_{sleeve}_cash"] = float(cash[sleeve])
            row[f"sleeve_{sleeve}_market_value"] = float(market_values[sleeve])
            row[f"sleeve_{sleeve}_nav"] = float(sleeve_nav[sleeve])
        daily.append(row)
        previous_nav = nav
        if day in signals:
            sleeve = index % 5
            pending = (day, sleeve, signals[day], float(sleeve_nav[sleeve]))

    daily_frame = pd.DataFrame(daily)
    order_frame = pd.DataFrame(orders, columns=ORDER_COLUMNS)
    position_frame = pd.DataFrame(positions, columns=POSITION_COLUMNS)
    returns = daily_frame.daily_return.dropna()
    volatility = float(returns.std(ddof=1)) if len(returns) >= 2 else None
    summary = dict(
        selection=selection, cost_mode=cost_mode, constraint_mode=constraint_mode,
        start_signal=str(start.date()), end_date=str(end.date()),
        last_allowed_signal=str((end - BDay(6)).date()),
        initial_nav=1., final_nav=previous_nav, n_return_days=len(returns),
        total_return=previous_nav - 1,
        annualized_return=previous_nav ** (252 / len(returns)) - 1 if len(returns) else None,
        annualized_volatility=volatility * np.sqrt(252) if volatility is not None else None,
        sharpe=float(returns.mean() / volatility * np.sqrt(252)) if volatility else None,
        max_drawdown=float((daily_frame.nav / daily_frame.nav.cummax() - 1).min()),
        cumulative_gross_turnover=float(daily_frame.gross_turnover.sum()),
        mean_daily_turnover=float(daily_frame.gross_turnover.iloc[1:].mean()) if len(returns) else None,
        gross_traded_notional=float(daily_frame.gross_traded_notional.sum()),
        total_fee=float(daily_frame.fee.sum()),
        total_spread_cost=float(daily_frame.spread_cost.sum()),
        total_impact_cost=float(daily_frame.impact_cost.sum()),
        total_writeoff=float(daily_frame.writeoff.sum()),
        actual_flag_conflict_orders=int(daily_frame.actual_flag_conflict_count.sum()),
        actual_flag_conflict_notional=float(daily_frame.actual_flag_conflict_notional.sum()),
        end_open_positions=int(daily_frame.open_position_count.iloc[-1]),
        end_holdings_value=float(daily_frame.holdings_value.iloc[-1]),
        end_cash=float(daily_frame.cash.iloc[-1]),
    )
    return dict(daily=daily_frame, orders=order_frame, positions=position_frame, summary=summary)
