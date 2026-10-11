"""Timing, self-financing, execution information, and exit contracts."""
import unittest

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay

from portfolio_ledger import simulate


class PortfolioLedgerTests(unittest.TestCase):
    def inputs(self, assets=("A",), periods=14):
        dates = pd.bdate_range("2023-01-02", periods=periods)
        keys = pd.MultiIndex.from_product([dates, assets], names=["date", "asset_id"]).to_frame(index=False)
        predictions = keys.assign(prediction=1., forward_5d=np.nan)
        market = keys.assign(open=100., close=100.)
        constraints = keys.assign(
            can_buy=True, can_sell=True, fee_bps_one_way=0., spread_bps=0., impact_bps_proxy=0.,
        )
        return predictions, market, constraints

    def run_case(self, inputs, **kwargs):
        predictions, market, constraints = inputs
        return simulate(
            predictions, market, constraints, start_signal=market.date.min(),
            end_date=market.date.max(), **kwargs,
        )

    def test_five_day_expiry_and_daily_accounting_with_initial_cash(self):
        result = self.run_case(self.inputs())
        orders, daily, summary = result["orders"], result["daily"], result["summary"]
        first_buy = orders.loc[(orders.sleeve == 0) & (orders.side == "buy")].iloc[0]
        first_sell = orders.loc[(orders.sleeve == 0) & (orders.side == "sell")].iloc[0]
        self.assertEqual(first_buy.signal_date, pd.Timestamp("2023-01-02"))
        self.assertEqual(first_buy.execution_date, pd.Timestamp("2023-01-03"))
        self.assertEqual(first_sell.execution_date, first_buy.execution_date + BDay(5))
        self.assertTrue(np.allclose(daily.nav, 1.))
        self.assertTrue(np.allclose(daily.nav, daily.cash + daily.holdings_value))
        self.assertEqual(daily.iloc[0].cash, 1.)
        self.assertTrue(pd.isna(daily.iloc[0].daily_return))
        self.assertEqual(summary["n_return_days"], len(daily) - 1)
        self.assertEqual(summary["total_return"], 0.)
        self.assertEqual(summary["max_drawdown"], 0.)
        self.assertEqual(summary["end_open_positions"], 0)
        # Unknown forward labels are deliberately present in every signal row.
        self.assertGreater(len(orders.loc[orders.side == "buy"]), 0)

    def test_buy_permission_leaves_cash_without_reallocation(self):
        inputs = self.inputs(("A", "B"))
        constraints = inputs[2]
        constraints.loc[(constraints.date == "2023-01-02") & (constraints.asset_id == "A"), "can_buy"] = False
        result = self.run_case(inputs, selection="equal_weight")
        first = result["orders"].loc[result["orders"].execution_date == "2023-01-03"].set_index("asset_id")
        self.assertEqual(first.loc["A", "reason"], "lagged_buy_blocked")
        self.assertAlmostEqual(first.loc["B", "filled_notional"], .1)
        day = result["daily"].set_index("date").loc["2023-01-03"]
        self.assertAlmostEqual(day.sleeve_0_cash, .1)
        self.assertAlmostEqual(day.sleeve_0_market_value, .1)

    def test_partial_blocked_exit_only_blocks_same_asset_and_preserves_expiry(self):
        inputs = self.inputs(("A", "B"))
        constraints = inputs[2]
        constraints.loc[(constraints.date == "2023-01-09") & (constraints.asset_id == "A"), "can_sell"] = False
        result = self.run_case(inputs, selection="equal_weight")
        orders = result["orders"]
        first_expiry = orders.loc[(orders.execution_date == "2023-01-10") & (orders.sleeve == 0)]
        sells = first_expiry.loc[first_expiry.side == "sell"].set_index("asset_id")
        self.assertEqual(sells.loc["A", "reason"], "lagged_sell_blocked")
        self.assertEqual(sells.loc["B", "reason"], "filled")
        buys = first_expiry.loc[first_expiry.side == "buy"].set_index("asset_id")
        self.assertEqual(buys.loc["A", "reason"], "existing_unsold_position")
        self.assertEqual(buys.loc["B", "reason"], "filled")
        self.assertAlmostEqual(buys.loc["B", "requested_budget"], .05)
        self.assertAlmostEqual(buys.loc["B", "filled_notional"], .05)
        day = result["daily"].set_index("date").loc["2023-01-10"]
        self.assertAlmostEqual(day.sleeve_0_cash, .05)
        self.assertAlmostEqual(day.sleeve_0_market_value, .15)
        positions = result["positions"]
        end_positions = positions.loc[(positions.date == "2023-01-10") & (positions.sleeve == 0)].set_index("asset_id")
        self.assertEqual(end_positions.loc["A", "scheduled_exit_date"], pd.Timestamp("2023-01-10"))
        self.assertEqual(end_positions.loc["B", "scheduled_exit_date"], pd.Timestamp("2023-01-17"))
        retry = orders.loc[(orders.execution_date == "2023-01-11") & (orders.sleeve == 0)]
        self.assertEqual(retry.side.tolist(), ["sell"])
        self.assertEqual(retry.asset_id.tolist(), ["A"])
        self.assertEqual(retry.reason.tolist(), ["filled"])
        self.assertEqual(retry.scheduled_exit_date.iloc[0], pd.Timestamp("2023-01-10"))
        next_positions = positions.loc[(positions.date == "2023-01-11") & (positions.sleeve == 0)]
        self.assertEqual(next_positions.asset_id.tolist(), ["B"])
        self.assertEqual(next_positions.scheduled_exit_date.iloc[0], pd.Timestamp("2023-01-17"))

    def test_inclusive_cost_budget_and_cash_conservation(self):
        inputs = self.inputs(periods=7)
        constraints = inputs[2]
        constraints["fee_bps_one_way"] = 10.
        constraints["spread_bps"] = 20.
        constraints["impact_bps_proxy"] = 10.
        result = self.run_case(inputs)
        orders = result["orders"]
        buy, sell = orders.iloc[0], orders.iloc[1]
        self.assertEqual(orders.side.tolist(), ["buy", "sell"])
        self.assertAlmostEqual(buy.filled_notional, .2 / 1.003)
        self.assertAlmostEqual(buy.filled_notional + buy.fee + buy.spread_cost + buy.impact_cost, .2)
        expected_cash = .8 + .2 / 1.003 * .997
        self.assertAlmostEqual(result["summary"]["final_nav"], expected_cash)
        self.assertAlmostEqual(1 - expected_cash, orders[["fee", "spread_cost", "impact_cost"]].to_numpy().sum())
        self.assertTrue((result["daily"].filter(regex="_cash$") >= 0).all().all())
        self.assertAlmostEqual(result["summary"]["total_return"], expected_cash - 1)
        self.assertAlmostEqual(sell.filled_shares, buy.filled_shares)

    def test_next_open_gap_cannot_increase_signal_close_budget(self):
        inputs = self.inputs()
        market = inputs[1]
        market.loc[market.date == "2023-01-10", ["open", "close"]] = 200.
        result = self.run_case(inputs)
        orders = result["orders"]
        rebuy = orders.loc[(orders.execution_date == "2023-01-10") & (orders.sleeve == 0) & (orders.side == "buy")].iloc[0]
        self.assertAlmostEqual(rebuy.planned_budget, .2)
        self.assertAlmostEqual(rebuy.requested_budget, .2)
        self.assertAlmostEqual(rebuy.filled_notional, .2)
        day = result["daily"].set_index("date").loc["2023-01-10"]
        self.assertAlmostEqual(day.sleeve_0_cash, .2)
        self.assertAlmostEqual(day.sleeve_0_market_value, .2)
        market.loc[market.date == "2023-01-10", ["open", "close"]] = 50.
        down = self.run_case(inputs)
        rebuy = down["orders"].loc[(down["orders"].execution_date == "2023-01-10") & (down["orders"].sleeve == 0) & (down["orders"].side == "buy")].iloc[0]
        self.assertAlmostEqual(rebuy.planned_budget, .2)
        self.assertAlmostEqual(rebuy.requested_budget, .1)
        self.assertAlmostEqual(rebuy.filled_notional, .1)

    def test_execution_day_close_and_final_flag_cannot_change_existing_plan(self):
        inputs = self.inputs()
        original = self.run_case(inputs)
        inputs[1].loc[inputs[1].date == "2023-01-03", "close"] = 300.
        inputs[2].loc[inputs[2].date == "2023-01-03", "can_buy"] = False
        changed = self.run_case(inputs)
        original_order = original["orders"].loc[original["orders"].execution_date == "2023-01-03"]
        changed_order = changed["orders"].loc[changed["orders"].execution_date == "2023-01-03"]
        pd.testing.assert_frame_equal(
            original_order.drop(columns="actual_flag_conflict"),
            changed_order.drop(columns="actual_flag_conflict"),
        )
        self.assertFalse(original_order.actual_flag_conflict.iloc[0])
        self.assertTrue(changed_order.actual_flag_conflict.iloc[0])
        self.assertEqual(changed["summary"]["actual_flag_conflict_orders"], 1)
        self.assertAlmostEqual(changed["summary"]["actual_flag_conflict_notional"], .2)

    def test_costs_are_known_previous_day_proxies(self):
        inputs = self.inputs(periods=7)
        inputs[2].loc[inputs[2].date == "2023-01-03", "fee_bps_one_way"] = 100.
        result = self.run_case(inputs)
        self.assertEqual(result["orders"].iloc[0].fee, 0.)
        # Exit on Jan 10 uses the Jan 9 fee, not Jan 10's final proxy.
        inputs[2].loc[inputs[2].date == "2023-01-09", "fee_bps_one_way"] = 50.
        inputs[2].loc[inputs[2].date == "2023-01-10", "fee_bps_one_way"] = 100.
        result = self.run_case(inputs)
        self.assertAlmostEqual(result["orders"].iloc[1].fee, .2 * .005)

    def test_exit_without_price_is_zero_recovery_writeoff_not_fake_sale(self):
        predictions, market, constraints = self.inputs(("A", "B"))
        predictions.loc[predictions.asset_id == "A", "prediction"] = 2.
        keep = ~((market.asset_id == "A") & (market.date >= "2023-01-04"))
        result = self.run_case((predictions.loc[keep], market.loc[keep], constraints.loc[keep]))
        orders = result["orders"]
        writeoff = orders.loc[orders.side == "writeoff"]
        self.assertEqual(len(writeoff), 1)
        self.assertEqual(writeoff.execution_date.iloc[0], pd.Timestamp("2023-01-04"))
        self.assertAlmostEqual(writeoff.writeoff_amount.iloc[0], .2)
        self.assertEqual(writeoff.filled_notional.iloc[0], 0.)
        self.assertEqual(writeoff.filled_shares.iloc[0], 0.)
        self.assertTrue(pd.isna(writeoff.fill_price.iloc[0]))
        self.assertAlmostEqual(result["summary"]["total_writeoff"], .2)
        self.assertAlmostEqual(result["summary"]["final_nav"], .8)
        self.assertAlmostEqual(result["summary"]["max_drawdown"], -.2)

    def test_end_positions_are_marked_not_forcibly_sold(self):
        inputs = self.inputs(periods=7)
        inputs[2]["can_sell"] = False
        inputs[1].loc[inputs[1].date == "2023-01-10", "close"] = 110.
        result = self.run_case(inputs)
        summary = result["summary"]
        self.assertEqual(summary["end_open_positions"], 1)
        self.assertAlmostEqual(summary["end_holdings_value"], .22)
        self.assertAlmostEqual(summary["final_nav"], 1.02)
        self.assertEqual(result["orders"].iloc[-1].reason, "lagged_sell_blocked")
        self.assertEqual(result["orders"].iloc[-1].filled_notional, 0.)

    def test_comparison_modes_are_explicit_and_final_flags_only_audit(self):
        inputs = self.inputs(periods=7)
        inputs[2][["can_buy", "can_sell"]] = False
        inputs[2]["fee_bps_one_way"] = 100.
        no_constraints = self.run_case(inputs, cost_mode="zero", constraint_mode="none")
        self.assertEqual(no_constraints["summary"]["total_return"], 0.)
        self.assertEqual(no_constraints["summary"]["total_fee"], 0.)
        self.assertEqual(no_constraints["summary"]["actual_flag_conflict_orders"], 2)
        lagged = self.run_case(inputs, cost_mode="base", constraint_mode="lagged")
        self.assertEqual(lagged["summary"]["gross_traded_notional"], 0.)

    def test_gaps_and_missing_constraints_fail_instead_of_becoming_exits(self):
        predictions, market, constraints = self.inputs(("A", "B"))
        keep = ~((market.asset_id == "A") & (market.date == "2023-01-05"))
        with self.assertRaisesRegex(ValueError, "internal date gap"):
            self.run_case((predictions.loc[keep], market.loc[keep], constraints.loc[keep]))
        with self.assertRaisesRegex(ValueError, "keys differ"):
            self.run_case((predictions, market, constraints.iloc[1:]))


if __name__ == "__main__":
    unittest.main()
