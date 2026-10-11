"""Fixed candidate rebalance contracts: target rules and executable cash accounts."""
import unittest

import numpy as np
import pandas as pd

from portfolio_ledger import simulate
from rebalance_ledger import simulate_rebalance, target_weights


def inputs(assets=("A",), periods=18):
    dates = pd.bdate_range("2023-01-02", periods=periods)
    keys = pd.MultiIndex.from_product([dates, assets], names=["date", "asset_id"]).to_frame(index=False)
    predictions = keys.assign(prediction=1., forward_5d=np.nan)
    market = keys.assign(open=100., close=100.)
    constraints = keys.assign(can_buy=True, can_sell=True, fee_bps_one_way=0.,
                              spread_bps=0., impact_bps_proxy=0.)
    return predictions, market, constraints


def run_case(values, policy="NET", selection="top_decile", **options):
    predictions, market, constraints = values
    return simulate_rebalance(predictions, market, constraints,
                              start_signal=market.date.min(), end_date=market.date.max(),
                              policy=policy, selection=selection, **options)


def ranked_scores():
    return pd.Series(np.arange(20., 0., -1), index=list("ABCDEFGHIJKLMNOPQRST"))


def sleeve_orders(result, date, sleeve=0):
    orders = result["orders"]
    return orders.loc[orders.execution_date.eq(date) & orders.sleeve.eq(sleeve)]


class RebalanceLedgerTests(unittest.TestCase):
    def test_zero_cost_net_reproduces_full_rotation_with_opening_gaps(self):
        for gap_price in [50., 200.]:
            with self.subTest(gap_price=gap_price):
                values = inputs(("A", "B"))
                predictions, market, constraints = values
                gap_day = market.date.drop_duplicates().iloc[6]
                market.loc[market.asset_id.eq("A") & market.date.ge(gap_day), ["open", "close"]] = gap_price
                old = simulate(predictions, market, constraints,
                               start_signal=market.date.min(), end_date=market.date.max(),
                               selection="equal_weight", cost_mode="zero", constraint_mode="none")
                new = run_case(values, selection="equal_weight")
                np.testing.assert_allclose(new["daily"].nav, old["daily"].nav, atol=1e-12, rtol=1e-12)
                keys = ["date", "sleeve", "asset_id"]
                left = old["positions"].set_index(keys)[["shares", "market_value"]].sort_index()
                right = new["positions"].set_index(keys)[["shares", "market_value"]].sort_index()
                left, right = left.align(right, join="outer", fill_value=0.)
                np.testing.assert_allclose(left, right, atol=1e-12, rtol=1e-12)
                self.assertEqual(new["summary"]["end_open_positions"], 0)
                self.assertNotIn("max_review_turnover", new["summary"])
                self.assertNotIn("signal_weight", new["plans"].columns)

    def test_review_budget_limits_initial_purchase(self):
        values = inputs()
        dates = values[1].date.drop_duplicates().tolist()
        result = run_case(values, max_review_turnover=.2)
        first = result["plans"].loc[lambda x: x.signal_date.eq(dates[0])].iloc[0]
        self.assertEqual(first.target_weight, 1.)
        self.assertEqual(first.signal_weight, 0.)
        self.assertEqual(first.uncapped_review_turnover, 1.)
        self.assertEqual(first.review_turnover_scale, .2)
        self.assertEqual(first.effective_weight, .2)
        self.assertFalse(result["plans"].freeze_quantity.any())
        self.assertAlmostEqual(sleeve_orders(result, dates[1]).filled_notional.sum(), .04)
        self.assertEqual(result["summary"]["max_review_turnover"], .2)
        ordinary = result["plans"].loc[lambda x: ~x.terminal].copy()
        ordinary["planned_turnover"] = (ordinary.effective_weight - ordinary.signal_weight).abs()
        self.assertTrue(ordinary.groupby(["signal_date", "sleeve"]).planned_turnover.sum().le(.2 + 1e-12).all())

    def test_review_budget_uses_drifted_holdings_and_partially_retains_departed_name(self):
        values = inputs(("A", "B"))
        predictions, market, _ = values
        dates = market.date.drop_duplicates().tolist()
        predictions.loc[predictions.asset_id.eq("A"), "prediction"] = 2.
        predictions.loc[predictions.asset_id.eq("B") & predictions.date.ge(dates[5]), "prediction"] = 3.
        market.loc[market.asset_id.eq("A") & market.date.ge(dates[5]), ["open", "close"]] = 200.
        result = run_case(values, max_review_turnover=.2)
        plan = result["plans"].loc[lambda x: x.sleeve.eq(0) & x.signal_date.eq(dates[5])].set_index("asset_id")
        # The first .04 purchase is now worth .08; cash stays .16, so A is
        # one third of the .24 sleeve NAV, rather than its previous .2 target.
        self.assertAlmostEqual(plan.loc["A", "signal_weight"], 1 / 3)
        np.testing.assert_allclose(plan.uncapped_review_turnover, 4 / 3)
        np.testing.assert_allclose(plan.review_turnover_scale, .15)
        self.assertEqual(plan.loc["A", "target_weight"], 0.)
        self.assertAlmostEqual(plan.loc["A", "effective_weight"], 17 / 60)
        self.assertAlmostEqual(plan.loc["B", "effective_weight"], .15)
        self.assertAlmostEqual((plan.effective_weight - plan.signal_weight).abs().sum(), .2)
        held = result["positions"].loc[lambda x: x.date.eq(dates[6]) & x.sleeve.eq(0)]
        self.assertEqual(set(held.asset_id), {"A", "B"})
        self.assertAlmostEqual(sleeve_orders(result, dates[6]).filled_notional.sum(), .24 * .2)

    def test_review_budget_handles_zero_demand_and_zero_budget(self):
        values = inputs()
        dates = values[1].date.drop_duplicates().tolist()
        result = run_case(values, max_review_turnover=1.)
        unchanged = result["plans"].loc[lambda x: x.sleeve.eq(0) & x.signal_date.eq(dates[5])].iloc[0]
        self.assertEqual(unchanged.uncapped_review_turnover, 0.)
        self.assertEqual(unchanged.review_turnover_scale, 1.)
        self.assertEqual(unchanged.effective_weight, 1.)
        self.assertFalse(sleeve_orders(result, dates[6]).filled_shares.gt(0).any())
        zero = run_case(values, max_review_turnover=0.)
        self.assertFalse(zero["orders"].filled_shares.gt(0).any())
        np.testing.assert_array_equal(zero["daily"].nav, 1.)

    def test_review_budget_exempts_terminal_liquidation(self):
        values = inputs()
        dates = values[1].date.drop_duplicates().tolist()
        result = run_case(values, max_review_turnover=.2)
        terminal = result["plans"].loc[lambda x: x.sleeve.eq(0) & x.terminal].iloc[0]
        self.assertGreater(terminal.uncapped_review_turnover, .2)
        self.assertEqual(terminal.review_turnover_scale, 1.)
        self.assertEqual(terminal.effective_weight, 0.)
        self.assertAlmostEqual(sleeve_orders(result, dates[16]).filled_notional.sum(), .12)
        self.assertEqual(result["summary"]["end_open_positions"], 0)

    def test_review_budget_rejects_unsupported_policy_and_invalid_limit(self):
        for policy in ["BOTH", "DAILY_NET"]:
            with self.subTest(policy=policy), self.assertRaisesRegex(ValueError, "only for NET"):
                run_case(inputs(), policy=policy, max_review_turnover=.2)
        for limit in [-.1, float("nan"), float("inf")]:
            with self.subTest(limit=limit), self.assertRaisesRegex(ValueError, "finite and nonnegative"):
                run_case(inputs(), max_review_turnover=limit)

    def test_unchanged_net_position_does_not_sell_and_rebuy(self):
        values = inputs()
        dates = values[1].date.drop_duplicates().tolist()
        result = run_case(values)
        review_orders = sleeve_orders(result, dates[6])
        self.assertFalse(review_orders.filled_shares.gt(0).any())
        positions = result["positions"].query("sleeve == 0").set_index("date")
        self.assertAlmostEqual(positions.loc[dates[5], "shares"], positions.loc[dates[6], "shares"])
        filled = result["orders"].loc[result["orders"].filled_shares.gt(0)]
        sides = filled.groupby(["execution_date", "sleeve", "asset_id"]).side.nunique()
        self.assertTrue(sides.le(1).all())

    def test_permissions_and_costs_use_previous_day_and_final_flags_only_audit(self):
        values = inputs(periods=7)
        dates = values[1].date.drop_duplicates().tolist()
        constraints = values[2]
        constraints.loc[constraints.date.eq(dates[0]), "fee_bps_one_way"] = 100.
        constraints.loc[constraints.date.eq(dates[1]), "fee_bps_one_way"] = 1000.
        constraints.loc[constraints.date.eq(dates[1]), "can_buy"] = False
        values[1].loc[values[1].date.eq(dates[1]), "close"] = 120.
        result = run_case(values)
        order = sleeve_orders(result, dates[1]).loc[lambda x: x.side.eq("buy")].iloc[0]
        self.assertAlmostEqual(order.filled_notional, .2 / 1.01)
        self.assertAlmostEqual(order.fee, order.filled_notional * .01)
        self.assertAlmostEqual(order.fill_price, 100.)
        self.assertTrue(order.actual_flag_conflict)

    def test_new_plan_replaces_overdue_sell_before_any_retry(self):
        values = inputs(("A", "B"), periods=21)
        predictions, market, constraints = values
        dates = market.date.drop_duplicates().tolist()
        predictions.loc[predictions.asset_id.eq("A"), "prediction"] = 2.
        predictions.loc[predictions.asset_id.eq("B") & predictions.date.between(dates[5], dates[9]), "prediction"] = 3.
        constraints.loc[constraints.asset_id.eq("A") & constraints.date.between(dates[5], dates[9]), "can_sell"] = False
        result = run_case(values)
        attempted = result["orders"].loc[lambda x: x.sleeve.eq(0) & x.asset_id.eq("A")
                                            & x.side.eq("sell") & x.execution_date.between(dates[6], dates[10])]
        self.assertGreater(len(attempted), 0)
        self.assertTrue(attempted.filled_shares.eq(0).all())
        refreshed = sleeve_orders(result, dates[11])
        self.assertFalse(refreshed.filled_shares.gt(0).any())
        held = result["positions"].loc[lambda x: x.date.eq(dates[11]) & x.sleeve.eq(0)]
        self.assertEqual(held.asset_id.tolist(), ["A"])
        self.assertAlmostEqual(held.shares.iloc[0], .002)

    def test_fee_inclusive_scaling_does_not_redistribute_blocked_buy_budget(self):
        values = inputs(("A", "B", "C"), periods=7)
        dates = values[1].date.drop_duplicates().tolist()
        constraints = values[2]
        constraints["fee_bps_one_way"] = 100.
        constraints.loc[constraints.date.eq(dates[0]) & constraints.asset_id.eq("A"), "can_buy"] = False
        result = run_case(values, selection="equal_weight")
        orders = sleeve_orders(result, dates[1]).set_index("asset_id")
        self.assertEqual(orders.loc["A", "filled_shares"], 0.)
        for asset in ["B", "C"]:
            self.assertAlmostEqual(orders.loc[asset, "filled_notional"], .2 / 3 / 1.01)
            self.assertAlmostEqual(orders.loc[asset, "fee"], orders.loc[asset, "filled_notional"] * .01)
        day = result["daily"].set_index("date").loc[dates[1]]
        self.assertAlmostEqual(day.sleeve_0_cash, .2 / 3)
        self.assertTrue(result["daily"].cash.ge(-1e-12).all())

    def test_missing_quote_writes_off_existing_position_without_a_fill(self):
        predictions, market, constraints = inputs(("A", "B"))
        dates = market.date.drop_duplicates().tolist()
        predictions.loc[predictions.asset_id.eq("A"), "prediction"] = 2.
        keep = ~(market.asset_id.eq("A") & market.date.ge(dates[2]))
        result = run_case((predictions.loc[keep], market.loc[keep], constraints.loc[keep]))
        events = result["orders"].loc[lambda x: x.side.eq("writeoff")]
        self.assertEqual(len(events), 1)
        self.assertEqual(events.execution_date.iloc[0], dates[2])
        self.assertAlmostEqual(events.writeoff_amount.iloc[0], .2)
        self.assertEqual(events.filled_notional.iloc[0], 0.)
        self.assertEqual(events.filled_shares.iloc[0], 0.)
        self.assertTrue(pd.isna(events.fill_price.iloc[0]))
        self.assertAlmostEqual(result["summary"]["total_writeoff"], .2)
        self.assertAlmostEqual(result["summary"]["final_nav"], .8)

    def test_terminal_zero_targets_override_ema_and_band_but_not_sell_permissions(self):
        for policy in ["NET", "WEIGHT", "EMA50"]:
            with self.subTest(policy=policy):
                values = inputs(periods=7)
                dates = values[1].date.drop_duplicates().tolist()
                result = run_case(values, policy=policy)
                terminal = result["plans"].loc[lambda x: x.sleeve.eq(0) & x.signal_date.eq(dates[5])]
                self.assertGreater(len(terminal), 0)
                self.assertTrue(terminal.terminal.all())
                self.assertTrue(terminal.target_weight.eq(0).all())
                self.assertEqual(result["summary"]["end_open_positions"], 0)
        values = inputs(periods=7)
        values[2]["can_sell"] = False
        values[1].loc[values[1].date.eq(values[1].date.max()), "close"] = 110.
        blocked = run_case(values)
        self.assertEqual(blocked["summary"]["end_open_positions"], 1)
        self.assertAlmostEqual(blocked["summary"]["final_nav"], 1.02)

    def test_daily_frequency_uses_one_account_and_terminal_previous_close(self):
        values = inputs(periods=18)
        dates = values[1].date.drop_duplicates().tolist()
        result = run_case(values, policy="DAILY_NET")
        reviews = result["reviews"]
        self.assertEqual(set(reviews.sleeve), {0})
        self.assertTrue(set(dates[:6]).issubset(set(reviews.signal_date)))
        self.assertTrue(reviews.loc[reviews.signal_date.eq(dates[-2]), "terminal"].all())
        terminal = result["plans"].loc[lambda x: x.signal_date.eq(dates[-2])]
        self.assertGreater(len(terminal), 0)
        self.assertTrue(terminal.target_weight.eq(0).all())
        first = sleeve_orders(result, dates[1]).loc[lambda x: x.side.eq("buy")].iloc[0]
        self.assertAlmostEqual(first.filled_notional, 1.)
        self.assertEqual(result["summary"]["end_open_positions"], 0)

    def test_rank_buffer_retains_actual_holdings_not_prior_unfilled_targets(self):
        result = target_weights(ranked_scores(), {"C": .01, "Z": .01}, "RANK", {"D": .5, "E": .5})
        self.assertEqual(set(result), {"A", "C"})
        self.assertAlmostEqual(result["A"], .5)
        self.assertAlmostEqual(result["C"], .5)

    def test_rank_fill_keeps_whole_tie_group_without_asset_name_priority(self):
        scores = ranked_scores()
        scores.loc["C"] = scores.loc["B"]
        first = target_weights(scores, {}, "RANK", {})
        second = target_weights(scores.iloc[::-1], {}, "RANK", {})
        self.assertEqual(first, second)
        self.assertEqual(set(first), {"A", "B", "C"})
        np.testing.assert_allclose(list(first.values()), 1 / 3)

    def test_weight_band_preserves_quantity_inside_relative_ten_percent_band(self):
        values = inputs(("A", "B"))
        dates = values[1].date.drop_duplicates().tolist()
        values[1].loc[values[1].asset_id.eq("A") & values[1].date.ge(dates[5]), ["open", "close"]] = 95.
        result = run_case(values, policy="WEIGHT", selection="equal_weight")
        plan = result["plans"].loc[lambda x: x.sleeve.eq(0) & x.signal_date.eq(dates[5])]
        self.assertTrue(plan.freeze_quantity.all())
        np.testing.assert_allclose(plan.frozen_quantity, .001)
        self.assertFalse(sleeve_orders(result, dates[6]).filled_shares.gt(0).any())

    def test_weight_band_trades_to_nearest_boundary_when_outside(self):
        values = inputs(("A", "B"))
        dates = values[1].date.drop_duplicates().tolist()
        values[1].loc[values[1].asset_id.eq("A") & values[1].date.ge(dates[5]), ["open", "close"]] = 80.
        result = run_case(values, policy="WEIGHT", selection="equal_weight")
        plan = result["plans"].loc[lambda x: x.sleeve.eq(0) & x.signal_date.eq(dates[5])].set_index("asset_id")
        self.assertAlmostEqual(plan.loc["A", "effective_weight"], .45)
        self.assertAlmostEqual(plan.loc["B", "effective_weight"], .55)
        self.assertFalse(plan.freeze_quantity.any())

    def test_rank_exit_bypasses_weight_band(self):
        values = inputs(("A", "B"))
        predictions, market, _ = values
        dates = market.date.drop_duplicates().tolist()
        predictions.loc[predictions.asset_id.eq("A"), "prediction"] = 2.
        predictions.loc[predictions.asset_id.eq("B") & predictions.date.ge(dates[5]), "prediction"] = 3.
        result = run_case(values, policy="WEIGHT")
        plan = result["plans"].loc[lambda x: x.sleeve.eq(0) & x.signal_date.eq(dates[5])].set_index("asset_id")
        self.assertEqual(plan.loc["A", "target_weight"], 0.)
        self.assertEqual(plan.loc["A", "effective_weight"], 0.)
        sold = sleeve_orders(result, dates[6]).loc[lambda x: x.asset_id.eq("A") & x.side.eq("sell")]
        self.assertAlmostEqual(sold.filled_shares.sum(), .002)

    def test_ema_half_step_drops_ineligible_assets_without_renormalizing(self):
        result = target_weights(ranked_scores(), {}, "EMA50", {"A": .2, "C": .4, "Z": .4})
        self.assertEqual(set(result), {"A", "B", "C"})
        self.assertAlmostEqual(result["A"], .35)
        self.assertAlmostEqual(result["B"], .25)
        self.assertAlmostEqual(result["C"], .2)
        self.assertAlmostEqual(sum(result.values()), .8)
        first = target_weights(ranked_scores(), {}, "EMA50", {})
        self.assertAlmostEqual(sum(first.values()), .5)

    def test_taper_preserves_selection_and_tied_scores_receive_equal_weights(self):
        scores = ranked_scores()
        scores.loc["C"] = scores.loc["B"]
        baseline = target_weights(scores, {}, "NET", {})
        tapered = target_weights(scores, {}, "TAPER", {})
        self.assertEqual(set(tapered), set(baseline))
        self.assertAlmostEqual(sum(tapered.values()), 1.)
        self.assertAlmostEqual(tapered["A"], .5)
        self.assertAlmostEqual(tapered["B"], .25)
        self.assertAlmostEqual(tapered["C"], .25)

    def test_score_caps_redistribute_and_insufficient_capacity_stays_cash(self):
        scores = pd.Series([100., 10., 10.] + [0.] * 27, index=[f"S{i:02d}" for i in range(30)])
        result = target_weights(scores, {}, "SCORE", {})
        self.assertEqual(set(result), {"S00", "S01", "S02"})
        self.assertAlmostEqual(result["S00"], 2 / 3)
        self.assertAlmostEqual(result["S01"], 1 / 6)
        self.assertAlmostEqual(result["S02"], 1 / 6)
        scores = pd.Series([100.] + [1.] * 19, index=ranked_scores().index)
        limited = target_weights(scores, {}, "SCORE", {})
        self.assertEqual(set(limited), {"A"})
        self.assertAlmostEqual(limited["A"], .1)
        self.assertEqual(target_weights(pd.Series(1., index=scores.index), {}, "SCORE", {}), {})


if __name__ == "__main__":
    unittest.main()
