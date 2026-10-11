"""Interpretation contracts for the new account diagnostics."""
import unittest

import numpy as np
import pandas as pd

import joint_factor_increment as study


class JointIncrementDiagnosticTests(unittest.TestCase):
    def test_risk_uses_equity_normalization_and_reports_cash_separately(self):
        dates = pd.bdate_range("2024-01-02", periods=2)
        daily = pd.DataFrame({"date": dates, "nav": [1., 1.]})
        positions = pd.DataFrame({"date": [dates[1]], "asset_id": ["A"], "market_value": [.4]})
        risk = pd.DataFrame({"date": dates.repeat(2), "asset_id": ["A", "B"] * 2,
                             "industry": ["I1", "I2"] * 2,
                             "market_cap": np.exp([1., 3.] * 2), "log_market_cap": [1., 3.] * 2})
        study._WORKER.update(risk=risk, risk_baseline=pd.DataFrame({"mean": 2., "std": np.sqrt(2)}, index=dates),
                             industry_baseline=pd.DataFrame({"I1": .5, "I2": .5}, index=dates))
        diagnostic, industry = study.risk_diagnostics(daily, positions)
        self.assertEqual(diagnostic.equity_fraction.tolist(), [0., .4])
        self.assertTrue(np.isnan(diagnostic.weighted_log_cap.iloc[0]))
        self.assertTrue(np.isnan(diagnostic.industry_active_l1_equity.iloc[0]))
        self.assertAlmostEqual(diagnostic.weighted_log_cap.iloc[1], 1.)
        self.assertAlmostEqual(diagnostic.size_z_equity.iloc[1], -1 / np.sqrt(2))
        self.assertAlmostEqual(diagnostic.industry_active_l1_equity.iloc[1], 1.)
        held = industry.loc[industry.date.eq(dates[1])].set_index("industry")
        self.assertAlmostEqual(held.weight_nav.sum(), .4)
        self.assertAlmostEqual(held.loc["I1", "weight_equity"], 1.)

    def test_budget_diagnostic_checks_actual_weights_and_terminal_exemption(self):
        dates = pd.bdate_range("2024-01-02", periods=2)
        reviews = pd.DataFrame({"signal_date": dates, "sleeve": [0, 0], "signal_nav": [1., 1.],
                                "terminal": [False, True]})
        plans = pd.DataFrame({"signal_date": dates.repeat(2), "sleeve": [0] * 4,
                              "signal_shares": [.4, .6, .3, .7], "signal_close": [1.] * 4,
                              "target_weight": [0., 1., 0., 0.],
                              "effective_weight": [.3, .7, 0., 0.]})
        values = study.plan_diagnostics(plans, reviews, .2)
        self.assertAlmostEqual(values.planned_gross_turnover.iloc[0], .2)
        self.assertAlmostEqual(values.ideal_gross_turnover.iloc[0], .8)
        self.assertTrue(values.budget_binding.iloc[0])
        self.assertFalse(values.budget_applies.iloc[1])
        self.assertAlmostEqual(values.planned_gross_turnover.iloc[1], 1.)
        plans.loc[0, "effective_weight"] = .2
        with self.assertRaises(AssertionError):
            study.plan_diagnostics(plans, reviews, .2)


if __name__ == "__main__":
    unittest.main()
