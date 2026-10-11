"""Contracts for complete signals, archived-model parity and NAV period accounting."""
import unittest

import numpy as np
import pandas as pd

from portfolio_model_comparison import (
    period_statistics, select_complete_signals, verify_archived,
)


def predictions():
    return pd.DataFrame({
        "date": pd.to_datetime(["2024-01-02", "2024-01-02", "2024-01-03"]),
        "asset_id": ["A", "B", "A"],
        "prediction": [.01, -.02, .03],
    })


def ledger():
    # The first year's final NAV is the second year's invested starting capital.
    daily = pd.DataFrame({
        "date": pd.to_datetime(["2023-12-28", "2023-12-29", "2024-01-01", "2024-01-02"]),
        "nav": [1., .8, .72, .88],
        "daily_return": [np.nan, -.2, -.1, .88 / .72 - 1],
        "cash": [1., .1, .1, .88],
        "gross_turnover": [99., .2, .3, .4],
        "fee": [0., 0., 0., 0.],
        "spread_cost": [0., 0., 0., 0.],
        "impact_cost": [0., 0., 0., 0.],
        "writeoff": [0., 0., 0., 0.],
        "actual_flag_conflict_count": [0, 0, 0, 0],
    })
    orders = pd.DataFrame({
        "execution_date": pd.to_datetime(["2023-12-29", "2024-01-01"]),
        "filled_shares": [1., 2.],
        "actual_flag_conflict": [False, False],
    })
    return daily, orders


class PortfolioModelComparisonTests(unittest.TestCase):
    def test_complete_signals_keep_missing_future_labels_and_sort_keys(self):
        panel = pd.DataFrame({
            "date": pd.to_datetime(["2024-12-31", "2023-12-29", "2024-01-02", "2024-01-02"]),
            "asset_id": ["A", "A", "B", "A"],
            "forward_5d": [np.nan, .1, np.nan, .2],
        }, index=[10, 20, 30, 40])
        result = select_complete_signals(panel, 2024)
        self.assertEqual(result.asset_id.tolist(), ["A", "B", "A"])
        self.assertEqual(result.date.tolist(), pd.to_datetime([
            "2024-01-02", "2024-01-02", "2024-12-31",
        ]).tolist())
        self.assertEqual(result.forward_5d.isna().sum(), 2)
        self.assertEqual(result.index.tolist(), [0, 1, 2])

    def test_archive_parity_accepts_complete_prediction_superset(self):
        result = predictions()
        archived = result.iloc[[2, 0]].reset_index(drop=True)
        self.assertAlmostEqual(verify_archived(result, archived), 0.)

    def test_archive_parity_rejects_missing_old_prediction_key(self):
        archived = predictions().iloc[[0, 2]].reset_index(drop=True)
        result = predictions().iloc[[0, 1]].reset_index(drop=True)
        with self.assertRaises(AssertionError):
            verify_archived(result, archived)

    def test_archive_parity_rejects_changed_old_prediction(self):
        result = predictions()
        archived = result.iloc[[0, 2]].reset_index(drop=True)
        result.loc[2, "prediction"] += .001
        with self.assertRaises(AssertionError):
            verify_archived(result, archived)

    def test_yearly_return_uses_prior_year_nav_and_drawdown_includes_start(self):
        daily, orders = ledger()
        first = period_statistics(daily, orders, "2023")
        second = period_statistics(daily, orders, "2024")
        self.assertAlmostEqual(first["total_return"], -.2)
        self.assertAlmostEqual(first["max_drawdown"], -.2)
        self.assertAlmostEqual(second["total_return"], .88 / .8 - 1)
        # Starting from .8, the first 2024 mark at .72 is already a 10% drawdown.
        self.assertAlmostEqual(second["max_drawdown"], -.1)
        self.assertAlmostEqual(second["mean_daily_turnover"], .35)

    def test_full_return_keeps_whole_nav_path_and_turnover_uses_return_days(self):
        daily, orders = ledger()
        result = period_statistics(daily, orders, "FULL")
        self.assertAlmostEqual(result["total_return"], -.12)
        self.assertAlmostEqual(result["max_drawdown"], -.28)
        self.assertAlmostEqual(result["mean_daily_turnover"], .3)


if __name__ == "__main__":
    unittest.main()
