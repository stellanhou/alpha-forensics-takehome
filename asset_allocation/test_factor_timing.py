"""Tests of release-time, label-calendar, and paired-comparison contracts."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from factor_timing import FACTORS, align_factor, build_labels, compare_factor, load_panels


class FactorTimingTests(unittest.TestCase):
    def test_release_close_boundary_and_asset_isolation(self):
        snapshots = pd.DataFrame({
            "date": pd.to_datetime(["2021-01-04", "2021-01-05", "2021-01-04"]),
            "asset_id": ["A", "A", "B"], "factor_03": [10., 20., 99.],
            "factor_03_available_on": pd.to_datetime(["2021-01-07", "2021-01-08", "2021-01-07"]),
        })
        signals = pd.DataFrame({
            "date": pd.to_datetime(["2021-01-06", "2021-01-07", "2021-01-08", "2021-01-07"]),
            "asset_id": ["A", "A", "A", "B"],
        })
        result = align_factor(signals, snapshots, "factor_03").set_index(["date", "asset_id"])
        self.assertTrue(pd.isna(result.loc[("2021-01-06", "A"), "factor_03"]))
        self.assertEqual(result.loc[("2021-01-07", "A"), "factor_03"], 10.)
        self.assertEqual(result.loc[("2021-01-08", "A"), "factor_03"], 20.)
        self.assertEqual(result.loc[("2021-01-07", "B"), "factor_03"], 99.)

    def test_latest_missing_snapshot_is_not_replaced_by_old_value(self):
        snapshots = pd.DataFrame({
            "date": pd.to_datetime(["2021-01-04", "2021-01-05"]),
            "asset_id": ["A", "A"], "factor_03": [10., np.nan],
            "factor_03_available_on": pd.to_datetime(["2021-01-07", "2021-01-08"]),
        })
        signals = pd.DataFrame({"date": pd.to_datetime(["2021-01-11"]), "asset_id": ["A"]})
        result = align_factor(signals, snapshots, "factor_03").iloc[0]
        self.assertTrue(pd.isna(result.factor_03))
        self.assertEqual(result.factor_03_reference_date, pd.Timestamp("2021-01-05"))

    def test_latest_reference_wins_when_releases_share_a_date(self):
        snapshots = pd.DataFrame({
            "date": pd.to_datetime(["2021-01-04", "2021-01-05"]),
            "asset_id": ["A", "A"], "factor_03": [10., 20.],
            "factor_03_available_on": pd.to_datetime(["2021-01-07", "2021-01-07"]),
        })
        signals = pd.DataFrame({"date": pd.to_datetime(["2021-01-07"]), "asset_id": ["A"]})
        self.assertEqual(align_factor(signals, snapshots, "factor_03").iloc[0].factor_03, 20.)

    def test_reordered_release_dates_fail_explicitly(self):
        snapshots = pd.DataFrame({
            "date": pd.to_datetime(["2021-01-04", "2021-01-05"]),
            "asset_id": ["A", "A"], "factor_03": [10., 20.],
            "factor_03_available_on": pd.to_datetime(["2021-01-08", "2021-01-07"]),
        })
        with self.assertRaisesRegex(ValueError, "release order"):
            align_factor(snapshots[["date", "asset_id"]], snapshots, "factor_03")

    def test_labels_use_business_dates_and_require_actual_endpoints(self):
        dates = pd.bdate_range("2021-01-04", periods=10)
        market = pd.DataFrame({"date": dates, "asset_id": "A", "open": np.arange(10.) + 100})
        labels = build_labels(market).set_index("date")
        self.assertEqual(labels.loc["2021-01-08", "entry_date"], pd.Timestamp("2021-01-11"))
        self.assertAlmostEqual(labels.loc["2021-01-04", "forward_5d"], 106 / 101 - 1)
        self.assertTrue(pd.isna(labels.loc["2021-01-15", "forward_5d"]))
        for missing in ["2021-01-05", "2021-01-12"]:
            with_gap = market.loc[market.date != pd.Timestamp(missing)]
            result = build_labels(with_gap).set_index("date")
            self.assertTrue(pd.isna(result.loc["2021-01-04", "forward_5d"]))

    def test_paired_ic_uses_same_assets_and_reranks(self):
        raw = pd.DataFrame({
            "date": pd.to_datetime(["2021-01-04"] * 4), "asset_id": list("ABCD"),
            "factor_03": [1., 2., 3., 4.], "forward_5d": [1., 2., 3., -100.],
        })
        pit = raw[["date", "asset_id"]].assign(factor_03=[3., 2., 1., np.nan])
        daily, summary = compare_factor(raw, pit, "factor_03")
        self.assertEqual(daily.iloc[0].n_assets, 3)
        self.assertAlmostEqual(daily.iloc[0].raw_rank_ic, 1.)
        self.assertAlmostEqual(daily.iloc[0].pit_rank_ic, -1.)
        self.assertEqual(summary.iloc[0].common_rows, 3)
        self.assertAlmostEqual(summary.iloc[0].delta_pit_minus_raw, -2.)

    def test_align_before_membership_and_preserve_other_factors(self):
        dates = pd.bdate_range("2021-01-04", periods=12)
        keys = pd.DataFrame({"date": dates, "asset_id": "A"})
        market = keys.assign(open=np.arange(12.) + 100)
        factors = keys.assign(**{f: np.arange(12.) + i * 100 for i, f in enumerate(FACTORS)})
        universe = keys.assign(is_listed=True, is_member=np.arange(12) >= 3)
        availability = keys.assign(
            factor_03_available_on=dates + pd.offsets.BDay(3),
            factor_12_available_on=dates + pd.offsets.BDay(1), published_at="after_close",
        )
        with TemporaryDirectory() as tmp:
            data = Path(tmp) / "data"
            data.mkdir()
            for name, frame in [("market_daily", market), ("factors", factors),
                                ("universe", universe), ("factor_availability", availability)]:
                frame.to_csv(data / f"{name}.csv.gz", index=False)
            raw, pit, audit = load_panels(Path(tmp))
        self.assertEqual(pit.iloc[0].date, pd.Timestamp("2021-01-07"))
        self.assertEqual(pit.iloc[0].factor_03_reference_date, pd.Timestamp("2021-01-04"))
        self.assertEqual(pit.iloc[0].factor_03, factors.iloc[0].factor_03)
        self.assertEqual(pit.iloc[0].factor_12, factors.iloc[2].factor_12)
        self.assertEqual(raw.iloc[0].factor_01, pit.iloc[0].factor_01)
        self.assertEqual(audit["availability"][0]["selected_future_release_rows"], 0)


if __name__ == "__main__":
    unittest.main()
