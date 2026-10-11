"""Point-in-time factor alignment and paired cross-sectional timing audit."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay


KEYS = ["date", "asset_id"]
FACTORS = [f"factor_{i:02d}" for i in range(1, 21)]
DELAYED = ["factor_03", "factor_12"]
PROJECT = Path(__file__).resolve().parent
DEFAULT_DATA = (
    PROJECT.parent / "Lab404-CUFEL-E" / "03-资产配置策略方向"
    / "Alpha_Forensics_Candidate_v1"
)


def validate_keys(frame: pd.DataFrame, name: str) -> pd.MultiIndex:
    if frame[KEYS].isna().any().any():
        raise ValueError(f"{name}: missing date or asset_id")
    return frame.set_index(KEYS, verify_integrity=True).index.sort_values()


def align_factor(
    signals: pd.DataFrame, snapshots: pd.DataFrame, factor: str
) -> pd.DataFrame:
    """Use the newest released reference date at each signal day's close.

    This package's releases are ordered by reference date within each asset.
    Enforce that property so latest-release asof also selects latest reference.
    Missing snapshot values are kept, rather than searching for an older value.
    """
    validate_keys(signals, "signals")
    validate_keys(snapshots, factor)
    available = f"{factor}_available_on"
    reference = f"{factor}_reference_date"
    ordered = snapshots.sort_values(["asset_id", "date"])
    if ordered[available].isna().any():
        raise ValueError(f"{factor}: missing release date")
    if (ordered[available] < ordered.date).any():
        raise ValueError(f"{factor}: release predates its reference date")
    if (ordered.groupby("asset_id")[available].diff() < pd.Timedelta(0)).any():
        raise ValueError(f"{factor}: release order differs from reference order")
    releases = ordered[KEYS + [factor, available]].rename(columns={"date": reference})
    releases = releases.sort_values([available, "asset_id", reference])
    aligned = pd.merge_asof(
        signals[KEYS].sort_values(["date", "asset_id"]),
        releases,
        left_on="date",
        right_on=available,
        by="asset_id",
        direction="backward",
        allow_exact_matches=True,
    )
    if (aligned[available] > aligned.date).any():
        raise AssertionError(f"{factor}: selected a future release")
    return aligned


def build_labels(market: pd.DataFrame) -> pd.DataFrame:
    """Match actual prices at t+1 and t+6 business days; never fill prices."""
    validate_keys(market, "market")
    if not np.isfinite(market.open).all() or (market.open <= 0).any():
        raise ValueError("market: open prices must be finite and positive")
    labels = market[KEYS].copy()
    labels["entry_date"] = labels.date + BDay(1)
    labels["exit_date"] = labels.date + BDay(6)
    for name in ["entry", "exit"]:
        prices = market[KEYS + ["open"]].rename(
            columns={"date": f"{name}_date", "open": f"{name}_open"}
        )
        labels = labels.merge(
            prices, on=[f"{name}_date", "asset_id"], how="left", validate="one_to_one"
        )
    labels["forward_5d"] = labels.exit_open / labels.entry_open - 1
    return labels


def load_panels(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Return eligible raw and PIT panels, sharing keys and unchanged labels."""
    data = root / "data"
    market = pd.read_csv(data / "market_daily.csv.gz", parse_dates=["date"])
    factors = pd.read_csv(data / "factors.csv.gz", parse_dates=["date"])
    universe = pd.read_csv(data / "universe.csv.gz", parse_dates=["date"])
    availability = pd.read_csv(
        data / "factor_availability.csv.gz",
        parse_dates=["date"] + [f"{f}_available_on" for f in DELAYED],
    )
    market_keys = validate_keys(market, "market")
    for name, frame in [("factors", factors), ("universe", universe), ("availability", availability)]:
        if not market_keys.equals(validate_keys(frame, name)):
            raise ValueError(f"{name}: keys differ from market; inspect before alignment")
    if not availability.published_at.eq("after_close").all():
        raise ValueError("availability: expected the documented after_close release convention")

    labels = build_labels(market)
    raw = labels.merge(factors, on=KEYS, validate="one_to_one").merge(
        universe, on=KEYS, validate="one_to_one"
    )
    pit = raw.copy()
    audit = []
    reference_scan = []
    for factor in DELAYED:
        available = f"{factor}_available_on"
        snapshots = factors[KEYS + [factor]].merge(
            availability[KEYS + [available]], on=KEYS, validate="one_to_one"
        )
        aligned = align_factor(market[KEYS], snapshots, factor)
        pit = pit.drop(columns=factor).merge(aligned, on=KEYS, validate="one_to_one")
        lag = np.busday_count(
            snapshots.date.to_numpy(dtype="datetime64[D]"),
            snapshots[available].to_numpy(dtype="datetime64[D]"),
        )
        audit.append({
            "factor": factor,
            "snapshot_rows": len(snapshots),
            "unavailable_on_reference_date_rows": int((snapshots[available] > snapshots.date).sum()),
            "release_lag_business_days": {
                str(int(k)): int(v) for k, v in pd.Series(lag).value_counts().sort_index().items()
            },
            "signals_before_first_release": int(aligned[available].isna().sum()),
            "selected_future_release_rows": int((aligned[available] > aligned.date).sum()),
        })
        # The supplied factor scan uses all assets, unlike the member-only model.
        all_assets = raw.dropna(subset=[factor, "forward_5d"])
        ic = daily_rank_ic(all_assets, [factor])
        reference_scan.append({
            "factor": factor, "scope": "all assets with labels; no is_member filter",
            "rows": len(all_assets), "valid_ic_dates": int(ic[factor].notna().sum()),
            "recomputed_mean_daily_rank_ic": float(ic[factor].mean()),
        })

    # Align from every snapshot BEFORE filtering membership and label eligibility.
    eligible = raw.is_listed & raw.is_member & raw.forward_5d.notna()
    eligible_keys = raw.loc[eligible, KEYS]
    raw = eligible_keys.merge(raw, on=KEYS, validate="one_to_one").sort_values(KEYS).reset_index(drop=True)
    pit = eligible_keys.merge(pit, on=KEYS, validate="one_to_one").sort_values(KEYS).reset_index(drop=True)
    untouched = [f for f in FACTORS if f not in DELAYED]
    pd.testing.assert_frame_equal(raw[KEYS + untouched + ["forward_5d"]], pit[KEYS + untouched + ["forward_5d"]])
    return raw, pit, {"availability": audit, "reference_scan": reference_scan}


def daily_rank_ic(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Spearman correlation computed on the caller's already selected rows."""
    columns = features + ["forward_5d"]
    ranks = frame.groupby("date")[columns].rank(method="average")
    centered = ranks - ranks.groupby(frame.date).transform("mean")
    target = centered["forward_5d"]
    sums = centered.pow(2).groupby(frame.date).sum()
    result = frame.groupby("date").size().rename("n_assets").to_frame()
    for feature in features:
        cross = (centered[feature] * target).groupby(frame.date).sum()
        result[feature] = cross / np.sqrt(sums[feature] * sums["forward_5d"])
    return result


def compare_factor(raw: pd.DataFrame, pit: pd.DataFrame, factor: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    paired = raw[KEYS + ["forward_5d", factor]].rename(columns={factor: "raw"}).merge(
        pit[KEYS + [factor]].rename(columns={factor: "pit"}), on=KEYS, validate="one_to_one"
    )
    full_raw = daily_rank_ic(paired.dropna(subset=["raw", "forward_5d"]), ["raw"])
    common = paired.dropna(subset=["raw", "pit", "forward_5d"])
    daily = daily_rank_ic(common, ["raw", "pit"]).rename(
        columns={"raw": "raw_rank_ic", "pit": "pit_rank_ic"}
    )
    daily["delta_pit_minus_raw"] = daily.pit_rank_ic - daily.raw_rank_ic
    daily["factor"] = factor
    daily = daily.reset_index()
    records = []
    periods = ["ALL"] + sorted(paired.date.dt.year.astype(str).unique())
    for period in periods:
        subset = daily if period == "ALL" else daily.loc[daily.date.dt.year == int(period)]
        original = full_raw if period == "ALL" else full_raw.loc[full_raw.index.year == int(period)]
        # Undefined IC on either side is excluded from BOTH means, with counts exposed.
        valid = subset.dropna(subset=["raw_rank_ic", "pit_rank_ic"])
        records.append({
            "factor": factor, "period": period,
            "common_rows": int(subset.n_assets.sum()),
            "common_dates": len(subset), "paired_valid_ic_dates": len(valid),
            "raw_eligible_rows": int(original.n_assets.sum()),
            "raw_eligible_valid_ic_dates": int(original.raw.notna().sum()),
            "raw_eligible_mean_daily_rank_ic": original.raw.mean(),
            "raw_common_mean_daily_rank_ic": valid.raw_rank_ic.mean(),
            "pit_common_mean_daily_rank_ic": valid.pit_rank_ic.mean(),
            "delta_pit_minus_raw": valid.delta_pit_minus_raw.mean(),
        })
    return daily, pd.DataFrame(records)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(root: Path, out: Path) -> None:
    raw, pit, audit = load_panels(root)
    daily_results, summaries = [], []
    for factor in DELAYED:
        daily, summary = compare_factor(raw, pit, factor)
        daily_results.append(daily)
        summaries.append(summary)
    summary = pd.concat(summaries, ignore_index=True)
    baseline_path = root / "baseline" / "baseline_factor_analysis.csv"
    baseline = pd.read_csv(baseline_path).set_index("factor")
    for record in audit["reference_scan"]:
        factor = record["factor"]
        actual = record["recomputed_mean_daily_rank_ic"]
        expected = baseline.loc[factor, "full_panel_naive_mean_daily_rank_ic"]
        record["supplied_mean_daily_rank_ic"] = float(expected)
        record["recomputed_minus_supplied"] = float(actual - expected)
        if not np.isclose(actual, expected, atol=1e-10, rtol=0):
            raise AssertionError(f"{factor}: all-asset IC {actual} does not reproduce supplied {expected}")

    out.mkdir(parents=True, exist_ok=True)
    pd.concat(daily_results, ignore_index=True).to_csv(out / "daily_rank_ic.csv", index=False)
    summary.to_csv(out / "timing_summary.csv", index=False)
    trace_columns = [f"{f}_{suffix}" for f in DELAYED for suffix in ["reference_date", "available_on"]]
    pit[KEYS + ["entry_date", "exit_date", "forward_5d"] + FACTORS + trace_columns].to_csv(
        out / "pit_research_panel.csv.gz", index=False,
        compression={"method": "gzip", "compresslevel": 1, "mtime": 0},
    )
    inputs = [root / "data" / f"{name}.csv.gz" for name in ["market_daily", "factors", "universe", "factor_availability"]]
    inputs += [baseline_path, root / "data" / "data_dictionary.md", root / "版本说明.md"]
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "data_package": "Alpha Forensics v1.0", "data_root": str(root.resolve()),
        "python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__,
        "inputs_sha256": {str(p.relative_to(root)): sha256(p) for p in inputs},
        "code_sha256": {p.name: sha256(p) for p in [Path(__file__), PROJECT / "test_factor_timing.py"]},
        "signal_time": "t after close; available_on <= t; latest available reference date",
        "label": "open(t+6 business days) / open(t+1 business day) - 1; actual endpoints only",
        "membership": "is_listed and is_member on signal date; align before filtering",
        "missing_values": "retain unreleased/missing snapshots; no backward fill or price fill",
        "comparison": "same date/asset pairs per factor; rerank inside the common cross-section",
        "scope": "Historical factor timing diagnostic; no OOS model validation or trading performance",
        "eligible_rows": len(pit), "eligible_dates": int(pit.date.nunique()),
        "eligible_assets": int(pit.asset_id.nunique()), "availability_audit": audit["availability"],
        "baseline_ic_reconciliation": audit["reference_scan"],
        "baseline_raw_ic_reproduced": True, "other_18_factors_unchanged": True,
        "outputs_sha256": {p.name: sha256(p) for p in [
            out / "daily_rank_ic.csv", out / "timing_summary.csv", out / "pit_research_panel.csv.gz"
        ]},
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, default=PROJECT / "outputs" / "factor_timing")
    args = parser.parse_args()
    run(args.root, args.out)
