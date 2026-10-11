"""Run preserved historical models and temporal OOS signals through one ledger.

Old model predictions are extended to the full signal-date member universe, so
the ledger never filters candidates using future five-day label availability.
The historical models remain non-OOS controls, even with correct accounting.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split

from factor_timing import DEFAULT_DATA, FACTORS, KEYS, PROJECT, sha256, validate_keys
from retrain_pit_model import PARAMETERS as ORIGINAL_PARAMETERS
from temporal_model import load_signal_panel
from portfolio_ledger import simulate


START_SIGNAL = pd.Timestamp("2023-01-02")
END_DATE = pd.Timestamp("2024-12-31")
PROTOCOL = {
    "start_signal": str(START_SIGNAL.date()), "end_date": str(END_DATE.date()),
    "last_entry_signal": str((END_DATE - pd.offsets.BDay(6)).date()),
    "strategy": "Long-only prediction >= same-day 90th percentile; equal-weight within selected group",
    "benchmark": "All signal-date listed members, equal-weight; same five-sleeve accounting and execution",
    "capital": "Initial NAV 1; five initially equal independent cash sleeves; no borrowing or shorting; fractional shares",
    "orders": "Signal-close names and fee-inclusive sleeve-value budget; next open sell before buy; cash shortfall scales the budget down",
    "holding": "Each new basket holds five business-day open-to-open intervals; blocked exits retry daily; do not add a new lot of an asset still held in that sleeve",
    "costs": "Previous-business-day fee_bps_one_way + half spread_bps + impact_bps_proxy on each actual buy/sell",
    "constraints": "Previous-business-day can_buy/can_sell are execution proxies; same-day final flags only audit hypothetical fills",
    "unfilled_buys": "Keep allocated cash; no replacement or redistribution among other names",
    "missing_quote": "First missing quote on an existing position: zero recovery write-off, separately identified valuation assumption",
    "terminal": "Predefined final entry date permits normal five-day exits; blocked terminal positions marked at actual final close, not fictitiously liquidated",
    "capacity": "Fixed supplied standard-order impact proxy; no inference of scalable capacity; no volume-based opening filter",
    "netting": "Sleeves accounted separately; cross-sleeve opposite trades are not netted, including their costs",
    "annualization": "252 synthetic trading days; zero cash interest and zero risk-free rate",
    "comparison": "Identical signal-date members and dates across predictors; old random-split models remain contaminated historical controls",
}
RUNS = [
    ("original_random_net", "original_random", "top_decile", "base", "lagged"),
    ("pit_random_net", "pit_random", "top_decile", "base", "lagged"),
    ("temporal_oos_gross", "temporal_oos", "top_decile", "zero", "none"),
    ("temporal_oos_costs", "temporal_oos", "top_decile", "base", "none"),
    ("temporal_oos_net", "temporal_oos", "top_decile", "base", "lagged"),
    ("benchmark_gross", "temporal_oos", "equal_weight", "zero", "none"),
    ("benchmark_costs", "temporal_oos", "equal_weight", "base", "none"),
    ("benchmark_net", "temporal_oos", "equal_weight", "base", "lagged"),
]


def verify_hash(path: Path, expected: str) -> None:
    if sha256(path) != expected:
        raise ValueError(f"Artifact hash changed: {path}")


def load_variants(root: Path, old_root: Path, temporal: Path, out: Path):
    """Restore the exact raw control, then extend both old models' coverage."""
    tm = json.loads((temporal / "run_manifest.json").read_text())
    verify_hash(temporal / "predictions.csv.gz", tm["outputs_sha256"]["predictions.csv.gz"])
    for path, digest in tm["inputs_sha256"].items():
        verify_hash(Path(path), digest)
    saved_meta = json.loads((old_root / "pit" / "baseline_run_manifest.json").read_text())
    verify_hash(old_root / "pit" / "model.joblib", saved_meta["outputs_sha256"]["pit/model.joblib"])
    complete = load_signal_panel(root)
    signal_panel = complete.loc[complete.date.between(START_SIGNAL, END_DATE)].copy()
    new = pd.read_csv(temporal / "predictions.csv.gz", parse_dates=["date"])
    new = new.loc[new.date.between(START_SIGNAL, END_DATE)].sort_values(KEYS).reset_index(drop=True)
    signal_panel = signal_panel.sort_values(KEYS).reset_index(drop=True)
    pd.testing.assert_frame_equal(new[KEYS], signal_panel[KEYS])
    validate_keys(new, "temporal OOS signals")
    if not np.isfinite(new.prediction).all():
        raise ValueError("Temporal predictions are not finite")

    source = root / "baseline" / "baseline_code" / "build_factor.py"
    spec = importlib.util.spec_from_file_location("original_handoff_loader", source)
    handoff = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(handoff)
    original_frame = handoff.load_research_frame(root / "data")
    sample = original_frame.sample(n=min(110_000, len(original_frame)), random_state=1021)
    train, _ = train_test_split(np.arange(len(sample)), test_size=.2, random_state=1021, shuffle=True)
    raw_model = HistGradientBoostingRegressor(**ORIGINAL_PARAMETERS)
    raw_model.fit(
        sample[FACTORS].to_numpy(dtype=np.float32)[train],
        sample.forward_5d.to_numpy(dtype=np.float32)[train],
    )
    restored = raw_model.predict(original_frame[FACTORS].to_numpy(dtype=np.float32)).astype(np.float32)
    old_prediction = pd.read_csv(old_root / "original" / "baseline_predictions.csv.gz")
    pd.testing.assert_frame_equal(original_frame[KEYS].reset_index(drop=True), old_prediction[KEYS])
    np.testing.assert_array_equal(restored, old_prediction.prediction.to_numpy(dtype=np.float32))
    (out / "models").mkdir()
    joblib.dump(raw_model, out / "models" / "restored_original.joblib")

    raw_factors = pd.read_csv(root / "data" / "factors.csv.gz", parse_dates=["date"])
    raw_inputs = signal_panel[KEYS].merge(raw_factors, on=KEYS, validate="one_to_one")
    pd.testing.assert_frame_equal(raw_inputs[KEYS], signal_panel[KEYS])
    pit_model = joblib.load(old_root / "pit" / "model.joblib")
    variants = {
        "original_random": signal_panel[KEYS].assign(
            prediction=raw_model.predict(raw_inputs[FACTORS].to_numpy(dtype=np.float32))
        ),
        "pit_random": signal_panel[KEYS].assign(
            prediction=pit_model.predict(signal_panel[FACTORS].to_numpy(dtype=np.float32))
        ),
        "temporal_oos": new[KEYS + ["prediction"]],
    }
    coverage = []
    for name, frame in variants.items():
        pd.testing.assert_frame_equal(frame[KEYS], new[KEYS])
        if not np.isfinite(frame.prediction).all():
            raise ValueError(f"{name}: non-finite full-universe predictions")
        frame.to_csv(
            out / f"{name}_complete_signals.csv.gz", index=False,
            compression={"method": "gzip", "compresslevel": 1, "mtime": 0},
        )
        coverage.append({
            "variant": name, "rows": len(frame), "dates": int(frame.date.nunique()),
            "rows_without_future_label": int(signal_panel.forward_5d.isna().sum()),
            "is_temporal_oos": name == "temporal_oos",
        })
    return variants, coverage


def make_figure(dailies: dict[str, pd.DataFrame], out: Path) -> None:
    selected = [
        ("temporal_oos_gross", "OOS: no costs / no constraints"),
        ("temporal_oos_costs", "OOS: costs"),
        ("temporal_oos_net", "OOS: costs + lagged constraints"),
        ("benchmark_net", "Equal-weight: costs + lagged constraints"),
    ]
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, layout="constrained")
    for key, label in selected:
        df = dailies[key]
        nav = df.nav.to_numpy()
        drawdown = nav / np.maximum.accumulate(nav) - 1
        line, = axes[0].plot(df.date, nav, label=label, linewidth=1.5)
        axes[1].plot(df.date, drawdown * 100, color=line.get_color(), linewidth=1.2)
    axes[0].set_ylabel("Close NAV (initial = 1)")
    axes[0].set_title("Synthetic long-only simulation | 2023-2024 | five staggered sleeves")
    axes[0].legend(loc="best", fontsize=8)
    axes[1].set_ylabel("Drawdown (%)")
    axes[1].set_xlabel("Synthetic trading date")
    for axis in axes:
        axis.grid(alpha=.2)
    fig.savefig(out / "oos_nav_drawdown.png", dpi=160)
    plt.close(fig)


def run(root: Path, old_root: Path, temporal: Path, out: Path) -> None:
    started = datetime.now(timezone.utc).isoformat()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite an existing comparison: {out}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "protocol.json").write_text(json.dumps(PROTOCOL, ensure_ascii=False, indent=2), encoding="utf-8")
    inputs = [root / "data" / f"{name}.csv.gz" for name in [
        "market_daily", "factors", "universe", "factor_availability", "trading_constraints"
    ]]
    inputs += [root / "data" / "data_dictionary.md", temporal / "predictions.csv.gz", temporal / "run_manifest.json"]
    inputs += [old_root / "original" / "baseline_predictions.csv.gz", old_root / "pit" / "model.joblib"]
    input_hashes = {str(p.resolve()): sha256(p) for p in inputs}
    print("Restoring preserved models and extending complete signal universes", flush=True)
    variants, coverage = load_variants(root, old_root, temporal, out)
    pd.DataFrame(coverage).to_csv(out / "signal_coverage.csv", index=False)
    market = pd.read_csv(root / "data" / "market_daily.csv.gz", parse_dates=["date"])
    constraints = pd.read_csv(root / "data" / "trading_constraints.csv.gz", parse_dates=["date"])
    dailies, summaries = {}, []
    compression = {"method": "gzip", "compresslevel": 1, "mtime": 0}
    for name, predictor, selection, cost, constraint in RUNS:
        print(f"Ledger {name}: {selection}, cost={cost}, constraints={constraint}", flush=True)
        result = simulate(
            variants[predictor], market, constraints,
            start_signal=START_SIGNAL, end_date=END_DATE,
            selection=selection, cost_mode=cost, constraint_mode=constraint,
        )
        expected_dates = pd.Series(pd.bdate_range(START_SIGNAL, END_DATE), name="date")
        pd.testing.assert_series_equal(result["daily"].date.reset_index(drop=True), expected_dates)
        run_out = out / name
        run_out.mkdir()
        result["daily"].to_csv(run_out / "daily.csv", index=False)
        result["orders"].to_csv(run_out / "orders.csv.gz", index=False, compression=compression)
        result["positions"].to_csv(run_out / "positions.csv.gz", index=False, compression=compression)
        (run_out / "summary.json").write_text(json.dumps(result["summary"], ensure_ascii=False, indent=2), encoding="utf-8")
        dailies[name] = result["daily"]
        summaries.append({
            "run": name, "predictor": predictor, "selection": selection,
            "cost_mode": cost, "constraint_mode": constraint,
            "is_temporal_oos_strategy": predictor == "temporal_oos" and selection == "top_decile",
            "is_benchmark": selection == "equal_weight", **result["summary"],
        })
    comparison = pd.DataFrame(summaries)
    benchmarks = comparison.loc[comparison.is_benchmark].set_index(["cost_mode", "constraint_mode"], verify_integrity=True)
    comparison["benchmark_run"] = [benchmarks.loc[(r.cost_mode, r.constraint_mode), "run"] for r in comparison.itertuples()]
    comparison["relative_terminal_wealth_vs_matching_benchmark"] = [
        (1 + r.total_return) / (1 + benchmarks.loc[(r.cost_mode, r.constraint_mode), "total_return"]) - 1
        for r in comparison.itertuples()
    ]
    comparison.to_csv(out / "portfolio_comparison.csv", index=False)
    make_figure(dailies, out)
    code = [Path(__file__), PROJECT / "portfolio_ledger.py", PROJECT / "test_portfolio_ledger.py",
            PROJECT / "temporal_model.py", PROJECT / "factor_timing.py", PROJECT / "retrain_pit_model.py"]
    manifest = {
        "started_at_utc": started, "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": sys.orig_argv, "protocol": PROTOCOL, "runs": RUNS,
        "output_root": str(out.resolve()),
        "environment": {
            "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__, "joblib": joblib.__version__, "matplotlib": matplotlib.__version__,
            "threads": {key: os.environ.get(key) for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"]},
        },
        "original_control_reproduced_at_float32": True,
        "signal_universe_identical_across_predictors": True,
        "coverage": coverage,
        "limitations": [
            "Historical model OOS replay, not a never-observed new-data blind test",
            "Old original/PIT random models remain historical controls, not valid time-OOS strategies",
            "Lagged trading states are proxies; conflicting same-day final flags are separately audited",
            "Zero recovery on disappeared quotes is an explicit conservative valuation assumption",
            "Standard-order linear impact costs do not establish execution capacity",
        ],
        "inputs_sha256": input_hashes,
        "code_sha256": {str(p.resolve()): sha256(p) for p in code},
        "outputs_sha256": {str(p.relative_to(out)): sha256(p) for p in sorted(out.rglob("*")) if p.is_file()},
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(comparison.to_string(index=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--old-root", type=Path, default=PROJECT / "outputs" / "model_rerun")
    parser.add_argument("--temporal", type=Path, default=PROJECT / "outputs" / "temporal_oos")
    parser.add_argument("--out", type=Path, default=PROJECT / "outputs" / "portfolio_upgrade")
    args = parser.parse_args()
    run(args.root, args.old_root, args.temporal, args.out)
