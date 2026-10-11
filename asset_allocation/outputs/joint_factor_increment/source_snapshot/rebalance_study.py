"""Fixed-policy historical rebalance comparison using archived annual signals.

No forecast is refitted, no parameter is selected, and no previous artifact is
overwritten. FULL is the previously completed five-sleeve liquidation reference.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing
import os
from pathlib import Path
import platform
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd

from compare_models import PROJECT, jsonable, moving_block_interval, object_hash, sha256, utc_now, write_json
from factor_timing import DEFAULT_DATA
from portfolio_model_comparison import period_statistics
from rebalance_ledger import POLICIES, simulate_rebalance


PREDICTORS = ("Ridge_ALL20", "HGB_ALL20", "Ridge_CONTROLS", "HGB_CONTROLS", "EqualWeight")
EW_POLICIES = ("NET", "WEIGHT", "EMA50", "DAILY_NET")
EW_ALIASES = {"RANK": "NET", "BOTH": "WEIGHT", "TAPER": "NET", "SCORE": "NET"}
COSTS = (0, 1, 2)
START, END = pd.Timestamp("2023-01-02"), pd.Timestamp("2024-12-31")
COMPRESSION = {"method": "gzip", "compresslevel": 1, "mtime": 0}
SCOPE = "Additional fixed-policy research on previously inspected synthetic history; no independent new sample or selection of a winning policy."
_WORKER = {}


def run_name(policy, predictor, cost):
    return f"{policy}__{predictor}__cost{cost}"


def run_specs():
    return [dict(run=run_name(policy, predictor, cost), policy=policy,
                 predictor=predictor, cost_multiplier=cost)
            for policy in POLICIES for predictor in PREDICTORS
            if predictor != "EqualWeight" or policy in EW_POLICIES
            for cost in COSTS]


def resolve_policy(policy, predictor):
    return EW_ALIASES.get(policy, policy) if predictor == "EqualWeight" else policy


def build_protocol(root, source):
    manifest_path = source / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    previous = json.loads((source / "protocol.json").read_text())
    if manifest["status"] != "completed" or manifest["protocol_sha256"] != object_hash(previous):
        raise ValueError("Source study is not complete under its frozen protocol")
    paths = [manifest_path, source / "verification.json", source / "statistical_verification.json"]
    for filename in ["verification.json", "statistical_verification.json"]:
        if json.loads((source / filename).read_text())["status"] != "passed":
            raise ValueError(f"Source verification incomplete: {filename}")
    relatives = ["protocol.json", "period_metrics.csv", "summary.csv"]
    relatives += [f"predictions/{name}.csv.gz" for name in PREDICTORS[:-1]]
    relatives += [f"runs/{name}__cost{cost}/daily.csv" for name in PREDICTORS for cost in COSTS]
    for relative in relatives:
        path = source / relative
        if sha256(path) != manifest["outputs_sha256"][relative]:
            raise ValueError(f"Source artifact changed: {relative}")
        paths.append(path)
    for filename in ["market_daily.csv.gz", "trading_constraints.csv.gz", "universe.csv.gz"]:
        path = root / "data" / filename
        hashes = {value for key, value in previous["inputs_sha256"].items() if key.endswith("/data/" + filename)}
        if len(hashes) != 1 or sha256(path) not in hashes:
            raise ValueError(f"Market/constraint/member data differs from source study: {filename}")
        paths.append(path)
    paths.append(root / "data" / "data_dictionary.md")
    code = [PROJECT / name for name in ["rebalance_study.py", "rebalance_ledger.py",
            "test_rebalance_ledger.py", "audit_rebalance_study.py", "portfolio_ledger.py",
            "portfolio_model_comparison.py", "compare_models.py", "factor_timing.py"]]
    return {
        "frozen_at_utc": utc_now(), "scope": SCOPE,
        "source": str(source.resolve()), "root": str(root.resolve()),
        "policies": list(POLICIES), "predictors": list(PREDICTORS), "cost_multipliers": list(COSTS),
        "runs": run_specs(), "aliases": [{"predictor": "EqualWeight", "policy": key, "actual_policy": value,
                                           "reason": "Equal-weight benchmark has no forecast rank or score"}
                                          for key, value in EW_ALIASES.items()],
        "primary_period": "2024", "primary_cost_multiplier": 1,
        "descriptive_periods": ["2023", "FULL"],
        "fixed_predictions": "Reuse complete annual 2023/2024 member predictions; no refit, score adjustment, factor selection, or model selection",
        "start_signal": str(START.date()), "end_date": str(END.date()),
        "last_entry_signal": str((END - pd.offsets.BDay(6)).date()),
        "netting": "Within each cash sleeve only; keep eligible holdings and trade target minus actual shares; no transfers or cross-sleeve netting",
        "timing": "Signal close freezes names, weights and close-NAV budget; next open uses execution budget min(signal close NAV, actual open NAV)",
        "quantities": "Normal target shares = execution budget * effective weight / open; within-band quantities stay frozen at signal-close actual shares",
        "sell_intentions": "A new plan supersedes old targets before any sell retry; blocked reductions keep a share target and retry with next day's prior-day proxies",
        "buy_budget": "Sell first; scale all intended fee-inclusive buys together to available sleeve cash, retaining blocked/missing-quote demand slots; no redistribution; failed buys expire",
        "permissions_costs": "Prior business-day permission and fee + half spread + standard-order impact; current-day flags audit only; each cost multiplier follows its own holdings",
        "missing_quote": "Existing position written off at zero recovery on first missing quote; intended missing-quote buy budget remains cash",
        "terminal": "Five-sleeve methods issue zero targets at each sleeve's first scheduled review after last entry; DAILY_NET freezes after last entry and issues zero target on end-1 business day; blocked final sells remain marked holdings",
        "rules": {
            "NET": "Five initially equal cash sleeves; index%5 sleeve reviews; score>=q90, all boundary ties, equal weights",
            "RANK": "Entry q90, retention q80 based on actual held shares; K=ceil(0.1*N); retain eligible old holdings first, fill by descending whole score-tie groups; no new entry if retained count>=K",
            "WEIGHT": "NET membership; signal-close relative +/-10% target band; hold actual shares in band, nearest boundary out of band, new positions at target, zero target exits override band",
            "BOTH": "RANK membership plus WEIGHT sizing, same fixed parameters",
            "EMA50": "Each sleeve mixes 0.5 fresh NET target with 0.5 its prior smoothed target; starts in cash; ineligible assets zeroed at review; no rescaling to full exposure; terminal zero overrides smoothing",
            "TAPER": "NET selected set; weights proportional to M+1-average descending rank, normalized to one, equal scores equal weights",
            "SCORE": "NET selected set; positive part of score minus all-member mean, proportional with cap 2/M and capped surplus redistributed; insufficient positive capacity leaves cash; flat cross-section cash",
            "DAILY_NET": "One full account, new NET target every day; changes both review frequency and initial capital deployment relative to five staggered sleeves",
        },
        "method_origin": "Adapted ideas from the user's multifactor RebalancePolicy, target-weight EMA, tapered and centered-score portfolios; this study uses the supplied daily long-only cash market, not the prior perpetual account or prior performance evidence",
        "metrics": "Continuous close NAV; 252-day annualization; prior year-end opening NAV included in annual return/drawdown; per-asset holdings aggregated across sleeves for concentration and breadth",
        "primary_uncertainty": "2024 cost1 only: paired daily arithmetic return and log relative growth; moving 20-business-day blocks, 2000 draws, seed1021; unadjusted reference intervals, no multiple-search winner claim",
        "inputs_sha256": {str(p.resolve()): sha256(p) for p in paths},
        "code_sha256": {str(p.resolve()): sha256(p) for p in code},
    }


def _initialize(root, source, out):
    root, source, out = Path(root), Path(source), Path(out)
    market = pd.read_csv(root / "data" / "market_daily.csv.gz", usecols=["date", "asset_id", "open", "close"], parse_dates=["date"])
    constraints = pd.read_csv(root / "data" / "trading_constraints.csv.gz", parse_dates=["date"])
    predictions = {name: pd.read_csv(source / "predictions" / f"{name}.csv.gz",
                                    usecols=["date", "asset_id", "prediction"], parse_dates=["date"])
                   for name in PREDICTORS[:-1]}
    predictions["EqualWeight"] = predictions["Ridge_ALL20"][["date", "asset_id"]].assign(prediction=0.)
    scenarios = {}
    for cost in COSTS:
        frame = constraints.copy()
        columns = ["fee_bps_one_way", "spread_bps", "impact_bps_proxy"]
        frame[columns] = frame[columns] * cost
        scenarios[cost] = frame
    _WORKER.update(market=market, scenarios=scenarios, predictions=predictions, out=out)


def exposure_diagnostics(daily, positions):
    dates = pd.DatetimeIndex(daily.date, name="date")
    aggregate = positions.groupby(["date", "asset_id"], sort=False).market_value.sum()
    values = aggregate.unstack("asset_id").reindex(dates).fillna(0.)
    weights = values.div(daily.set_index("date").nav, axis=0)
    return pd.DataFrame({"date": dates, "asset_count": values.gt(0).sum(axis=1).to_numpy(),
                         "hhi": weights.pow(2).sum(axis=1).to_numpy(),
                         "max_asset_weight": weights.max(axis=1).to_numpy() if len(weights.columns) else np.zeros(len(dates)),
                         "gross_exposure": (1 - daily.cash / daily.nav).to_numpy()})


def _execute(spec):
    result = simulate_rebalance(_WORKER["predictions"][spec["predictor"]], _WORKER["market"],
                                _WORKER["scenarios"][spec["cost_multiplier"]], start_signal=START,
                                end_date=END, policy=spec["policy"],
                                selection="equal_weight" if spec["predictor"] == "EqualWeight" else "top_decile")
    out = _WORKER["out"] / "runs" / spec["run"]
    out.mkdir()
    exposures = exposure_diagnostics(result["daily"], result["positions"])
    exposures.to_csv(out / "exposure_daily.csv", index=False)
    result["daily"].to_csv(out / "daily.csv", index=False)
    result["reviews"].to_csv(out / "reviews.csv", index=False)
    for kind in ["orders", "positions", "plans"]:
        result[kind].to_csv(out / f"{kind}.csv.gz", index=False, compression=COMPRESSION)
    records = []
    for period in ["FULL", "2023", "2024"]:
        stats = period_statistics(result["daily"], result["orders"], period)
        frame = exposures if period == "FULL" else exposures.loc[exposures.date.dt.year == int(period)]
        stats.update(mean_asset_count=float(frame.asset_count.mean()), mean_hhi=float(frame.hhi.mean()),
                     max_asset_weight=float(frame.max_asset_weight.max()),
                     mean_gross_exposure=float(frame.gross_exposure.mean()),
                     turnover_per_mean_exposure=float(stats["mean_daily_turnover"] / frame.gross_exposure.mean())
                     if frame.gross_exposure.mean() > 0 else None)
        records.append({**spec, **stats})
    summary = {**result["summary"], **records[0]}
    write_json(out / "summary.json", jsonable(summary))
    return summary, records


def add_references_and_aliases(periods, source):
    source_periods = pd.read_csv(source / "period_metrics.csv", dtype={"period": str})
    source_periods["actual_run"] = [run_name("FULL", r.predictor, r.cost_multiplier) for r in source_periods.itertuples()]
    source_periods["run"] = source_periods.actual_run
    source_periods["policy"] = "FULL"
    source_periods["source_kind"] = "archived_full"
    source_periods["is_alias"] = False
    current = periods.assign(actual_run=periods.run, source_kind="new", is_alias=False)
    alias_parts = []
    for requested, actual in EW_ALIASES.items():
        subset = current.loc[current.predictor.eq("EqualWeight") & current.policy.eq(actual)].copy()
        subset["policy"] = requested
        subset["run"] = [run_name(requested, r.predictor, r.cost_multiplier) for r in subset.itertuples()]
        subset["source_kind"] = "alias"
        subset["is_alias"] = True
        alias_parts.append(subset)
    return pd.concat([source_periods, current, *alias_parts], ignore_index=True)


def paired_statistics(out, source, specs):
    daily = {}
    for predictor in PREDICTORS:
        daily[("FULL", predictor)] = pd.read_csv(source / "runs" / f"{predictor}__cost1" / "daily.csv",
                                                 parse_dates=["date"]).set_index("date").daily_return
    for spec in specs:
        if spec["cost_multiplier"] == 1:
            daily[(spec["policy"], spec["predictor"])] = pd.read_csv(
                out / "runs" / spec["run"] / "daily.csv", parse_dates=["date"]).set_index("date").daily_return
    comparisons = []
    for spec in specs:
        if spec["cost_multiplier"] != 1:
            continue
        reference = "FULL" if spec["policy"] == "NET" else "NET"
        comparisons.append(("method_effect", spec["policy"], spec["predictor"], reference, spec["predictor"]))
    for policy in POLICIES:
        for candidate, reference in [("Ridge_ALL20", "HGB_ALL20"), ("Ridge_ALL20", "Ridge_CONTROLS"),
                                     ("HGB_ALL20", "HGB_CONTROLS"), ("Ridge_ALL20", "EqualWeight"),
                                     ("HGB_ALL20", "EqualWeight")]:
            comparisons.append(("model_effect", policy, candidate, resolve_policy(policy, reference), reference))
    records = []
    for kind, policy, predictor, ref_policy, ref_predictor in comparisons:
        left, right = daily[(policy, predictor)], daily[(ref_policy, ref_predictor)]
        if not left.index.equals(right.index):
            raise AssertionError("Paired accounts have different dates")
        for metric, differences in [("daily_return_difference", left - right),
                                    ("daily_log_relative_growth", np.log1p(left) - np.log1p(right))]:
            records.append({"comparison_type": kind, "policy": policy, "predictor": predictor,
                            "reference_policy": ref_policy, "reference_predictor": ref_predictor,
                            "period": "2024", "cost_multiplier": 1, "metric": metric,
                            **moving_block_interval(differences.loc[differences.index.year == 2024])})
    return pd.DataFrame(records)


def make_figures(comparison, out):
    policies = ["FULL", *POLICIES]
    table = comparison.loc[comparison.period.eq("2024") & comparison.cost_multiplier.eq(1)]
    grid = table.pivot(index="policy", columns="predictor", values="total_return").reindex(index=policies, columns=PREDICTORS)
    extent = max(float(grid.abs().to_numpy().max()), 1e-8)
    fig, ax = plt.subplots(figsize=(10, 7), layout="constrained")
    plotted = ax.imshow(grid * 100, cmap="RdYlGn", norm=TwoSlopeNorm(vmin=-extent*100, vcenter=0, vmax=extent*100), aspect="auto")
    for i, policy in enumerate(policies):
        for j, predictor in enumerate(PREDICTORS):
            alias = predictor == "EqualWeight" and policy in EW_ALIASES
            ax.text(j, i, f"{grid.iloc[i,j]*100:.2f}%" + (" *" if alias else ""), ha="center", va="center", fontsize=10)
    ax.set_xticks(range(5), ["Ridge\n20 factors", "HGB\n20 factors", "Ridge\nindustry + size", "HGB\nindustry + size", "Member\nequal weight"])
    ax.set_yticks(range(len(policies)), policies)
    ax.set_title("2024 net returns | supplied costs + lagged permissions\nFixed rules on previously inspected synthetic history")
    ax.set_xlabel("* Equal-weight aliases: RANK / TAPER / SCORE use NET; BOTH uses WEIGHT")
    fig.colorbar(plotted, ax=ax, label="2024 return (%)", shrink=.8)
    fig.savefig(out / "returns_2024.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True, layout="constrained")
    for ax, predictor in zip(axes, ["Ridge_ALL20", "HGB_ALL20"]):
        for policy in policies:
            part = comparison.loc[comparison.period.eq("2024") & comparison.policy.eq(policy) & comparison.predictor.eq(predictor)].sort_values("cost_multiplier")
            ax.plot(part.cost_multiplier, part.total_return*100, marker="o", label=policy, lw=1.5)
        ax.set_title(predictor.replace("_ALL20", ": 20 factors"))
        ax.set_xticks([0,1,2], ["0 x costs", "1 x costs", "2 x costs"])
        ax.axhline(0, color="gray", lw=.8)
        ax.grid(alpha=.2)
    axes[0].set_ylabel("2024 return (%)")
    axes[1].legend(fontsize=8, ncols=2)
    fig.suptitle("Cost sensitivity | policy candidates fixed before this study")
    fig.savefig(out / "cost_sensitivity_2024.png", dpi=160)
    plt.close(fig)


def run(root, source, out, workers):
    started = utc_now()
    if out.exists():
        raise FileExistsError(f"Refusing to replace study: {out}")
    if workers not in (1, 2):
        raise ValueError("Use one or two ledger workers for the saved research workload")
    protocol = build_protocol(root, source)
    out.mkdir(parents=True)
    (out / "runs").mkdir()
    write_json(out / "protocol.json", protocol)
    summaries, periods = [], []
    specs = protocol["runs"]
    print(f"Frozen {len(specs)} new accounts; reusing 15 FULL references; {workers} workers", flush=True)
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                             initializer=_initialize, initargs=(str(root), str(source), str(out))) as pool:
        futures = {pool.submit(_execute, spec): spec for spec in specs}
        for future in as_completed(futures):
            summary, records = future.result()
            summaries.append(summary)
            periods.extend(records)
            print(f"Completed {len(summaries)}/{len(specs)} {summary['run']}: NAV={summary['final_nav']:.6f}", flush=True)
    summary_frame = pd.DataFrame(summaries).sort_values(["policy", "predictor", "cost_multiplier"])
    period_frame = pd.DataFrame(periods).sort_values(["policy", "predictor", "cost_multiplier", "period"])
    summary_frame.to_csv(out / "summary.csv", index=False)
    period_frame.to_csv(out / "period_metrics.csv", index=False)
    comparison = add_references_and_aliases(period_frame, source)
    comparison.to_csv(out / "comparison.csv", index=False)
    paired_statistics(out, source, specs).to_csv(out / "paired_comparisons.csv", index=False)
    make_figures(comparison, out)
    for field in ["inputs_sha256", "code_sha256"]:
        for path, expected in protocol[field].items():
            if sha256(Path(path)) != expected:
                raise AssertionError(f"Input or code changed during study: {path}")
    write_json(out / "run_manifest.json", {
        "status": "completed", "started_at_utc": started, "finished_at_utc": utc_now(),
        "command": sys.orig_argv, "scope": SCOPE, "protocol_sha256": object_hash(protocol),
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                        "workers": workers, "threads": {key: os.environ.get(key) for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]}},
        "new_account_runs": len(specs), "archived_full_accounts": 15, "model_fits": 0,
        "inputs_sha256": protocol["inputs_sha256"], "code_sha256": protocol["code_sha256"],
        "outputs_sha256": {str(path.relative_to(out)): sha256(path) for path in sorted(out.rglob("*")) if path.is_file()},
    })
    print("Fixed-policy study completed", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--source", type=Path, default=PROJECT / "outputs" / "portfolio_model_comparison")
    parser.add_argument("--out", type=Path, default=PROJECT / "outputs" / "rebalance_study")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    run(args.root, args.source, args.out, args.workers)
