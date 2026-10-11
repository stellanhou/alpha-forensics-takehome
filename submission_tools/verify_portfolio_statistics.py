"""Reproduce portfolio-comparison statistics without importing research code.

Run after audit_portfolio_model_comparison.py. This checks saved daily NAVs and
aggregate statistics; the prerequisite verification.json supplies the separate
order/account audit. Existing output files are never replaced.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


PREDICTORS = ["Ridge_ALL20", "HGB_ALL20", "Ridge_CONTROLS", "HGB_CONTROLS", "EqualWeight"]
COSTS = [0, 1, 2]
PERIODS = ["FULL", "2023", "2024"]
METRICS = ["daily_return_difference", "daily_log_relative_growth"]
PAIRS = [("Ridge_ALL20", "HGB_ALL20"), ("Ridge_ALL20", "Ridge_CONTROLS"),
         ("HGB_ALL20", "HGB_CONTROLS")] + [(arm, "EqualWeight") for arm in PREDICTORS[:-1]]
ABSOLUTE_TOLERANCE = 1e-12


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path: Path, hashes: dict) -> dict:
    content = path.read_bytes()
    hashes[str(path.resolve())] = hashlib.sha256(content).hexdigest()
    return json.loads(content)


def verify_hashes(hashes: dict) -> None:
    for name, expected in hashes.items():
        require(sha256(Path(name)) == expected, f"Changed input: {name}")


def add_hashes(hashes: dict, mapping: dict, base: Path) -> None:
    for name, digest in mapping.items():
        resolved = str((base / name).resolve())
        if resolved in hashes:
            require(hashes[resolved] == digest, f"Conflicting recorded hashes: {resolved}")
        hashes[resolved] = digest


def bootstrap_parameters(protocol: dict) -> tuple[int, int, int]:
    """Read literal constants from the hash-checked source, without executing it."""
    sources = [Path(name) for name in protocol["code_sha256"] if Path(name).name == "compare_models.py"]
    require(len(sources) == 1, "Expected exactly one frozen compare_models.py source")
    constants = {}
    for node in ast.parse(sources[0].read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in {"SEED", "BLOCK_DAYS", "BOOTSTRAP_DRAWS"}:
                require(name not in constants, f"Duplicate source constant: {name}")
                constants[name] = ast.literal_eval(node.value)
    require(constants == {"SEED": 1021, "BLOCK_DAYS": 20, "BOOTSTRAP_DRAWS": 2000},
            "Frozen bootstrap parameters differ from this verification contract")
    require(protocol["uncertainty"] == (
        "Paired daily arithmetic return difference and log relative growth; moving 20-day blocks, "
        "2000 draws; reference intervals, no multiplicity correction"), "Unexpected uncertainty protocol")
    return constants["SEED"], constants["BLOCK_DAYS"], constants["BOOTSTRAP_DRAWS"]


def independent_interval(values: np.ndarray, seed: int, block_days: int, draws: int) -> dict:
    """Noncircular contiguous blocks; missing calendar observations stay in place."""
    require(len(values) >= block_days, "Insufficient calendar length for the fixed bootstrap")
    require(not np.isinf(values).any() and np.isfinite(values).any(), "Invalid bootstrap values")
    blocks_per_draw = (len(values) + block_days - 1) // block_days
    starts = np.random.default_rng(seed).integers(
        len(values) - block_days + 1, size=(draws, blocks_per_draw))
    offsets = np.arange(block_days)
    samples = values[(starts[:, :, None] + offsets).reshape(draws, -1)[:, :len(values)]]
    counts = np.isfinite(samples).sum(axis=1)
    require((counts > 0).all(), "A bootstrap draw contains no valid observations")
    means = np.nansum(samples, axis=1) / counts
    low, high = np.percentile(means, [2.5, 97.5])
    return {"n_dates": int(np.isfinite(values).sum()), "mean": float(np.nanmean(values)),
            "ci_low": float(low), "ci_high": float(high)}


def compare_fields(saved, expected: dict, errors: dict, context: str) -> None:
    for field, wanted in expected.items():
        actual = float(saved[field])
        if pd.isna(wanted):
            require(pd.isna(actual), f"{context}: expected missing {field}")
            continue
        require(np.isfinite(actual), f"{context}: nonfinite {field}")
        error = abs(actual - wanted)
        require(error <= ABSOLUTE_TOLERANCE, f"{context}: {field} differs by {error:.17g}")
        errors[field] = max(errors.get(field, 0.0), float(error))


def period_values(daily: pd.DataFrame, period: str) -> dict:
    selected = daily if period == "FULL" else daily.loc[daily.index.year == int(period)]
    require(not selected.empty, f"No daily records for {period}")
    previous = daily.loc[daily.index < selected.index[0], "nav"]
    initial = float(previous.iloc[-1]) if len(previous) else 1.0
    final = float(selected.nav.iloc[-1])
    returns = selected.reconstructed_return.dropna()
    require(len(returns) >= 2, f"Insufficient observed daily returns for {period}")
    volatility = float(returns.std(ddof=1))
    path = np.concatenate(([initial], selected.nav.to_numpy()))
    result = {
        "initial_nav": initial, "final_nav": final, "n_return_days": len(returns),
        "total_return": final / initial - 1,
        "annualized_return": (final / initial) ** (252 / len(returns)) - 1,
        "annualized_volatility": volatility * np.sqrt(252),
        "sharpe": float(returns.mean() / volatility * np.sqrt(252)) if volatility else np.nan,
        "max_drawdown": float(np.min(path / np.maximum.accumulate(path) - 1)),
        "mean_daily_turnover": float(selected.loc[returns.index, "gross_turnover"].mean()),
        "mean_cash_fraction": float((selected.cash / selected.nav).mean()),
        "actual_flag_conflict_orders": int(selected.actual_flag_conflict_count.sum()),
    }
    for column in ["fee", "spread_cost", "impact_cost", "writeoff"]:
        result["total_" + column] = float(selected[column].sum())
    return result


def run(out: Path, destination: Path) -> None:
    require(not destination.exists(), f"Refusing to overwrite verification: {destination}")
    script_path = Path(__file__).resolve()
    script_hash = sha256(script_path)
    hashes = {}
    protocol = read_json(out / "protocol.json", hashes)
    manifest = read_json(out / "run_manifest.json", hashes)
    audit = read_json(out / "verification.json", hashes)
    require(manifest["status"] == "completed", "Portfolio study is incomplete")
    canonical_hash = hashlib.sha256(json.dumps(protocol, sort_keys=True, allow_nan=False).encode()).hexdigest()
    require(manifest["protocol_sha256"] == canonical_hash, "Manifest does not bind the saved protocol")
    require(audit["status"] == "passed", "Prerequisite portfolio account audit has not passed")
    require(audit["run_manifest_sha256"] == hashes[str((out / "run_manifest.json").resolve())],
            "Prerequisite account audit refers to a different run manifest")
    for field in ["inputs_sha256", "code_sha256"]:
        require(manifest[field] == protocol[field], f"Manifest and protocol disagree on {field}")
        add_hashes(hashes, protocol[field], out)
    add_hashes(hashes, manifest["outputs_sha256"], out)
    audit_sources = [digest for name, digest in protocol["code_sha256"].items()
                     if Path(name).name == "audit_portfolio_model_comparison.py"]
    require(audit_sources == [audit["audit_code_sha256"]], "Prerequisite audit code hash differs")
    verify_hashes(hashes)
    require(protocol["predictors"] == PREDICTORS and protocol["cost_multipliers"] == COSTS,
            "Unexpected predictor or cost scenarios")
    require(protocol["start_signal"] == "2023-01-02" and protocol["end_date"] == "2024-12-31",
            "Unexpected study date range")
    seed, block_days, draws = bootstrap_parameters(protocol)
    runs = [f"{predictor}__cost{cost}" for cost in COSTS for predictor in PREDICTORS]
    audited_runs = [record["run"] for record in audit["audited_runs"]]
    require(len(audited_runs) == len(runs) and set(audited_runs) == set(runs), "Account audit run set differs")
    require(manifest["ledger_runs"] == len(runs), "Manifest ledger run count differs")
    dates = pd.bdate_range(protocol["start_signal"], protocol["end_date"], name="date")
    daily_errors, dailies = {}, {}
    daily_columns = ["nav", "cash", "holdings_value", "gross_turnover", "fee", "spread_cost",
                     "impact_cost", "writeoff", "actual_flag_conflict_count"]
    required_outputs = {"paired_comparisons.csv", "period_metrics.csv"}
    required_outputs.update(f"runs/{name}/daily.csv" for name in runs)
    require(required_outputs <= manifest["outputs_sha256"].keys(), "Manifest omits statistics inputs")
    for name in runs:
        daily = pd.read_csv(out / "runs" / name / "daily.csv", parse_dates=["date"]).set_index("date", verify_integrity=True)
        require(daily.index.equals(dates), f"{name}: daily calendar differs")
        require(np.isfinite(daily[daily_columns].to_numpy()).all(), f"{name}: nonfinite daily values")
        require((daily.nav > 0).all(), f"{name}: nonpositive NAV")
        require(daily.daily_return.isna().to_list() == [True] + [False] * (len(daily) - 1),
                f"{name}: unexpected missing daily returns")
        reconstructed = daily.nav / daily.nav.shift(1) - 1
        return_error = float(np.max(np.abs(daily.daily_return.iloc[1:] - reconstructed.iloc[1:])))
        require(return_error <= ABSOLUTE_TOLERANCE, f"{name}: returns disagree with consecutive NAV ratios")
        nav_error = float(np.max(np.abs(daily.nav - daily.cash - daily.holdings_value)))
        require(nav_error <= ABSOLUTE_TOLERANCE, f"{name}: daily cash plus holdings does not equal NAV")
        daily_errors[name] = {"return_ratio_max_absolute_error": return_error,
                              "cash_plus_holdings_max_absolute_error": nav_error}
        dailies[name] = daily.assign(reconstructed_return=reconstructed)

    periods = pd.read_csv(out / "period_metrics.csv", dtype={"period": str}).set_index(["run", "period"], verify_integrity=True)
    require(set(periods.index) == {(name, period) for name in runs for period in PERIODS}, "Period table row set differs")
    recalculated = {(name, period): period_values(dailies[name], period) for name in runs for period in PERIODS}
    period_errors = {}
    for (name, period), expected in recalculated.items():
        predictor, cost_part = name.split("__cost")
        saved = periods.loc[(name, period)]
        require(saved.predictor == predictor and saved.cost_multiplier == int(cost_part), f"{name}: period identity differs")
        baseline = recalculated[(f"EqualWeight__cost{cost_part}", period)]["total_return"]
        expected["relative_terminal_wealth_vs_equalweight"] = (1 + expected["total_return"]) / (1 + baseline) - 1
        expected["relative_terminal_wealth_vs_matching_controls"] = np.nan
        if predictor.endswith("ALL20"):
            baseline = recalculated[(f"{predictor.replace('ALL20', 'CONTROLS')}__cost{cost_part}", period)]["total_return"]
            expected["relative_terminal_wealth_vs_matching_controls"] = (1 + expected["total_return"]) / (1 + baseline) - 1
        compare_fields(saved, expected, period_errors, f"{name}/{period}")

    keys = ["candidate", "reference", "cost_multiplier", "period", "metric"]
    paired = pd.read_csv(out / "paired_comparisons.csv", dtype={"period": str}).set_index(keys, verify_integrity=True)
    expected_keys = {(left, right, cost, period, metric)
                     for cost in COSTS for left, right in PAIRS for period in PERIODS for metric in METRICS}
    require(set(paired.index) == expected_keys, "Paired comparison row set differs")
    interval_errors = {}
    for key, saved in paired.iterrows():
        candidate, reference, cost, period, metric = key
        left = dailies[f"{candidate}__cost{cost}"].reconstructed_return
        right = dailies[f"{reference}__cost{cost}"].reconstructed_return
        difference = left - right if metric == "daily_return_difference" else np.log1p(left) - np.log1p(right)
        if period != "FULL":
            difference = difference.loc[difference.index.year == int(period)]
        expected = independent_interval(difference.to_numpy(), seed, block_days, draws)
        compare_fields(saved, expected, interval_errors, str(key))

    verify_hashes(hashes)
    require(sha256(script_path) == script_hash, "Verifier source changed during execution")
    result = {
        "status": "passed", "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": sys.orig_argv, "daily_runs_checked": len(runs), "calendar_days_per_run": len(dates),
        "statistical_rows_checked": len(paired), "fields_checked": ["n_dates", "mean", "ci_low", "ci_high"],
        "max_absolute_error": max(interval_errors.values()), "max_absolute_error_by_field": interval_errors,
        "period_rows_checked": len(periods), "period_fields_checked": list(next(iter(recalculated.values()))),
        "period_max_absolute_error_by_field": period_errors, "daily_checks": daily_errors,
        "absolute_tolerance": ABSOLUTE_TOLERANCE,
        "method": {"source": "Consecutive saved NAV ratios; no research statistics functions imported",
                   "metrics": METRICS, "periods": PERIODS,
                   "alignment": "Identical full business-day calendars; initial missing return retained",
                   "bootstrap": "Noncircular moving contiguous blocks, concatenated and truncated to calendar length",
                   "block_days": block_days, "draws": draws, "seed": seed, "percentiles": [2.5, 97.5],
                   "rng": "numpy.default_rng; restarted per comparison"},
        "scope": {"prerequisite_account_audit": "Existing passed verification.json bound to this run manifest and audit source",
                  "current_checks": "Frozen input/code/output hashes, daily NAV arithmetic, daily-derived period aggregates and paired intervals",
                  "excluded_period_fields": ["filled_orders"],
                  "limitations": "Order execution is not re-audited here; daily cost/conflict totals are aggregate checks. Reference intervals do not establish new-sample validity."},
        "inputs_sha256": hashes, "verification_code_sha256": script_hash,
        "environment": {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__},
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"Passed: {len(runs)} daily runs, {len(periods)} period rows, {len(paired)} paired interval rows.")
    print(f"Maximum interval absolute error: {result['max_absolute_error']:.17g}")
    print(f"Saved: {destination}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="Complete portfolio_model_comparison output directory")
    parser.add_argument("--verification-output", type=Path, help="New verification JSON; defaults to OUT/statistical_verification.json")
    args = parser.parse_args()
    output = args.out.resolve()
    destination = args.verification_output.resolve() if args.verification_output else output / "statistical_verification.json"
    run(output, destination)
