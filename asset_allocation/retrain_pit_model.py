"""Retrain the supplied 20-factor model on PIT inputs, then run its backtest.

The control is the unmodified handoff train_model.py/backtest.py, run separately.
Random row holdout and full-panel paper-spread evaluation are intentionally kept
for this paired timing intervention. They are not time-out-of-sample evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from factor_timing import DEFAULT_DATA, FACTORS, KEYS, PROJECT, sha256


SEED = 1021
PARAMETERS = dict(
    max_iter=55, learning_rate=.065, max_leaf_nodes=17,
    l2_regularization=.2, random_state=SEED,
)


def key_hash(frame: pd.DataFrame) -> str:
    return hashlib.sha256(frame[KEYS].to_csv(index=False).encode("utf-8")).hexdigest()


def matched_split(raw: pd.DataFrame, pit: pd.DataFrame):
    """Independently repeat original sampling on both arms and verify the keys."""
    pd.testing.assert_frame_equal(raw[KEYS], pit[KEYS])
    n = min(110_000, len(raw))
    raw_sample = raw.sample(n=n, random_state=SEED).reset_index(drop=True)
    pit_sample = pit.sample(n=n, random_state=SEED).reset_index(drop=True)
    raw_train, raw_valid = train_test_split(
        np.arange(n), test_size=.2, random_state=SEED, shuffle=True
    )
    pit_train, pit_valid = train_test_split(
        np.arange(n), test_size=.2, random_state=SEED, shuffle=True
    )
    proof = {}
    for name, ri, pi in [("train", raw_train, pit_train), ("valid", raw_valid, pit_valid)]:
        original_keys = raw_sample.iloc[ri][KEYS].reset_index(drop=True)
        pit_keys = pit_sample.iloc[pi][KEYS].reset_index(drop=True)
        pd.testing.assert_frame_equal(original_keys, pit_keys)
        proof[name] = {
            "rows": len(ri), "original_ordered_keys_sha256": key_hash(original_keys),
            "pit_ordered_keys_sha256": key_hash(pit_keys),
        }
    assignments = pit_sample[KEYS].copy()
    assignments.insert(0, "sample_order", np.arange(n))
    assignments["split"] = "valid"
    assignments.loc[pit_train, "split"] = "train"
    return pit_sample, pit_train, pit_valid, assignments, proof


def validate_predictions(predictions: pd.DataFrame, expected: pd.DataFrame) -> None:
    predictions.set_index(KEYS, verify_integrity=True)
    pd.testing.assert_frame_equal(predictions[KEYS].reset_index(drop=True), expected[KEYS].reset_index(drop=True))
    if not np.isfinite(predictions.prediction).all():
        raise ValueError("Non-finite model predictions")


def run(root: Path, original: Path, panel_path: Path, out: Path) -> None:
    started = datetime.now(timezone.utc).isoformat()
    # Verify that the saved PIT panel belongs to the already completed timing run.
    timing_manifest_path = panel_path.parent / "run_manifest.json"
    timing = json.loads(timing_manifest_path.read_text())
    if sha256(panel_path) != timing["outputs_sha256"][panel_path.name]:
        raise ValueError("PIT panel differs from its timing-audit manifest")
    for relative, expected in timing["inputs_sha256"].items():
        if sha256(root / relative) != expected:
            raise ValueError(f"Timing-audit input changed: {relative}")

    source = root / "baseline" / "baseline_code"
    spec = importlib.util.spec_from_file_location("handoff_build_factor", source / "build_factor.py")
    handoff = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(handoff)
    raw = handoff.load_research_frame(root / "data").reset_index(drop=True)
    pit = pd.read_csv(panel_path).sort_values(["asset_id", "date"]).reset_index(drop=True)
    if handoff.FACTOR_COLUMNS != FACTORS:
        raise ValueError("Factor column order differs from the original model")
    pd.testing.assert_frame_equal(raw[KEYS], pit[KEYS])
    untouched = [f for f in FACTORS if f not in ["factor_03", "factor_12"]]
    # Compare at exactly the precision consumed by the supplied model.
    np.testing.assert_array_equal(
        raw[untouched].to_numpy(dtype=np.float32), pit[untouched].to_numpy(dtype=np.float32)
    )
    np.testing.assert_array_equal(
        raw.forward_5d.to_numpy(dtype=np.float32), pit.forward_5d.to_numpy(dtype=np.float32)
    )
    original_predictions = pd.read_csv(original / "baseline_predictions.csv.gz")
    validate_predictions(original_predictions, raw)
    sample, train, valid, assignments, split_proof = matched_split(raw, pit)

    x = sample[FACTORS].to_numpy(dtype=np.float32)
    y = sample.forward_5d.to_numpy(dtype=np.float32)
    model = HistGradientBoostingRegressor(**PARAMETERS)
    model.fit(x[train], y[train])
    valid_prediction = model.predict(x[valid])
    correlation = float(np.corrcoef(valid_prediction, y[valid])[0, 1])
    rmse = float(np.sqrt(mean_squared_error(y[valid], valid_prediction)))
    full_prediction = model.predict(pit[FACTORS].to_numpy(dtype=np.float32))
    predictions = pit[KEYS].assign(prediction=full_prediction.astype("float32"))
    validate_predictions(predictions, raw)

    out.mkdir(parents=True, exist_ok=True)
    compression = {"method": "gzip", "compresslevel": 1, "mtime": 0}
    predictions.to_csv(out / "baseline_predictions.csv.gz", index=False, compression=compression)
    assignments.to_csv(out / "sample_assignments.csv.gz", index=False, compression=compression)
    pd.DataFrame([
        {"metric": "sampled_rows", "value": len(sample)},
        {"metric": "sampled_train_rows", "value": len(train)},
        {"metric": "sampled_valid_rows", "value": len(valid)},
        {"metric": "random_valid_pearson", "value": correlation},
        {"metric": "random_valid_rmse", "value": rmse},
    ]).to_csv(out / "baseline_results.csv", index=False)
    joblib.dump(model, out / "model.joblib")

    # Run the supplied backtest unchanged against the corrected model predictions.
    command = [sys.executable, "-B", str(source / "backtest.py"), "--root", str(root), "--out", str(out)]
    with (out / "backtest.log").open("w") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    original_backtest = pd.read_csv(original / "baseline_backtest.csv")
    pit_backtest = pd.read_csv(out / "baseline_backtest.csv")
    pd.testing.assert_series_equal(original_backtest.date, pit_backtest.date)

    sources = {
        "supplied_original": root / "baseline" / "baseline_results.csv",
        "rerun_original": original / "baseline_results.csv",
        "rerun_pit": out / "baseline_results.csv",
    }
    metrics = {name: pd.read_csv(path).set_index("metric", verify_integrity=True).value for name, path in sources.items()}
    for series in metrics.values():
        pd.testing.assert_index_equal(series.index, metrics["rerun_original"].index)
    comparison = pd.DataFrame(metrics)
    comparison["delta_pit_minus_rerun_original"] = comparison.rerun_pit - comparison.rerun_original
    comparison.to_csv(out.parent / "model_comparison.csv")
    metadata = {
        "started_at_utc": started, "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": sys.orig_argv, "backtest_command": command,
        "data_package": "Alpha Forensics v1.0", "data_root": str(root.resolve()),
        "python": platform.python_version(), "numpy": np.__version__,
        "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
        "threads": {k: os.environ.get(k) for k in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"]},
        "n_features": len(FACTORS), "factor_columns": FACTORS,
        "model_parameters": model.get_params(), "fitted_iterations": int(model.n_iter_),
        "train_test_split": "original random 80/20 at row-level; same ordered train/valid keys",
        "sampled_rows": len(sample), "random_seed": SEED, "matched_split_proof": split_proof,
        "full_prediction_count": len(predictions), "full_ordered_keys_sha256": key_hash(predictions),
        "label": "open(t+1) to open(t+6) simple return; float32 labels identical to control",
        "changed_inputs": ["factor_03", "factor_12"], "other_18_inputs_identical_at_float32": True,
        "pit_missing_rows": pit[FACTORS].isna().sum().astype(int).to_dict(),
        "missing_value_handling": "unmodified HistGradientBoostingRegressor native missing-value handling",
        "backtest_dates_identical": True,
        "scope": "Timing-only intervention; random holdout and full-panel overlapping five-day gross spread retained",
        "limitations": "Not temporal OOS; includes training predictions; no position ledger, costs, or execution constraints",
        "inputs_sha256": {
            "pit_panel": sha256(panel_path), "timing_manifest": sha256(timing_manifest_path),
            **{f"original/{name}": sha256(original / name) for name in [
                "baseline_predictions.csv.gz", "baseline_results.csv", "baseline_backtest.csv", "baseline_run_manifest.json"
            ]},
        },
        "code_sha256": {str(p.resolve()): sha256(p) for p in [
            Path(__file__), PROJECT / "factor_timing.py", PROJECT / "test_model_rerun.py",
            source / "build_factor.py", source / "train_model.py", source / "backtest.py",
        ]},
        "output_root": str(out.parent.resolve()),
        "outputs_sha256": {str(p.relative_to(out.parent)): sha256(p) for p in [
            out / "baseline_predictions.csv.gz", out / "baseline_results.csv", out / "baseline_backtest.csv",
            out / "sample_assignments.csv.gz", out / "model.joblib", out.parent / "model_comparison.csv",
        ]},
    }
    (out / "baseline_run_manifest.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(comparison.to_string(), flush=True)
    print(f"Matched samples: train={len(train)}, valid={len(valid)}; iterations={model.n_iter_}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--original", type=Path, default=PROJECT / "outputs" / "model_rerun" / "original")
    parser.add_argument("--panel", type=Path, default=PROJECT / "outputs" / "factor_timing" / "pit_research_panel.csv.gz")
    parser.add_argument("--out", type=Path, default=PROJECT / "outputs" / "model_rerun" / "pit")
    args = parser.parse_args()
    run(args.root, args.original, args.panel, args.out)
