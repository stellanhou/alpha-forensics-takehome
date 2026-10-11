"""Frozen nested historical study of IC, correlation clustering and sparse selection.

Run tune, validation, then confirmation in one new output directory. The data,
including 2024 outcomes, were inspected before this additional historical study.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import cut_tree, linkage
from scipy.spatial.distance import squareform
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.preprocessing import StandardScaler

from compare_models import (
    BLOCK_DAYS, BOOTSTRAP_DRAWS, FACTORS, KEYS, PHASE_ORIGIN, PROJECT, SEED,
    daily_metrics, jsonable, key_hash, model_pipeline, object_hash,
    paired_mean_difference, sha256, summarize_model, temporal_split, utc_now,
    validate_panel, write_json,
)


LEARNERS = ["Ridge", "HGB"]
FAMILIES = ["ALL", "IC", "HAC", "EN"]
INNER_FOLDS = [f"2022Q{i}" for i in range(1, 5)]
SELECTOR_CANDIDATES = (
    [{"family": "ALL", "selector_id": "all20"}]
    + [{"family": family, "selector_id": f"{family}_k{k}", "k": k}
       for family in ["IC", "HAC"] for k in [5, 10, 15]]
    + [{"family": "EN", "selector_id": f"EN_alpha{alpha}", "alpha": alpha}
       for alpha in [.001, .01, .05]]
)
SCOPE = (
    "Additional historical research designed after inspecting 2024 outcomes; "
    "not a new independent blind test. Uses the supplied observed-label member "
    "panel. Gross five-day grouping diagnostics are not portfolio returns; "
    "no NAV, Sharpe, trading costs or executable strategy claims."
)


def make_learner(name: str):
    if name == "Ridge":
        return model_pipeline(Ridge(alpha=100), linear=True)
    if name == "HGB":
        return model_pipeline(HistGradientBoostingRegressor(
            max_iter=55, learning_rate=.065, max_leaf_nodes=17,
            l2_regularization=.2, early_stopping=False, random_state=SEED))
    raise ValueError(f"Unknown learner: {name}")


def elastic_estimator(alpha: float):
    return ElasticNet(alpha=alpha, l1_ratio=.5, max_iter=5000, tol=1e-5,
                      selection="cyclic", random_state=SEED)


def describe_rows(frame: pd.DataFrame) -> dict:
    return {"rows": len(frame), "signal_dates": int(frame.date.nunique()),
            "signal_start": str(frame.date.min().date()),
            "signal_end": str(frame.date.max().date()),
            "max_exit_date": str(frame.exit_date.max().date()),
            "ordered_keys_sha256": key_hash(frame)}


def inner_splits(outer_training: pd.DataFrame):
    validate_panel(outer_training)
    if not outer_training.date.dt.year.between(2021, 2022).all():
        raise ValueError("Inner tuning may only receive the 2021-2022 outer training segment")
    if not (outer_training.exit_date < pd.Timestamp("2023-01-02")).all():
        raise ValueError("Inner source includes labels crossing the outer boundary")
    splits = []
    for fold in INNER_FOLDS:
        quarter = pd.Period(fold, freq="Q")
        first = pd.offsets.BDay().rollforward(quarter.start_time)
        following = pd.offsets.BDay().rollforward((quarter + 1).start_time)
        train_candidates = outer_training.loc[outer_training.date < first]
        eval_candidates = outer_training.loc[outer_training.date.dt.to_period("Q") == quarter]
        train = train_candidates.loc[train_candidates.exit_date < first].sort_values(KEYS).reset_index(drop=True)
        evaluation = eval_candidates.loc[eval_candidates.exit_date < following].sort_values(KEYS).reset_index(drop=True)
        if train.empty or evaluation.empty:
            raise ValueError(f"{fold}: empty training or evaluation segment")
        if train[FACTORS].isna().all().any():
            raise ValueError(f"{fold}: entirely missing training factor")
        if not train.exit_date.max() < first or not evaluation.exit_date.max() < following:
            raise ValueError(f"{fold}: label boundary overlap")
        manifest = {"fold": fold, "train": describe_rows(train), "evaluation": describe_rows(evaluation),
                    "quarter_first_business_day": str(first.date()),
                    "next_quarter_first_business_day": str(following.date()),
                    "purged_training_rows": len(train_candidates) - len(train),
                    "purged_evaluation_rows": len(eval_candidates) - len(evaluation),
                    "source": "Already purged outer validation training segment",
                    "training_exit_before_quarter_start": True,
                    "evaluation_exit_before_next_quarter": True}
        splits.append((fold, train, evaluation, manifest))
    return splits


def selector_statistics(training: pd.DataFrame) -> dict:
    """Training-only common-sample daily Spearman statistics and HAC geometry."""
    validate_panel(training)
    if training[FACTORS].isna().all().any():
        raise ValueError("Entirely missing training factor")
    imputer = SimpleImputer(strategy="median")
    common = pd.DataFrame(imputer.fit_transform(training[FACTORS]), columns=FACTORS,
                          index=training.index)
    common["forward_5d"] = training.forward_5d
    common["date"] = training.date
    correlations, dates = [], []
    for date, daily in common.groupby("date", sort=True):
        ranks = daily[FACTORS + ["forward_5d"]].rank(method="average").to_numpy()
        centered = ranks - ranks.mean(axis=0)
        scale = np.sqrt(np.sum(centered ** 2, axis=0))
        with np.errstate(divide="ignore", invalid="ignore"):
            corr = (centered.T @ centered) / np.outer(scale, scale)
        correlations.append(corr)
        dates.append(date)
    cube = np.asarray(correlations)
    valid_days = np.isfinite(cube).sum(axis=0)
    mean_corr_all = np.divide(np.nansum(cube, axis=0), valid_days,
                              out=np.full(valid_days.shape, np.nan), where=valid_days > 0)
    if (valid_days[:20, :20] == 0).any():
        raise ValueError("A factor pair has no defined training daily correlation")
    if (valid_days[:20, 20] == 0).any():
        raise ValueError("A factor has no defined training daily IC")
    daily_ic = pd.DataFrame(cube[:, :20, 20], index=pd.DatetimeIndex(dates, name="date"), columns=FACTORS)
    quarterly = daily_ic.groupby(daily_ic.index.to_period("Q")).mean()
    means = daily_ic.mean()
    stability = quarterly.apply(lambda col: float(
        (np.sign(col.dropna()) == np.sign(means[col.name])).mean()))
    ic = pd.DataFrame({"factor": FACTORS, "mean_daily_ic": means.values,
                       "abs_mean_daily_ic": means.abs().values,
                       "valid_ic_days": daily_ic.notna().sum().values,
                       "quarter_sign_stability": stability.values})
    ic = ic.sort_values(["abs_mean_daily_ic", "quarter_sign_stability", "factor"],
                        ascending=[False, False, True]).reset_index(drop=True)
    ic["ic_rank"] = np.arange(1, 21)
    corr = (mean_corr_all[:20, :20] + mean_corr_all[:20, :20].T) / 2
    # Correlation roundoff can exceed one by machine epsilon; assert the bound first.
    if (np.abs(corr) > 1 + 1e-12).any():
        raise ValueError("Training correlation is outside [-1, 1]")
    corr = np.clip(corr, -1, 1)
    distance = 1 - np.abs(corr)
    np.fill_diagonal(distance, 0)
    condensed = squareform(distance, checks=True)
    tree = linkage(condensed, method="average")
    return {"ic": ic, "daily_ic": daily_ic, "quarterly_ic": quarterly,
            "mean_corr": pd.DataFrame(corr, index=FACTORS, columns=FACTORS),
            "pair_valid_days": pd.DataFrame(valid_days[:20, :20], index=FACTORS, columns=FACTORS),
            "distance": distance, "linkage": tree,
            "imputer_medians": dict(zip(FACTORS, imputer.statistics_.tolist()))}


def select_ic(stats: dict, k: int) -> list[str]:
    if not 1 <= k <= len(FACTORS):
        raise ValueError("IC feature count is outside the 20-factor universe")
    chosen = set(stats["ic"].sort_values("ic_rank").factor.iloc[:k])
    return [factor for factor in FACTORS if factor in chosen]


def select_hac(stats: dict, k: int):
    if not 1 <= k <= len(FACTORS):
        raise ValueError("HAC cluster count is outside the 20-factor universe")
    labels = cut_tree(stats["linkage"], n_clusters=[k]).ravel()
    if len(np.unique(labels)) != k:
        raise ValueError("HAC did not produce exactly the requested number of clusters")
    ranks = stats["ic"].set_index("factor").ic_rank
    clusters = pd.DataFrame({"factor": FACTORS, "cluster": labels})
    representatives = {}
    for cluster, part in clusters.groupby("cluster"):
        representatives[int(cluster)] = min(part.factor, key=lambda factor: ranks[factor])
    clusters["representative"] = clusters.cluster.map(representatives)
    clusters["selected"] = clusters.factor.eq(clusters.representative)
    return clusters.loc[clusters.selected, "factor"].tolist(), clusters


def fit_elastic_selector(training: pd.DataFrame, alpha: float) -> dict:
    validate_panel(training)
    if training[FACTORS].isna().all().any():
        raise ValueError("Entirely missing training factor")
    imputer = SimpleImputer(strategy="median")
    x_scaler, y_scaler = StandardScaler(), StandardScaler()
    x = x_scaler.fit_transform(imputer.fit_transform(training[FACTORS]))
    y = y_scaler.fit_transform(training[["forward_5d"]]).ravel()
    model = elastic_estimator(alpha)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(x, y)
    convergence = [str(item.message) for item in caught if issubclass(item.category, ConvergenceWarning)]
    features = [factor for factor, coefficient in zip(FACTORS, model.coef_) if coefficient != 0]
    reason = "; ".join(convergence) if convergence else ("ElasticNet selected zero original factors" if not features else "")
    return {"status": "ineligible" if reason else "eligible", "reason": reason,
            "features": features, "nonzero_count": len(features),
            "coefficients": dict(zip(FACTORS, model.coef_.tolist())),
            "n_iter": int(model.n_iter_), "dual_gap": float(model.dual_gap_),
            "convergence_warnings": convergence,
            "parameters": jsonable(model.get_params()),
            "imputer_medians": dict(zip(FACTORS, imputer.statistics_.tolist())),
            "x_scaler_mean": x_scaler.mean_.tolist(), "x_scaler_scale": x_scaler.scale_.tolist(),
            "y_scaler_mean": y_scaler.mean_.tolist(), "y_scaler_scale": y_scaler.scale_.tolist()}


def fit_selector(training: pd.DataFrame, candidate: dict, stats: dict) -> dict:
    family = candidate["family"]
    if family == "EN":
        return fit_elastic_selector(training, candidate["alpha"])
    if family == "ALL":
        features, metadata = FACTORS.copy(), {}
    elif family == "IC":
        features, metadata = select_ic(stats, candidate["k"]), {"ranking": stats["ic"].factor.tolist()}
    elif family == "HAC":
        features, clusters = select_hac(stats, candidate["k"])
        metadata = {"clusters": clusters.to_dict("records")}
    else:
        raise ValueError(f"Unknown selector family: {family}")
    return {"status": "eligible", "reason": "", "features": features, "metadata": metadata}


def save_statistics(out: Path, stats: dict) -> None:
    out.mkdir()
    stats["ic"].to_csv(out / "ic_statistics.csv", index=False)
    stats["daily_ic"].to_csv(out / "daily_ic.csv")
    stats["quarterly_ic"].to_csv(out / "quarterly_ic.csv", index_label="quarter")
    stats["mean_corr"].to_csv(out / "mean_daily_corr.csv", index_label="factor")
    stats["pair_valid_days"].to_csv(out / "pair_valid_days.csv", index_label="factor")
    pd.DataFrame(stats["distance"], index=FACTORS, columns=FACTORS).to_csv(out / "distance.csv", index_label="factor")
    pd.DataFrame(stats["linkage"], columns=["left", "right", "distance", "count"]).to_csv(out / "linkage.csv", index=False)
    write_json(out / "imputer_metadata.json", {"training_medians": stats["imputer_medians"]})
    for k in [5, 10, 15]:
        _, clusters = select_hac(stats, k)
        clusters.to_csv(out / f"clusters_k{k}.csv", index=False)


def fit_model(model, training, evaluation, features):
    tick = time.perf_counter()
    model.fit(training[features], training.forward_5d)
    fit_seconds = time.perf_counter() - tick
    fitted = {"features": features, "training_rows": len(training),
              "imputer_medians": model.named_steps["imputer"].statistics_.tolist()}
    if "scaler" in model.named_steps:
        fitted.update(x_scaler_mean=model.named_steps["scaler"].mean_.tolist(),
                      x_scaler_scale=model.named_steps["scaler"].scale_.tolist())
    if isinstance(model.named_steps["model"], Ridge):
        fitted["coefficients"] = model.named_steps["model"].coef_.tolist()
    else:
        fitted["iterations"] = int(model.named_steps["model"].n_iter_)
    tick = time.perf_counter()
    prediction = model.predict(evaluation[features])
    predict_seconds = time.perf_counter() - tick
    if not np.isfinite(prediction).all():
        raise ValueError("Non-finite downstream predictions")
    result = evaluation[KEYS + ["entry_date", "exit_date", "forward_5d"]].assign(prediction=prediction)
    return result, fitted, fit_seconds, predict_seconds


def choose_inner_configs(fold_scores: pd.DataFrame):
    records, selected = [], {}
    for learner in LEARNERS:
        selected[learner] = {}
        for candidate in SELECTOR_CANDIDATES:
            rows = fold_scores.loc[(fold_scores.learner == learner)
                                   & (fold_scores.selector_id == candidate["selector_id"])]
            if rows.fold.duplicated().any() or set(rows.fold) != set(INNER_FOLDS):
                raise ValueError("Inner candidate is missing a fold or contains duplicate folds")
            valid = (rows.status.eq("eligible") & np.isfinite(rows.rank_ic_mean)
                     & rows.rank_ic_valid_dates.gt(0))
            score = float(np.average(rows.rank_ic_mean, weights=rows.rank_ic_valid_dates)) if valid.all() else np.nan
            reasons = [f"{row.fold}: {row.reason or 'undefined daily RankIC'}"
                       for row in rows.loc[~valid].itertuples()]
            records.append({"learner": learner, **candidate, "status": "eligible" if valid.all() else "ineligible",
                            "reason": "; ".join(reasons), "valid_folds": int(valid.sum()),
                            "rank_ic_mean": score, "rank_ic_valid_dates": int(rows.rank_ic_valid_dates.sum())})
        learner_records = records[-len(SELECTOR_CANDIDATES):]
        for family in FAMILIES:
            eligible = [row for row in learner_records if row["family"] == family and row["status"] == "eligible"]
            if eligible:
                best = max(eligible, key=lambda row: row["rank_ic_mean"])
                selected[learner][family] = next(candidate.copy() for candidate in SELECTOR_CANDIDATES
                                                  if candidate["selector_id"] == best["selector_id"])
            else:
                selected[learner][family] = None
    return selected, pd.DataFrame(records)


def build_outer_arms(learner: str, tuned_for_learner: dict, training: pd.DataFrame, stats: dict):
    arms = []
    for family in FAMILIES:
        candidate = tuned_for_learner[family]
        if candidate is None:
            result = {"status": "ineligible", "reason": "No eligible complete inner-CV configuration in this family", "features": []}
        else:
            result = fit_selector(training, candidate, stats)
        arms.append({"arm_id": f"{learner}__{family}", "learner": learner, "family": family,
                     "primary": True, "selector": candidate, "status": result["status"],
                     "reason": result["reason"], "features": result["features"], "metadata": result})
    for source_family in ["HAC", "EN"]:
        source = next(arm for arm in arms if arm["family"] == source_family)
        family = f"IC_match_{source_family}"
        k = len(source["features"])
        features = select_ic(stats, k) if source["status"] == "eligible" else []
        arms.append({"arm_id": f"{learner}__{family}", "learner": learner, "family": family,
                     "primary": False, "selector": {"family": "IC", "k": k, "matched_to": source["arm_id"]},
                     "status": source["status"], "reason": source["reason"], "features": features,
                     "metadata": {"matched_count": k, "matched_to": source["arm_id"]}})
    for index, arm in enumerate(arms):
        duplicates = [other["arm_id"] for other in arms[:index]
                      if arm["status"] == other["status"] == "eligible" and arm["features"] == other["features"]]
        arm["duplicate_of"] = duplicates[0] if duplicates else ""
    return arms


def build_protocol(panel_path: Path) -> dict:
    timing_path = panel_path.parent / "run_manifest.json"
    timing = json.loads(timing_path.read_text())
    panel_hash = sha256(panel_path)
    if timing["outputs_sha256"][panel_path.name] != panel_hash:
        raise ValueError("PIT panel differs from its factor-timing manifest")
    return {
        "format_version": 1, "scope": SCOPE, "seed": SEED,
        "factor_columns": FACTORS, "learners": LEARNERS, "primary_family_order": FAMILIES,
        "selector_candidates": SELECTOR_CANDIDATES,
        "model_parameters": {name: jsonable(make_learner(name).get_params(deep=True)) for name in LEARNERS},
        "elastic_parameters": {str(alpha): jsonable(elastic_estimator(alpha).get_params()) for alpha in [.001, .01, .05]},
        "preprocessing": "Fit median imputation, X scaling and EN y scaling on each applicable training segment only",
        "inner_folds": INNER_FOLDS, "inner_training": "Expanding; signal and exit before quarter first business day",
        "inner_evaluation": "Signal in quarter; exit before following quarter first business day; Q4 excludes 2023 prices",
        "outer_split": "Exact compare_models.temporal_split; 2023 validation labels exit before 2024-01-01",
        "ic_statistics": "Imputed common 20-factor sample; average-rank daily Spearman; equal-weight daily mean",
        "representative_order": "abs(mean daily IC) descending, quarterly sign stability descending, factor ID ascending",
        "quarter_sign_stability": "Share of defined training quarterly mean ICs whose sign equals full-training mean IC sign",
        "hac": "D=1-abs(mean daily correlation), symmetric and diagonal zero; condensed squareform; average linkage; cut_tree exact k",
        "undefined_correlations": "Keep daily NaN and average over valid dates; save pair counts; fail if any pair wholly undefined",
        "elastic_selection": "Nonzero coefficients select original factors; empty set or ConvergenceWarning is explicitly ineligible",
        "inner_selection": "Per learner and selector family; all four folds eligible; maximize pooled valid-day mean signed prediction RankIC; declaration-order tie break",
        "auxiliary_controls": "IC_match_HAC and IC_match_EN use current training selected count; excluded from winner selection; duplicates retain identity",
        "validation_selection": "Highest 2023 mean RankIC among eligible primary arms; exact ties ALL,IC,HAC,EN then Ridge,HGB",
        "confirmation": "Frozen chosen pipeline and selector hyperparameters; refit selector on 2021-23; IDs/EN count may update; no retuning",
        "bootstrap": {"block_days": BLOCK_DAYS, "draws": BOOTSTRAP_DRAWS, "seed": SEED,
                      "main_reference": "Same learner ALL20", "matched_comparisons": ["HAC-IC_match_HAC", "EN-IC_match_EN"]},
        "phase_origin": PHASE_ORIGIN,
        "panel_path": str(panel_path.resolve()),
        "inputs_sha256": {"pit_panel": panel_hash, "timing_manifest": sha256(timing_path)},
        "code_sha256": {path.name: sha256(path) for path in [Path(__file__), PROJECT / "test_factor_selection_study.py", PROJECT / "compare_models.py"]},
        "versions": {"python": platform.python_version(), **{name: importlib.metadata.version(name)
                     for name in ["numpy", "pandas", "scipy", "scikit-learn"]}},
        "threads_environment": {name: os.environ.get(name) for name in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]},
    }


def verify_existing_protocol(out: Path, protocol: dict) -> None:
    frozen = json.loads((out / "protocol.json").read_text())
    digest = object_hash(protocol)
    if frozen["protocol_sha256"] != digest or object_hash(frozen["protocol"]) != digest:
        raise ValueError("Frozen protocol changed; use a new output directory")


def verify_stage(out: Path, stage: str, protocol_digest: str) -> dict:
    manifest = json.loads((out / stage / "run_manifest.json").read_text())
    if manifest["status"] != "completed" or manifest["protocol_sha256"] != protocol_digest:
        raise ValueError(f"{stage}: not complete under the current protocol")
    for relative, expected in manifest["outputs_sha256"].items():
        if sha256(out / relative) != expected:
            raise ValueError(f"{stage}: evidence changed: {relative}")
    for dependency in ["tune", "validation"]:
        name = f"{dependency}_manifest"
        if name in manifest["inputs_sha256"]:
            actual = sha256(out / dependency / "run_manifest.json")
            if actual != manifest["inputs_sha256"][name]:
                raise ValueError(f"{stage}: dependency manifest changed: {dependency}")
    return manifest


def finish_stage(out: Path, stage: str, protocol: dict, started: str, extra_paths: list[Path]) -> None:
    stage_out = out / stage
    paths = sorted(path for path in stage_out.rglob("*") if path.is_file()) + [out / "protocol.json"] + extra_paths
    inputs = protocol["inputs_sha256"].copy()
    for dependency in ["tune", "validation"]:
        path = out / dependency / "run_manifest.json"
        if dependency != stage and path.exists():
            inputs[f"{dependency}_manifest"] = sha256(path)
    write_json(stage_out / "run_manifest.json", {
        "status": "completed", "stage": stage, "started_at_utc": started, "finished_at_utc": utc_now(),
        "protocol_sha256": object_hash(protocol), "scope": SCOPE, "command": sys.orig_argv,
        "code_sha256": protocol["code_sha256"], "inputs_sha256": inputs,
        "outputs_sha256": {str(path.relative_to(out)): sha256(path) for path in paths},
    })


def run_tune(out: Path, training: pd.DataFrame, protocol: dict, outer_manifest: dict, started: str):
    stage_out = out / "tune"
    stage_out.mkdir()
    write_json(stage_out / "outer_training_manifest.json", outer_manifest)
    scores, daily_parts, timings = [], [], []
    for fold, train, evaluation, manifest in inner_splits(training):
        fold_out = stage_out / fold
        fold_out.mkdir()
        write_json(fold_out / "split_manifest.json", manifest)
        stats_tick = time.perf_counter()
        stats = selector_statistics(train)
        common_seconds = time.perf_counter() - stats_tick
        save_statistics(fold_out / "common_statistics", stats)
        print(f"{utc_now()} {fold}: training={len(train)} evaluation={len(evaluation)}", flush=True)
        for candidate in SELECTOR_CANDIDATES:
            tick = time.perf_counter()
            fitted_selector = fit_selector(train, candidate, stats)
            selector_seconds = time.perf_counter() - tick
            # Freeze the selected inputs before fitting or scoring either learner.
            write_json(fold_out / f"{candidate['selector_id']}_selector.json", {"candidate": candidate, **fitted_selector})
            for learner in LEARNERS:
                identity = {"learner": learner, "selector_id": candidate["selector_id"], "family": candidate["family"], "fold": fold}
                status, reason = fitted_selector["status"], fitted_selector["reason"]
                fit_seconds = predict_seconds = 0.
                metric, valid_dates = np.nan, 0
                if status == "eligible":
                    result, fitted, fit_seconds, predict_seconds = fit_model(
                        make_learner(learner), train, evaluation, fitted_selector["features"])
                    write_json(fold_out / f"{learner}__{candidate['selector_id']}_fit.json", fitted)
                    daily = daily_metrics(result)
                    valid_dates = int(daily.rank_ic.notna().sum())
                    metric = float(daily.rank_ic.mean())
                    daily_parts.append(daily.assign(**identity))
                    if valid_dates == 0:
                        status, reason = "ineligible", "No defined daily prediction RankIC in this fold"
                scores.append({**identity, "status": status, "reason": reason,
                               "n_features": len(fitted_selector["features"]),
                               "selected_features": json.dumps(fitted_selector["features"]),
                               "rank_ic_mean": metric, "rank_ic_valid_dates": valid_dates})
                timings.append({**identity, "common_statistics_seconds": common_seconds,
                                "selector_seconds": selector_seconds, "selector_shared_by_learners": True,
                                "fit_seconds": fit_seconds, "prediction_seconds": predict_seconds})
                print(f"  {learner}/{candidate['selector_id']}: {status}, n={len(fitted_selector['features'])}, RankIC={metric:.6f}", flush=True)
    score_table = pd.DataFrame(scores)
    selected, candidate_scores = choose_inner_configs(score_table)
    score_table.to_csv(stage_out / "inner_scores.csv", index=False)
    pd.concat(daily_parts, ignore_index=True).to_csv(stage_out / "inner_daily_metrics.csv", index=False)
    candidate_scores.to_csv(stage_out / "inner_candidate_scores.csv", index=False)
    pd.DataFrame(timings).to_csv(stage_out / "timing.csv", index=False)
    write_json(out / "tuned_selectors.json", {"frozen_at_utc": utc_now(), "protocol_sha256": object_hash(protocol),
               "selected": selected, "inner_scores_sha256": sha256(stage_out / "inner_candidate_scores.csv"),
               "ineligible_candidates": candidate_scores.loc[candidate_scores.status == "ineligible", ["learner", "selector_id", "reason"]].to_dict("records")})
    finish_stage(out, "tune", protocol, started, [out / "tuned_selectors.json"])


def arm_identity(arm: dict) -> dict:
    return {key: arm[key] for key in ["arm_id", "learner", "family", "primary", "status", "reason", "duplicate_of"]}


def choose_pipeline(summaries: pd.DataFrame) -> pd.Series:
    eligible = summaries.loc[summaries.primary & summaries.status.eq("eligible") & np.isfinite(summaries.rank_ic_mean)]
    if eligible.empty:
        raise ValueError("No primary arm has a defined 2023 RankIC; no valid pipeline selection")
    order = {f"{learner}__{family}": i for i, (family, learner) in enumerate(
        (family, learner) for family in FAMILIES for learner in LEARNERS)}
    return eligible.assign(tie_order=eligible.arm_id.map(order)).sort_values(
        ["rank_ic_mean", "tie_order"], ascending=[False, True]).iloc[0]


def matched_comparisons(arms: list[dict], daily_by_arm: dict) -> pd.DataFrame:
    records = []
    by_id = {arm["arm_id"]: arm for arm in arms}
    for learner in LEARNERS:
        for family in ["HAC", "EN"]:
            left, right = f"{learner}__{family}", f"{learner}__IC_match_{family}"
            for metric in ["rank_ic", "gross_decile_spread"]:
                identity = {"learner": learner, "comparison": f"{family}_minus_IC_match_{family}",
                            "candidate_arm": left, "reference_arm": right, "metric": metric}
                if left not in daily_by_arm or right not in daily_by_arm:
                    reasons = [by_id[name]["reason"] for name in [left, right] if by_id[name]["reason"]]
                    records.append({**identity, "status": "ineligible", "reason": "; ".join(reasons),
                                    "n_dates": 0, "mean": np.nan, "ci_low": np.nan, "ci_high": np.nan})
                else:
                    if len(by_id[left]["features"]) != len(by_id[right]["features"]):
                        raise ValueError("Matched control uses a different number of features")
                    result = paired_mean_difference(daily_by_arm[left].set_index("date")[metric],
                                                    daily_by_arm[right].set_index("date")[metric])
                    records.append({**identity, "status": "eligible", "reason": "", **result})
    return pd.DataFrame(records)


def run_outer(stage: str, out: Path, training: pd.DataFrame, evaluation: pd.DataFrame,
              split: dict, protocol: dict, started: str):
    tuned = json.loads((out / "tuned_selectors.json").read_text())
    if tuned["protocol_sha256"] != object_hash(protocol):
        raise ValueError("Tuned selectors do not match the current protocol")
    chosen = json.loads((out / "chosen_pipeline.json").read_text()) if stage == "confirmation" else None
    if chosen and chosen["protocol_sha256"] != object_hash(protocol):
        raise ValueError("Chosen pipeline does not match the current protocol")
    stage_out = out / stage
    stage_out.mkdir()
    write_json(stage_out / "split_manifest.json", split)
    tick = time.perf_counter()
    stats = selector_statistics(training)
    common_seconds = time.perf_counter() - tick
    save_statistics(stage_out / "common_statistics", stats)
    arms, selector_timings = [], {}
    for learner in LEARNERS:
        tick = time.perf_counter()
        learner_arms = build_outer_arms(learner, tuned["selected"][learner], training, stats)
        selector_timings[learner] = time.perf_counter() - tick
        arms.extend(learner_arms)
    # All current training feature lists are frozen before evaluation metrics are read.
    write_json(stage_out / "selected_factors.json", {"frozen_at_utc": utc_now(), "arms": arms})
    summaries, quarters, phases, timings, daily_parts, daily_by_arm, references = [], [], [], [], [], {}, {}
    for arm in arms:
        identity = arm_identity(arm)
        metadata = {**identity, "selector": arm["selector"], "selector_metadata": arm["metadata"]}
        summary = {**identity, "n_features": len(arm["features"]),
                   "selected_features": json.dumps(arm["features"]),
                   "prespecified_confirmation_candidate": bool(chosen and arm["arm_id"] == chosen["arm_id"])}
        fit_seconds = predict_seconds = 0.
        if arm["status"] == "eligible":
            print(f"{utc_now()} {stage} fitting {arm['arm_id']} n={len(arm['features'])}", flush=True)
            result, fitted, fit_seconds, predict_seconds = fit_model(
                make_learner(arm["learner"]), training, evaluation, arm["features"])
            metadata["fitted_model"] = fitted
            if key_hash(result) != split["evaluation"]["ordered_keys_sha256"]:
                raise ValueError("Outer arm does not cover the common evaluation keys")
            result.to_csv(stage_out / f"{arm['arm_id']}_predictions.csv.gz", index=False,
                          compression={"method": "gzip", "compresslevel": 1, "mtime": 0})
            daily = daily_metrics(result)
            daily_by_arm[arm["arm_id"]] = daily
            if arm["family"] == "ALL":
                references[arm["learner"]] = daily.set_index("date").rank_ic
            details, quarterly, phase_rows = summarize_model(
                arm["arm_id"], result, daily, references[arm["learner"]], float(training.forward_5d.mean()))
            details.pop("model")
            summary.update({key.replace("rank_ic_minus_HGB", "rank_ic_minus_ALL20"): value for key, value in details.items()})
            for rows, destination in [(quarterly, quarters), (phase_rows, phases)]:
                for row in rows:
                    row.pop("model")
                    destination.append({**identity, **row})
            daily_parts.append(daily.assign(**identity))
        summaries.append(summary)
        write_json(stage_out / f"{arm['arm_id']}_fit_metadata.json", metadata)
        timings.append({**identity, "common_statistics_seconds": common_seconds,
                        "learner_all_selectors_seconds": selector_timings[arm["learner"]],
                        "fit_seconds": fit_seconds, "prediction_seconds": predict_seconds})
    summary_table = pd.DataFrame(summaries)
    summary_table.to_csv(stage_out / "summary.csv", index=False)
    pd.concat(daily_parts, ignore_index=True).to_csv(stage_out / "daily_metrics.csv", index=False)
    pd.DataFrame(quarters).to_csv(stage_out / "quarterly.csv", index=False)
    pd.DataFrame(phases).to_csv(stage_out / "phases.csv", index=False)
    pd.DataFrame(timings).to_csv(stage_out / "timing.csv", index=False)
    matched_comparisons(arms, daily_by_arm).to_csv(stage_out / "matched_comparisons.csv", index=False)
    if stage == "validation":
        winner = choose_pipeline(summary_table)
        winning_arm = next(arm for arm in arms if arm["arm_id"] == winner.arm_id)
        write_json(out / "chosen_pipeline.json", {
            "frozen_at_utc": utc_now(), "protocol_sha256": object_hash(protocol), "scope": SCOPE,
            "arm_id": winner.arm_id, "learner": winner.learner, "family": winner.family,
            "selector": winning_arm["selector"], "validation_features": winning_arm["features"],
            "validation_mean_daily_rank_ic": float(winner.rank_ic_mean),
            "validation_summary_sha256": sha256(stage_out / "summary.csv"),
            "code_sha256": protocol["code_sha256"],
        })
    finish_stage(out, stage, protocol, started, [out / "tuned_selectors.json", out / "chosen_pipeline.json"])


def run(stage: str, panel_path: Path, out: Path):
    started = utc_now()
    protocol = build_protocol(panel_path)
    digest = object_hash(protocol)
    if stage == "tune":
        out.mkdir(parents=True, exist_ok=True)
        write_json(out / "protocol.json", {"frozen_at_utc": utc_now(), "protocol_sha256": digest, "protocol": protocol})
    else:
        verify_existing_protocol(out, protocol)
        verify_stage(out, "tune", digest)
        if stage == "confirmation":
            verify_stage(out, "validation", digest)
    panel = pd.read_csv(panel_path, parse_dates=["date", "entry_date", "exit_date"])
    outer_stage = "validation" if stage == "tune" else stage
    training, evaluation, split = temporal_split(panel, outer_stage)
    if stage == "tune":
        run_tune(out, training, protocol, split, started)
    else:
        run_outer(stage, out, training, evaluation, split, protocol, started)
    print(f"{stage} completed: {out}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["tune", "validation", "confirmation"], required=True)
    parser.add_argument("--panel", type=Path, default=PROJECT / "outputs" / "factor_timing" / "pit_research_panel.csv.gz")
    parser.add_argument("--out", type=Path, default=PROJECT / "outputs" / "factor_selection_study")
    args = parser.parse_args()
    run(args.stage, args.panel, args.out)
