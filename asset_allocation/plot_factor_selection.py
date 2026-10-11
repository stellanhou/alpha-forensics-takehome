"""Render completed factor-selection evidence without fitting or selecting models."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, leaves_list


FAMILIES = ["ALL", "IC", "HAC", "EN"]
LEARNERS = ["Ridge", "HGB"]
LABELS = {"ALL": "All factors", "IC": "RankIC selection", "HAC": "HAC selection", "EN": "Elastic Net selection"}


def run(root: Path) -> None:
    paths = [root / stage / "summary.csv" for stage in ["validation", "confirmation"]]
    stats = root / "validation" / "common_statistics"
    paths += [stats / "mean_daily_corr.csv", stats / "linkage.csv"]
    validation, confirmation = [pd.read_csv(p).set_index("arm_id") for p in paths[:2]]
    correlation = pd.read_csv(paths[2], index_col=0)
    linkage = pd.read_csv(paths[3]).to_numpy(dtype=float)
    factors = correlation.index.tolist()
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    outputs = []

    def save(fig, stem):
        for suffix in ["png", "pdf"]:
            path = root / f"{stem}.{suffix}"
            fig.savefig(path, dpi=180, bbox_inches="tight")
            outputs.append(path)
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.3))
    for ax, learner in zip(axes, LEARNERS):
        arm_ids = [f"{learner}__{family}" for family in FAMILIES]
        y = np.arange(4)
        for frame, offset, color, label in [
            (validation, -.12, "#8B98A6", "2023 evaluation"),
            (confirmation, .12, "#197A90", "2024 historical confirmation"),
        ]:
            rows = frame.loc[arm_ids]
            ax.hlines(y + offset, rows.rank_ic_ci_low, rows.rank_ic_ci_high, color=color, lw=2)
            ax.scatter(rows.rank_ic_mean, y + offset, color=color, s=32, label=label, zorder=3)
        labels = []
        for family, arm in zip(FAMILIES, arm_ids):
            n1, n2 = int(validation.loc[arm, "n_features"]), int(confirmation.loc[arm, "n_features"])
            marker = " *" if confirmation.loc[arm, "prespecified_confirmation_candidate"] else ""
            labels.append(f"{LABELS[family]} ({n1} / {n2}){marker}")
        ax.set(yticks=y, yticklabels=labels, xlabel="Mean daily Rank IC", title=f"Fixed downstream model: {learner}")
        ax.invert_yaxis()
        ax.axvline(0, color="#555555", linestyle="--", lw=.8)
        ax.grid(axis="x", alpha=.15)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(.055, .91), frameon=False, ncol=2)
    fig.suptitle("Does factor selection improve predictive ranking?", x=.06, ha="left", fontsize=15)
    fig.text(.06, .025,
             "Counts in parentheses: factors selected for 2023 / 2024. * Pipeline chosen using 2023.\n"
             "Intervals: 20-day moving-block bootstrap, 2,000 draws. Previously studied synthetic data.",
             fontsize=8, color="#555555")
    fig.tight_layout(rect=(.01, .12, 1, .86))
    save(fig, "selection_comparison")

    fig, axes = plt.subplots(1, 2, figsize=(13, 6), gridspec_kw={"width_ratios": [1, 1.05]})
    dendrogram(linkage, labels=[f.replace("factor_", "F") for f in factors],
               leaf_rotation=90, leaf_font_size=8, ax=axes[0], color_threshold=0,
               above_threshold_color="#197A90")
    axes[0].set(title="HAC tree: average linkage", ylabel="Distance = 1 - |mean daily Spearman correlation|")
    order = leaves_list(linkage)
    matrix = correlation.to_numpy()[np.ix_(order, order)]
    short_labels = [factors[i].replace("factor_", "F") for i in order]
    im = axes[1].imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1)
    axes[1].set(xticks=np.arange(20), yticks=np.arange(20),
                xticklabels=short_labels, yticklabels=short_labels, title="Training-period factor correlations")
    axes[1].tick_params(axis="x", rotation=90, labelsize=8)
    axes[1].tick_params(axis="y", labelsize=8)
    fig.colorbar(im, ax=axes[1], shrink=.8, label="Signed mean daily correlation")
    counts = [int(validation.loc[f"{model}__HAC", "n_features"]) for model in LEARNERS]
    fig.suptitle("Factor redundancy before the 2023 evaluation", x=.06, ha="left", fontsize=15)
    fig.text(.06, .025,
             f"Only 2021-2022 training rows are used. Inner time validation chose {counts[0]} clusters for Ridge and {counts[1]} for HGB.\n"
             "Clustering measures similarity; representatives are chosen by training-period RankIC.",
             fontsize=8, color="#555555")
    fig.tight_layout(rect=(0, .1, 1, .92))
    save(fig, "hac_structure")

    fig, axes = plt.subplots(2, 1, figsize=(12, 7.6))
    ids = [f"{learner}__{family}" for learner in LEARNERS for family in FAMILIES]
    for ax, frame, title in zip(axes, [validation, confirmation],
                                 ["Selected using 2021-2022 training data", "Refitted using 2021-2023 training data"]):
        matrix = np.array([[factor in json.loads(frame.loc[arm, "selected_features"])
                            for factor in factors] for arm in ids], dtype=int)
        ax.imshow(matrix, cmap=ListedColormap(["#EDF1F4", "#197A90"]), vmin=0, vmax=1, aspect="auto")
        row_labels = [f"{arm.replace('__', ' / ')} ({int(frame.loc[arm, 'n_features'])})" for arm in ids]
        ax.set(xticks=np.arange(20), yticks=np.arange(8),
               xticklabels=[f.replace("factor_", "F") for f in factors],
               yticklabels=row_labels, title=title)
        ax.set_xticks(np.arange(-.5, 20), minor=True)
        ax.set_yticks(np.arange(-.5, 8), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.2)
        ax.tick_params(which="minor", bottom=False, left=False)
        ax.tick_params(axis="x", labelsize=8)
    fig.suptitle("Which factors were retained?", x=.06, ha="left", fontsize=15)
    fig.text(.06, .02, "Blue = retained. Hyperparameters stay frozen; factor identities can change when the training period expands.",
             fontsize=8, color="#555555")
    fig.tight_layout(rect=(0, .06, 1, .94))
    save(fig, "selected_factor_map")
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    manifest = {"inputs_sha256": {str(p.relative_to(root)): digest(p) for p in paths},
                "script_sha256": digest(Path(__file__)),
                "outputs_sha256": {p.name: digest(p) for p in outputs}}
    (root / "plot_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path,
                        default=Path(__file__).resolve().parent / "outputs" / "factor_selection_study")
    run(parser.parse_args().out)
