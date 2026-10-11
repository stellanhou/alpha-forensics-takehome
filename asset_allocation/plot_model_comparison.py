"""Plot completed temporal model comparisons; never trains or selects a model."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def run(root: Path) -> None:
    inputs = [root / stage / "summary.csv" for stage in ["validation", "confirmation"]]
    inputs += [root / "selection.json", root / "protocol.json"]
    validation, confirmation = [pd.read_csv(path).set_index("model") for path in inputs[:2]]
    selected = json.loads(inputs[2].read_text())["selected_model"]
    models = json.loads(inputs[3].read_text())["protocol"]["candidate_order"]
    y = np.arange(len(models))
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.8), gridspec_kw={"width_ratios": [1.15, 1]})
    for frame, offset, color, label in [
        (validation, -.13, "#8796A5", "2023 selection period"),
        (confirmation, .13, "#197A90", "2024 historical confirmation"),
    ]:
        rows = frame.loc[models]
        axes[0].hlines(y + offset, rows.rank_ic_ci_low, rows.rank_ic_ci_high, color=color, lw=2)
        axes[0].scatter(rows.rank_ic_mean, y + offset, color=color, label=label, s=30, zorder=3)
    labels = [f"{name} *" if name == selected else name for name in models]
    axes[0].set(yticks=y, yticklabels=labels, xlabel="Mean daily cross-sectional Rank IC",
                title="Predictive ranking across years")
    rows = confirmation.loc[models]
    colors = ["#9163B4" if name == selected else "#197A90" for name in models]
    axes[1].hlines(y, rows.rank_ic_minus_HGB_ci_low, rows.rank_ic_minus_HGB_ci_high, color=colors, lw=2)
    axes[1].scatter(rows.rank_ic_minus_HGB_mean, y, color=colors, s=35, zorder=3)
    axes[1].set(yticks=y, yticklabels=labels, xlabel="Paired mean Rank IC difference vs HGB",
                title="2024 incremental ranking performance")
    for ax in axes:
        ax.axvline(0, color="#444444", linestyle="--", lw=.8)
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=.15)
    fig.suptitle("CPU model comparison on point-in-time factors", fontsize=15, x=.07, ha="left")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper left", bbox_to_anchor=(.065, .92),
               ncol=2, frameon=False, fontsize=9)
    fig.text(.07, .035,
             "* Candidate selected using 2023 only. Intervals: 20-day moving-block bootstrap, 2,000 draws.\n"
             "Historical synthetic data previously examined; intervals are marginal, not adjusted for model selection.",
             fontsize=8, color="#555555")
    fig.tight_layout(rect=(.02, .09, 1, .87))
    paths = [root / "model_comparison.png", root / "model_comparison.pdf"]
    for path in paths:
        fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {"inputs_sha256": {str(p.relative_to(root)): digest(p) for p in inputs},
                "script_sha256": digest(Path(__file__)),
                "outputs_sha256": {p.name: digest(p) for p in paths}}
    (root / "plot_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path,
                        default=Path(__file__).resolve().parent / "outputs" / "model_benchmark")
    run(parser.parse_args().out)
