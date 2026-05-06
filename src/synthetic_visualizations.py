import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from synthetic_data import make_scenario


# This file makes plots for the synthetic experiments.

DEFAULT_RESULTS = Path(__file__).with_name("synthetic_results.csv")
DEFAULT_OUTPUT_DIR = Path(__file__).with_name("synthetic_plots")

# Shorter labels make the x-axis easier to read in the figures.
SCENARIO_LABELS = {
    "weak_features_strong_graph": "Weak features\nstrong graph",
    "weak_graph_strong_features": "Weak graph\nstrong features",
    "few_signal_features": "Few signal\nfeatures",
    "train_only_spurious": "Train-only\nspurious",
    "anti_spurious_test": "Anti-spurious\ntest",
}

METHOD_COLORS = {
    "No selection": "#4c566a",
    "Raw l1": "#d08770",
    "Graph-aware l1": "#bf616a",
    "Raw mutual_info": "#5e81ac",
    "Graph-aware mutual_info": "#88c0d0",
    "PCA": "#b48ead",
    "Autoencoder": "#a3be8c",
    "Learned mask": "#ebcb8b",
}

# These colors are used when we break selected features into ground-truth groups.
GROUP_COLORS = {
    "signal": "#2a9d8f",
    "junk": "#adb5bd",
    "spurious": "#e76f51",
}


def read_results(path):
    # Read the CSV produced by synthetic_experiments.py.
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))

    numeric_columns = [
        "accuracy_mean",
        "accuracy_std",
        "signal_selected_mean",
        "junk_selected_mean",
        "spurious_selected_mean",
        "signal_recall_mean",
        "signal_precision_mean",
        "k",
        "num_features",
        "num_signal_features",
        "num_junk_features",
        "num_spurious_features",
        "p_in",
        "p_out",
    ]

    for row in rows:
        for column in numeric_columns:
            value = row.get(column, "")
            row[column] = float(value) if value not in ("", None) else None

    return rows


def ordered_values(rows, key):
    # Keep the same order as the CSV instead of sorting alphabetically.
    return list(dict.fromkeys(row[key] for row in rows))


def label_scenario(scenario):
    # Use nicer labels when we know the scenario, otherwise fall back to the raw name.
    return SCENARIO_LABELS.get(scenario, scenario.replace("_", "\n"))


def save_figure(fig, output_dir, filename):
    # Save every plot into the synthetic_plots folder so outputs stay organized.
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_accuracy_by_scenario(rows, output_dir):
    # Plot accuracy for each method and scenario, averaged across seeds. 
    scenarios = ordered_values(rows, "scenario")
    methods = ordered_values(rows, "method")
    by_pair = {(row["scenario"], row["method"]): row for row in rows}

    x = np.arange(len(scenarios))
    width = 0.12

    fig, ax = plt.subplots(figsize=(13, 6))
    for i, method in enumerate(methods):
        offsets = x + (i - (len(methods) - 1) / 2) * width
        means = [by_pair[(scenario, method)]["accuracy_mean"] for scenario in scenarios]
        stds = [by_pair[(scenario, method)]["accuracy_std"] for scenario in scenarios]
        ax.bar(
            offsets,
            means,
            width,
            yerr=stds,
            capsize=2,
            label=method,
            color=METHOD_COLORS.get(method),
            edgecolor="white",
            linewidth=0.6,
        )

    ax.set_title("Synthetic Scenario Accuracy")
    ax.set_ylabel("Test accuracy")
    ax.set_ylim(0, 1.08)
    ax.set_xticks(x)
    ax.set_xticklabels([label_scenario(scenario) for scenario in scenarios])
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=3, fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.14))
    fig.tight_layout()
    return save_figure(fig, output_dir, "synthetic_accuracy_by_scenario.png")


def plot_accuracy_heatmap(rows, output_dir):
    # Heatmap version of the accuracy results.
    scenarios = ordered_values(rows, "scenario")
    methods = ordered_values(rows, "method")
    by_pair = {(row["scenario"], row["method"]): row for row in rows}

    matrix = np.full((len(methods), len(scenarios)), np.nan)
    for row_idx, method in enumerate(methods):
        for col_idx, scenario in enumerate(scenarios):
            row = by_pair.get((scenario, method))
            if row is not None:
                matrix[row_idx, col_idx] = row["accuracy_mean"]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    image = ax.imshow(matrix, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set_title("Synthetic Accuracy Heatmap")
    ax.set_xlabel("Synthetic scenario")
    ax.set_ylabel("Method")
    ax.set_xticks(np.arange(len(scenarios)))
    ax.set_xticklabels([label_scenario(scenario) for scenario in scenarios])
    ax.set_yticks(np.arange(len(methods)))
    ax.set_yticklabels(methods)

    for row_idx in range(len(methods)):
        for col_idx in range(len(scenarios)):
            value = matrix[row_idx, col_idx]
            if not np.isnan(value):
                text_color = "white" if value < 0.55 else "black"
                ax.text(col_idx, row_idx, f"{value:.2f}", ha="center", va="center", color=text_color, fontsize=8)

    fig.colorbar(image, ax=ax, label="Test accuracy")
    fig.tight_layout()
    return save_figure(fig, output_dir, "synthetic_accuracy_heatmap.png")


def plot_graph_aware_gain(rows, output_dir):
   # Plot comparison between graph-aware and raw versions of the same method. 
    scenarios = ordered_values(rows, "scenario")
    by_pair = {(row["scenario"], row["method"]): row for row in rows}
    comparisons = [
        ("L1", "Raw l1", "Graph-aware l1", "#bf616a"),
        ("Mutual info", "Raw mutual_info", "Graph-aware mutual_info", "#5e81ac"),
    ]

    x = np.arange(len(scenarios))
    width = 0.28

    fig, ax = plt.subplots(figsize=(11, 5))
    for i, (label, raw_method, graph_method, color) in enumerate(comparisons):
        gains = []
        for scenario in scenarios:
            raw_row = by_pair.get((scenario, raw_method))
            graph_row = by_pair.get((scenario, graph_method))
            if raw_row is None or graph_row is None:
                gains.append(np.nan)
            else:
                gains.append(graph_row["accuracy_mean"] - raw_row["accuracy_mean"])

        ax.bar(
            x + (i - 0.5) * width,
            gains,
            width,
            label=label,
            color=color,
            edgecolor="white",
            linewidth=0.6,
        )

    ax.axhline(0, color="#2e3440", linewidth=1)
    ax.set_title("Graph-Aware Accuracy Gain")
    ax.set_ylabel("Graph-aware accuracy - raw accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels([label_scenario(scenario) for scenario in scenarios])
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    return save_figure(fig, output_dir, "synthetic_graph_aware_gain.png")


def plot_method_family_summary(rows, output_dir):
    # Average each method across all synthetic scenarios.
    methods = ordered_values(rows, "method")
    means = []
    stds = []

    for method in methods:
        accuracies = [row["accuracy_mean"] for row in rows if row["method"] == method]
        means.append(float(np.mean(accuracies)))
        stds.append(float(np.std(accuracies)))

    order = np.argsort(means)
    sorted_methods = [methods[i] for i in order]
    sorted_means = [means[i] for i in order]
    sorted_stds = [stds[i] for i in order]

    fig, ax = plt.subplots(figsize=(9, 5))
    y = np.arange(len(sorted_methods))
    colors = [METHOD_COLORS.get(method, "#6c757d") for method in sorted_methods]
    ax.barh(y, sorted_means, xerr=sorted_stds, color=colors, capsize=3)
    ax.set_title("Average Synthetic Accuracy by Method")
    ax.set_xlabel("Mean test accuracy across synthetic scenarios")
    ax.set_yticks(y)
    ax.set_yticklabels(sorted_methods)
    ax.set_xlim(0, 1.05)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    return save_figure(fig, output_dir, "synthetic_method_family_summary.png")


def plot_signal_recovery(rows, output_dir):
    # Plot how well each method recovered signal features, averaged across seeds.
    selection_rows = [row for row in rows if row["signal_recall_mean"] is not None]
    scenarios = ordered_values(selection_rows, "scenario")
    methods = ordered_values(selection_rows, "method")
    by_pair = {(row["scenario"], row["method"]): row for row in selection_rows}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    metrics = [
        ("signal_recall_mean", "Signal recall"),
        ("signal_precision_mean", "Signal precision"),
    ]
    width = 0.18
    x = np.arange(len(scenarios))

    for ax, (metric, title) in zip(axes, metrics):
        for i, method in enumerate(methods):
            offsets = x + (i - (len(methods) - 1) / 2) * width
            values = [by_pair[(scenario, method)][metric] for scenario in scenarios]
            ax.bar(offsets, values, width, label=method, color=METHOD_COLORS.get(method))

        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels([label_scenario(scenario) for scenario in scenarios])
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.25)

    axes[0].set_ylabel("Score")
    axes[1].legend(ncol=2, fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    fig.tight_layout()
    return save_figure(fig, output_dir, "synthetic_signal_recovery.png")

def sorted_indices_by_class(data):
    # Sorting nodes by class makes synthetic heatmaps easier to read.
    labels = data.y.numpy()
    return np.argsort(labels)


def make_all_plots(results_csv=DEFAULT_RESULTS, output_dir=DEFAULT_OUTPUT_DIR, scenario_for_heatmaps="train_only_spurious"):
    # Main entry point used by the script.
    rows = read_results(results_csv)
    generated = [
        plot_accuracy_by_scenario(rows, output_dir),
        plot_accuracy_heatmap(rows, output_dir),
        plot_method_family_summary(rows, output_dir),
        plot_graph_aware_gain(rows, output_dir),
        plot_signal_recovery(rows, output_dir),
    ]
    return generated


def main():
    # Allows the file to be run directly from the terminal.
    parser = argparse.ArgumentParser(description="Create visualizations for synthetic graph experiments.")
    parser.add_argument("--results-csv", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--heatmap-scenario", default="train_only_spurious")
    args = parser.parse_args()

    generated = make_all_plots(
        results_csv=args.results_csv,
        output_dir=args.output_dir,
        scenario_for_heatmaps=args.heatmap_scenario,
    )

    print("Generated synthetic visualizations:")
    for path in generated:
        print(path)


if __name__ == "__main__":
    main()
