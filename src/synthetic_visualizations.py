import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from synthetic_data import make_scenario


DEFAULT_RESULTS = Path(__file__).with_name("synthetic_results.csv")
DEFAULT_OUTPUT_DIR = Path(__file__).with_name("synthetic_plots")

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

GROUP_COLORS = {
    "signal": "#2a9d8f",
    "junk": "#adb5bd",
    "spurious": "#e76f51",
}


def read_results(path):
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
    return list(dict.fromkeys(row[key] for row in rows))


def label_scenario(scenario):
    return SCENARIO_LABELS.get(scenario, scenario.replace("_", "\n"))


def save_figure(fig, output_dir, filename):
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_accuracy_by_scenario(rows, output_dir):
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


def plot_selected_feature_groups(rows, output_dir):
    selection_rows = [row for row in rows if row["signal_selected_mean"] is not None]
    scenarios = ordered_values(selection_rows, "scenario")
    methods = ordered_values(selection_rows, "method")
    by_pair = {(row["scenario"], row["method"]): row for row in selection_rows}

    fig, axes = plt.subplots(len(scenarios), 1, figsize=(11, 10), sharex=True)
    if len(scenarios) == 1:
        axes = [axes]

    for ax, scenario in zip(axes, scenarios):
        signal = []
        junk = []
        spurious = []
        for method in methods:
            row = by_pair[(scenario, method)]
            signal.append(row["signal_selected_mean"] or 0)
            junk.append(row["junk_selected_mean"] or 0)
            spurious.append(row["spurious_selected_mean"] or 0)

        x = np.arange(len(methods))
        ax.bar(x, signal, color=GROUP_COLORS["signal"], label="Signal")
        ax.bar(x, junk, bottom=signal, color=GROUP_COLORS["junk"], label="Junk")
        ax.bar(
            x,
            spurious,
            bottom=np.array(signal) + np.array(junk),
            color=GROUP_COLORS["spurious"],
            label="Spurious",
        )
        ax.set_ylabel(label_scenario(scenario), rotation=0, ha="right", va="center")
        ax.set_ylim(0, max(row["k"] or 256 for row in selection_rows) * 1.04)
        ax.grid(axis="y", alpha=0.2)

    axes[-1].set_xticks(np.arange(len(methods)))
    axes[-1].set_xticklabels(methods, rotation=25, ha="right")
    axes[0].set_title("Selected Feature Groups")
    axes[0].legend(ncol=3, frameon=False, loc="upper right")
    fig.text(0.02, 0.5, "Scenario", rotation=90, va="center")
    fig.text(0.5, 0.02, "Feature-selection method", ha="center")
    fig.tight_layout(rect=(0.04, 0.04, 1, 1))
    return save_figure(fig, output_dir, "synthetic_selected_feature_groups.png")


def plot_signal_recovery(rows, output_dir):
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


def plot_recovery_vs_accuracy(rows, output_dir):
    selection_rows = [row for row in rows if row["signal_recall_mean"] is not None]
    methods = ordered_values(selection_rows, "method")

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for method in methods:
        method_rows = [row for row in selection_rows if row["method"] == method]
        x = [row["signal_recall_mean"] for row in method_rows]
        y = [row["accuracy_mean"] for row in method_rows]
        ax.scatter(
            x,
            y,
            s=85,
            label=method,
            color=METHOD_COLORS.get(method),
            edgecolor="white",
            linewidth=0.7,
            alpha=0.9,
        )

    ax.set_title("Feature Recovery vs Classification Accuracy")
    ax.set_xlabel("Signal recall")
    ax.set_ylabel("Test accuracy")
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return save_figure(fig, output_dir, "synthetic_recovery_vs_accuracy.png")


def plot_spurious_selection_vs_accuracy(rows, output_dir):
    spurious_rows = [
        row
        for row in rows
        if row["scenario"] in {"train_only_spurious", "anti_spurious_test"}
        and row["spurious_selected_mean"] is not None
    ]
    methods = ordered_values(spurious_rows, "method")

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for method in methods:
        method_rows = [row for row in spurious_rows if row["method"] == method]
        x = [row["spurious_selected_mean"] for row in method_rows]
        y = [row["accuracy_mean"] for row in method_rows]
        ax.scatter(
            x,
            y,
            s=85,
            label=method,
            color=METHOD_COLORS.get(method),
            edgecolor="white",
            linewidth=0.7,
            alpha=0.9,
        )
        for row in method_rows:
            ax.annotate(
                "anti" if row["scenario"] == "anti_spurious_test" else "random",
                (row["spurious_selected_mean"], row["accuracy_mean"]),
                textcoords="offset points",
                xytext=(5, 4),
                fontsize=7,
            )

    ax.set_title("Shortcut Feature Selection vs Accuracy")
    ax.set_xlabel("Mean spurious features selected")
    ax.set_ylabel("Test accuracy")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return save_figure(fig, output_dir, "synthetic_spurious_selection_vs_accuracy.png")


def sorted_indices_by_class(data):
    labels = data.y.numpy()
    return np.argsort(labels)


def plot_adjacency_heatmap(scenario, output_dir, seed=0):
    data = make_scenario(scenario, seed=seed)[0]
    order = sorted_indices_by_class(data)
    inverse = np.empty_like(order)
    inverse[order] = np.arange(len(order))

    adjacency = np.zeros((data.num_nodes, data.num_nodes), dtype=np.float32)
    edges = data.edge_index.numpy()
    adjacency[inverse[edges[0]], inverse[edges[1]]] = 1.0

    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.imshow(adjacency, cmap="Greys", interpolation="nearest", vmin=0, vmax=1)
    ax.set_title(f"Adjacency Heatmap: {scenario}")
    ax.set_xlabel("Nodes sorted by class")
    ax.set_ylabel("Nodes sorted by class")

    nodes_per_class = data.synthetic_config.nodes_per_class
    for boundary in range(nodes_per_class, data.num_nodes, nodes_per_class):
        ax.axhline(boundary - 0.5, color="#d95f02", linewidth=0.8)
        ax.axvline(boundary - 0.5, color="#d95f02", linewidth=0.8)

    fig.tight_layout()
    return save_figure(fig, output_dir, f"synthetic_adjacency_{scenario}.png")


def plot_feature_matrix_heatmap(scenario, output_dir, seed=0, max_junk_features=60):
    data = make_scenario(scenario, seed=seed)[0]
    order = sorted_indices_by_class(data)
    x_clean = data.x_clean.numpy()[order]

    groups = data.feature_groups
    selected_columns = (
        groups["signal"]
        + groups["junk"][:max_junk_features]
        + groups["spurious"]
    )
    matrix = x_clean[:, selected_columns]

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.imshow(matrix, cmap="viridis", interpolation="nearest", aspect="auto", vmin=0, vmax=1)
    ax.set_title(f"Clean Feature Matrix: {scenario}")
    ax.set_xlabel("Feature columns: signal | junk sample | spurious")
    ax.set_ylabel("Nodes sorted by class")

    signal_end = len(groups["signal"])
    junk_end = signal_end + min(max_junk_features, len(groups["junk"]))
    for boundary in [signal_end, junk_end]:
        if 0 < boundary < matrix.shape[1]:
            ax.axvline(boundary - 0.5, color="white", linewidth=1.2)

    nodes_per_class = data.synthetic_config.nodes_per_class
    for boundary in range(nodes_per_class, data.num_nodes, nodes_per_class):
        ax.axhline(boundary - 0.5, color="white", linewidth=0.8)

    fig.tight_layout()
    return save_figure(fig, output_dir, f"synthetic_feature_matrix_{scenario}.png")


def plot_spurious_train_test_correlation(output_dir, seed=0):
    scenarios = ["train_only_spurious", "anti_spurious_test"]
    splits = ["train", "val", "test"]
    values = []

    for scenario in scenarios:
        data = make_scenario(scenario, seed=seed)[0]
        spurious = data.feature_groups["spurious"]
        if not spurious:
            values.append([0.0, 0.0, 0.0])
            continue

        x = data.x_clean[:, spurious].numpy()
        y = data.y.numpy()
        scenario_values = []
        for split in splits:
            mask = getattr(data, f"{split}_mask").numpy()
            split_x = x[mask]
            split_y = y[mask]
            correlations = []
            for feature_idx in range(split_x.shape[1]):
                if np.std(split_x[:, feature_idx]) == 0 or np.std(split_y) == 0:
                    correlations.append(0.0)
                else:
                    correlations.append(np.corrcoef(split_x[:, feature_idx], split_y)[0, 1])
            scenario_values.append(float(np.mean(correlations)))
        values.append(scenario_values)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(scenarios))
    width = 0.22
    colors = ["#2a9d8f", "#457b9d", "#e76f51"]
    for i, split in enumerate(splits):
        ax.bar(x + (i - 1) * width, [row[i] for row in values], width, label=split, color=colors[i])

    ax.axhline(0, color="#2e3440", linewidth=1)
    ax.set_title("Spurious Feature Correlation by Split")
    ax.set_ylabel("Mean signed correlation with label")
    ax.set_xticks(x)
    ax.set_xticklabels([label_scenario(scenario) for scenario in scenarios])
    ax.set_ylim(-1, 1)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    return save_figure(fig, output_dir, "synthetic_spurious_correlation.png")


def make_all_plots(results_csv=DEFAULT_RESULTS, output_dir=DEFAULT_OUTPUT_DIR, scenario_for_heatmaps="train_only_spurious"):
    rows = read_results(results_csv)
    generated = [
        plot_accuracy_by_scenario(rows, output_dir),
        plot_accuracy_heatmap(rows, output_dir),
        plot_method_family_summary(rows, output_dir),
        plot_graph_aware_gain(rows, output_dir),
        plot_selected_feature_groups(rows, output_dir),
        plot_signal_recovery(rows, output_dir),
        plot_recovery_vs_accuracy(rows, output_dir),
        plot_spurious_selection_vs_accuracy(rows, output_dir),
        plot_spurious_train_test_correlation(output_dir),
    ]
    return generated


def main():
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
