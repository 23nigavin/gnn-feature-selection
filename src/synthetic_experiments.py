import csv

import numpy as np

from experiments import (
    run_autoencoder_experiment_avg,
    run_no_selection_baseline_avg,
    run_preprocessing_selection_experiment_avg,
)
from preprocessing_selection import select_features_mutual_info, select_top_k_features_l1
from synthetic_data import make_scenario
from util import aggregate_features


SCENARIOS_TO_RUN = [
    "weak_features_strong_graph",
    "weak_graph_strong_features",
    "few_signal_features",
    "train_only_spurious",
    "anti_spurious_test",
]

FEATURE_BUDGET = 256

# These are the methods we compare on each synthetic graph.
# The main comparison we care about is raw feature selection vs graph-aware feature selection, but the baseline and autoencoder give useful context.
METHODS_TO_RUN = [
    "No selection",
    "Raw l1",
    "Graph-aware l1",
    "Raw mutual_info",
    "Graph-aware mutual_info",
    "Autoencoder",
]

SELECTION_METHODS = {
    "Raw l1": {"selection_method": "l1", "graph_aware": False},
    "Graph-aware l1": {"selection_method": "l1", "graph_aware": True},
    "Raw mutual_info": {"selection_method": "mutual_info", "graph_aware": False},
    "Graph-aware mutual_info": {"selection_method": "mutual_info", "graph_aware": True},
}

SELECTION_FUNCTIONS = {
    "l1": select_top_k_features_l1,
    "mutual_info": select_features_mutual_info,
}


def selected_feature_budget(data):
    """
    Use the project-wide feature budget when the graph has enough columns.

    Some synthetic scenarios have fewer than 256 total features, so we cap k at
    the available feature count to avoid asking selectors for impossible output.
    """
    return min(FEATURE_BUDGET, data.x.shape[1])


def run_method(dataset, method, k, seed):
    # Each branch calls the existing experiment code.
    if method == "No selection":
        return run_no_selection_baseline_avg(
            dataset,
            noise_ratio=0.0,
            noise_type="dense_junk",
            seeds=[seed],
        )

    if method in SELECTION_METHODS:
        config = SELECTION_METHODS[method]
        return run_preprocessing_selection_experiment_avg(
            dataset,
            noise_ratio=0.0,
            noise_type="dense_junk",
            k=k,
            selection_method=config["selection_method"],
            graph_aware=config["graph_aware"],
            seeds=[seed],
        )

    if method == "Autoencoder":
        return run_autoencoder_experiment_avg(
            dataset,
            noise_ratio=0.0,
            noise_type="dense_junk",
            latent_dim=min(64, dataset[0].x.shape[1]),
            seeds=[seed],
        )

    raise ValueError(f"Unknown method: {method}")


def selected_indices_for_method(data, method, k):
    """
    Re-run just the feature scoring step so we can see what columns were picked.

    This is not used for training. It is only for measuring whether feature
    selection chose true signal features or got distracted by junk/spurious ones.
    """
    if method in SELECTION_METHODS:
        config = SELECTION_METHODS[method]
        selection_fn = SELECTION_FUNCTIONS[config["selection_method"]]
        scoring_features = aggregate_features(data.x, data.edge_index) if config["graph_aware"] else data.x
        return selection_fn(scoring_features, data.y, data.train_mask, k)

    return None


def feature_recovery_metrics(data, selected_indices):
    """Count how many selected columns came from signal, junk, or spurious groups."""
    if selected_indices is None:
        return {
            "signal_selected": None,
            "junk_selected": None,
            "spurious_selected": None,
            "signal_recall": None,
            "signal_precision": None,
        }

    selected = set(int(i) for i in selected_indices)
    signal = set(data.feature_groups["signal"])
    junk = set(data.feature_groups["junk"])
    spurious = set(data.feature_groups["spurious"])

    signal_selected = len(selected & signal)
    junk_selected = len(selected & junk)
    spurious_selected = len(selected & spurious)

    return {
        "signal_selected": signal_selected,
        "junk_selected": junk_selected,
        "spurious_selected": spurious_selected,
        "signal_recall": signal_selected / max(1, len(signal)),
        "signal_precision": signal_selected / max(1, len(selected)),
    }


def mean_or_none(values):
    values = [value for value in values if value is not None]
    if not values:
        return None
    return float(np.mean(values))


def run_synthetic_scenarios(
    scenarios=None,
    methods=None,
    seeds=None,
    output_csv="synthetic_results.csv",
):
    if scenarios is None:
        scenarios = SCENARIOS_TO_RUN
    if methods is None:
        methods = METHODS_TO_RUN
    if seeds is None:
        seeds = [0, 1, 2]

    rows = []

    # For each synthetic assumption, run every method across a few random seeds.
    for scenario in scenarios:
        for method in methods:
            accuracies = []
            recovery_by_seed = []
            metadata = None

            for seed in seeds:
                # Rebuild the synthetic graph with this seed so results are not
                # tied to one lucky/random graph draw.
                dataset = make_scenario(scenario, seed=seed)
                data = dataset[0]
                k = selected_feature_budget(data)
                metadata = {
                    "k": k,
                    "requested_k": FEATURE_BUDGET,
                    "num_features": data.x.shape[1],
                    "num_signal_features": len(data.feature_groups["signal"]),
                    "num_junk_features": len(data.feature_groups["junk"]),
                    "num_spurious_features": len(data.feature_groups["spurious"]),
                    "p_in": data.assumptions["p_in"],
                    "p_out": data.assumptions["p_out"],
                }

                print(f"Running synthetic scenario={scenario}, method={method}, seed={seed}, k={k}")
                mean_acc, std_acc = run_method(dataset, method, k, seed)
                accuracies.append(mean_acc)

                selected_indices = selected_indices_for_method(data, method, k)
                recovery = feature_recovery_metrics(data, selected_indices)
                recovery_by_seed.append(recovery)

                print({"accuracy": mean_acc, "within_seed_std": std_acc})

            row = {
                "scenario": scenario,
                "method": method,
                "num_synthetic_seeds": len(seeds),
                "accuracy_mean": float(np.mean(accuracies)),
                "accuracy_std": float(np.std(accuracies)),
                "signal_selected_mean": mean_or_none([r["signal_selected"] for r in recovery_by_seed]),
                "junk_selected_mean": mean_or_none([r["junk_selected"] for r in recovery_by_seed]),
                "spurious_selected_mean": mean_or_none([r["spurious_selected"] for r in recovery_by_seed]),
                "signal_recall_mean": mean_or_none([r["signal_recall"] for r in recovery_by_seed]),
                "signal_precision_mean": mean_or_none([r["signal_precision"] for r in recovery_by_seed]),
                **metadata,
            }
            rows.append(row)
            print(row)

    save_synthetic_results(rows, output_csv)
    return rows


def save_synthetic_results(rows, filename):
    if not rows:
        return

    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved synthetic results to {filename}")


def main():
    run_synthetic_scenarios()


if __name__ == "__main__":
    main()
