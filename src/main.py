import csv
from torch_geometric.datasets import Planetoid
from experiments import (
    run_no_selection_baseline_avg,
    run_preprocessing_selection_experiment_avg,
    run_pca_experiment_avg,
    run_autoencoder_experiment_avg,
    run_learned_mask_experiment_avg,
)
from plotting import plot_method_accuracy_vs_noise, plot_accuracy_vs_k

def save_results_to_csv(noise_type, noise_levels, results_by_method, std_by_method, filename):
    """
    Save the results to a CSV file.
    """
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["noise_type", "noise_level", "noise_percent", "method", "accuracy_mean", "accuracy_std"])

        for method, accuracies in results_by_method.items():
            for noise_level, accuracy, std in zip(noise_levels, accuracies, std_by_method[method]):
                writer.writerow([noise_type, noise_level, 100 * noise_level, method, accuracy, std])

def add_result(results_by_method, std_by_method, method, result):
    """    
    Helper function to add results to the results_by_method and std_by_method dictionaries.
    """
    mean, std = result
    results_by_method[method].append(mean)
    std_by_method[method].append(std)

def run_noise_sweep(dataset, original_num_features, noise_levels, noise_type, seeds):
    """
    Run the main noise sweep experiment, which varies the noise level for a given noise type and evaluates all methods at each noise level.
    Returns dictionaries of results and standard deviations by method.
    """
    results_by_method = {"No selection": [], "Raw l1": [], "Graph-aware l1": [], "Raw mutual_info": [], "Graph-aware mutual_info": [], "PCA": [], "Autoencoder": [], "Learned mask": []}
    std_by_method = {"No selection": [], "Raw l1": [], "Graph-aware l1": [], "Raw mutual_info": [], "Graph-aware mutual_info": [], "PCA": [], "Autoencoder": [], "Learned mask": []}

    k = 256

    for noise_ratio in noise_levels:
        print(f"Running noise_type={noise_type}, noise_ratio={noise_ratio}")
        add_result(results_by_method, std_by_method, "No selection", run_no_selection_baseline_avg(dataset, noise_ratio=noise_ratio, noise_type=noise_type, seeds=seeds))
        add_result(results_by_method, std_by_method, "Raw l1", run_preprocessing_selection_experiment_avg(dataset, noise_ratio=noise_ratio, noise_type=noise_type, k=k, selection_method="l1", graph_aware=False, seeds=seeds))
        add_result(results_by_method, std_by_method, "Graph-aware l1", run_preprocessing_selection_experiment_avg(dataset, noise_ratio=noise_ratio, noise_type=noise_type, k=k, selection_method="l1", graph_aware=True, seeds=seeds))
        add_result(results_by_method, std_by_method, "Raw mutual_info", run_preprocessing_selection_experiment_avg(dataset, noise_ratio=noise_ratio, noise_type=noise_type, k=k, selection_method="mutual_info", graph_aware=False, seeds=seeds))
        add_result(results_by_method, std_by_method, "Graph-aware mutual_info", run_preprocessing_selection_experiment_avg(dataset, noise_ratio=noise_ratio, noise_type=noise_type, k=k, selection_method="mutual_info", graph_aware=True, seeds=seeds))
        add_result(results_by_method, std_by_method, "PCA", run_pca_experiment_avg(dataset, noise_ratio=noise_ratio, noise_type=noise_type, n_components=256, seeds=seeds))
        add_result(results_by_method, std_by_method, "Autoencoder", run_autoencoder_experiment_avg(dataset, noise_ratio=noise_ratio, noise_type=noise_type, latent_dim=256, seeds=seeds))
        add_result(results_by_method, std_by_method, "Learned mask", run_learned_mask_experiment_avg(dataset, noise_ratio=noise_ratio, noise_type=noise_type, mask_lambda=0.0, k=k, seeds=seeds))

    return results_by_method, std_by_method

def run_k_sweep(dataset, noise_type, noise_ratio, k_values, seeds):
    """
    Run the k-sweep experiment, which varies the number of selected features for a given noise type and noise level.
    Returns the accuracy of the clean baseline, the corrupted baseline, and the k-selection experiment.
    """
    clean_baseline_acc, _ = run_no_selection_baseline_avg(dataset, noise_ratio=0.0, noise_type=noise_type, seeds=seeds)
    corrupted_baseline_acc, _ = run_no_selection_baseline_avg(dataset, noise_ratio=noise_ratio, noise_type=noise_type, seeds=seeds)

    k_selection_acc = []

    for k in k_values:
        print(f"Running k sweep: noise_type={noise_type}, noise_ratio={noise_ratio}, k={k}")
        acc, std = run_preprocessing_selection_experiment_avg(dataset, noise_ratio=noise_ratio, noise_type=noise_type, k=k, selection_method="l1", graph_aware=True, seeds=seeds)
        k_selection_acc.append(acc)

    return clean_baseline_acc, corrupted_baseline_acc, k_selection_acc

def main():
    """
    Main entry point for running the experiments and generating the plots.
    """
    dataset = Planetoid(root="data/Planetoid", name="Cora")
    original_num_features = dataset[0].x.shape[1]
    
    seeds = [0, 1, 2]
    noise_configs = {
        "dense_junk": [0.0, .25, .5, .75, 1.0, 1.25, 1.5],
        "feature_dropout": [0.0, .05, .1, .2, .3],
        "bit_flip": [0.0, .01, .03, .05, .1],
    }

    for noise_type, noise_levels in noise_configs.items():
        noise_percent = [100 * n for n in noise_levels]

        results_by_method, std_by_method = run_noise_sweep(dataset, original_num_features, noise_levels, noise_type, seeds)

        print("Noise type:", noise_type)
        print("Noise levels:", noise_levels)
        print(results_by_method)
        print(std_by_method)
        save_results_to_csv(noise_type, noise_levels, results_by_method, std_by_method, filename=f"results_{noise_type}.csv")

        plot_method_accuracy_vs_noise(noise_percent, results_by_method, std_by_method=std_by_method, filename=f"accuracy_vs_{noise_type}.png", title=f"Accuracy vs. noise level ({noise_type})")

    # accuracy vs k
    k_values = [50, 100, 256, 500, 1000]
    clean_baseline_acc, corrupted_baseline_acc, k_selection_acc = run_k_sweep(dataset, noise_type="dense_junk", noise_ratio=1.0, k_values=k_values, seeds=seeds)

    plot_accuracy_vs_k(
        k_values,
        clean_baseline_acc,
        corrupted_baseline_acc,
        k_selection_acc,
    )

if __name__ == "__main__":
    main()