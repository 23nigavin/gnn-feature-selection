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

def save_results_to_csv(noise_type, noise_levels, results_by_method, filename):
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["noise_type", "noise_level", "noise_percent", "method", "accuracy"])

        for method, accuracies in results_by_method.items():
            for noise_level, accuracy in zip(noise_levels, accuracies):
                writer.writerow([noise_type, noise_level, 100 * noise_level, method, accuracy])

def run_noise_sweep(dataset, original_num_features, noise_levels, noise_type, seeds):
    results_by_method = {"No selection": [], "Raw l1": [], "Graph-aware l1": [], "Raw mutual_info": [], "Graph-aware mutual_info": [], "PCA": [], "Autoencoder": [], "Learned mask": []}

    if noise_type == "dense_junk":
        k = original_num_features
    else:
        k = 500

    for noise_ratio in noise_levels:
        print(f"Running noise_type={noise_type}, noise_ratio={noise_ratio}")
        results_by_method["No selection"].append(run_no_selection_baseline_avg(dataset, noise_ratio=noise_ratio, noise_type=noise_type, seeds=seeds))
        results_by_method["Raw l1"].append(run_preprocessing_selection_experiment_avg(dataset, noise_ratio=noise_ratio, noise_type=noise_type, k=k, selection_method="l1", graph_aware=False, seeds=seeds))
        results_by_method["Graph-aware l1"].append(run_preprocessing_selection_experiment_avg(dataset, noise_ratio=noise_ratio, noise_type=noise_type, k=k, selection_method="l1", graph_aware=True, seeds=seeds))
        results_by_method["Raw mutual_info"].append(run_preprocessing_selection_experiment_avg(dataset, noise_ratio=noise_ratio, noise_type=noise_type, k=k, selection_method="mutual_info", graph_aware=False, seeds=seeds))
        results_by_method["Graph-aware mutual_info"].append(run_preprocessing_selection_experiment_avg(dataset, noise_ratio=noise_ratio, noise_type=noise_type, k=k, selection_method="mutual_info", graph_aware=True, seeds=seeds))
        results_by_method["PCA"].append(run_pca_experiment_avg(dataset, noise_ratio=noise_ratio, noise_type=noise_type, n_components=256, seeds=seeds))
        results_by_method["Autoencoder"].append(run_autoencoder_experiment_avg(dataset, noise_ratio=noise_ratio, noise_type=noise_type, latent_dim=256, seeds=seeds))
        results_by_method["Learned mask"].append(run_learned_mask_experiment_avg(dataset, noise_ratio=noise_ratio, noise_type=noise_type, mask_lambda=0.0, k=k, seeds=seeds))

    return results_by_method

def main():
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

        results_by_method = run_noise_sweep(dataset, original_num_features, noise_levels, noise_type, seeds)

        print("Noise type:", noise_type)
        print("Noise levels:", noise_levels)
        print(results_by_method)
        save_results_to_csv(noise_type, noise_levels, results_by_method, filename=f"results_{noise_type}.csv")

        plot_method_accuracy_vs_noise(noise_percent, results_by_method, filename=f"accuracy_vs_{noise_type}.png", title=f"Accuracy vs. noise level ({noise_type})")

if __name__ == "__main__":
    main()