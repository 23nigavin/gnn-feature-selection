from torch_geometric.datasets import Planetoid

from experiments import (
    run_no_selection_baseline_avg,
    run_preprocessing_selection_experiment_avg,
    run_pca_experiment_avg,
    run_autoencoder_experiment_avg,
    run_learned_mask_experiment_avg,
)
from plotting import plot_method_accuracy_vs_noise, plot_accuracy_vs_k


def main():
    dataset = Planetoid(root="data/Planetoid", name="Cora")
    original_num_features = dataset[0].x.shape[1]
    seeds = [0, 1, 2, 3, 4]
    # seeds = [0]

    noise_levels = [0.0, .25, .5, .75, 1.0, 1.25, 1.5]
    # noise_levels = [0.0, 1.0]
    noise_percent = [100 * n for n in noise_levels]

    results_by_method = {"No selection": [], "Graph-aware L1": [], "PCA": [], "Autoencoder": [], "Learned mask": []}

    for noise_ratio in noise_levels:
        print(f"Running noise_ratio={noise_ratio}")

        results_by_method["No selection"].append(
            run_no_selection_baseline_avg(
                dataset,
                noise_ratio=noise_ratio,
                seeds=seeds,
            )
        )
        results_by_method["Graph-aware L1"].append(
            run_preprocessing_selection_experiment_avg(
                dataset,
                noise_ratio=noise_ratio,
                k=original_num_features,
                selection_method="l1",
                seeds=seeds,
            )
        )
        results_by_method["PCA"].append(
            run_pca_experiment_avg(
                dataset,
                noise_ratio=noise_ratio,
                n_components=64,
                seeds=seeds,
            )
        )
        results_by_method["Autoencoder"].append(
            run_autoencoder_experiment_avg(
                dataset,
                noise_ratio=noise_ratio,
                latent_dim=256,
                seeds=seeds,
            )
        )
        results_by_method["Learned mask"].append(
            run_learned_mask_experiment_avg(
                dataset,
                noise_ratio=noise_ratio,
                mask_lambda=0.0,
                k=original_num_features,
                seeds=seeds,
            )
        )

    print("Noise levels:", noise_levels)
    print(results_by_method)

    plot_method_accuracy_vs_noise(noise_percent, results_by_method, filename="accuracy_vs_noise_all_methods.png")

    # k_values = [100, 300, 500, 700, 1000, original_num_features]
    k_values = [100]
    fixed_noise_ratio = 1.0

    clean_baseline_acc = run_no_selection_baseline_avg(
        dataset,
        noise_ratio=0.0,
        seeds=seeds,
    )

    corrupted_baseline_acc = run_no_selection_baseline_avg(
        dataset,
        noise_ratio=fixed_noise_ratio,
        seeds=seeds,
    )

    k_selection_acc = []

    for k in k_values:
        acc = run_preprocessing_selection_experiment_avg(
            dataset,
            noise_ratio=fixed_noise_ratio,
            k=k,
            selection_method="l1",
            seeds=seeds,
        )
        k_selection_acc.append(acc)

    print("k values:", k_values)
    print("Clean baseline:", clean_baseline_acc)
    print("Corrupted baseline:", corrupted_baseline_acc)
    print("Graph-aware L1 over k:", k_selection_acc)

    plot_accuracy_vs_k(k_values, clean_baseline_acc, corrupted_baseline_acc, k_selection_acc)

if __name__ == "__main__":
    main()