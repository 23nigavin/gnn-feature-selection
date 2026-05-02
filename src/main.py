import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
import numpy as np
from sklearn.linear_model import LogisticRegression
from torch_geometric.utils import add_self_loops, degree
import matplotlib.pyplot as plt

def main():
    dataset = Planetoid(root="data/Planetoid", name="Cora")
    original_num_features = dataset[0].x.shape[1]
    seeds = [0, 1, 2, 3, 4]

    noise_levels = [0.0, .25, .5, .75, 1.0, 1.25, 1.5]
    noise_percent = [100 * n for n in noise_levels]
    clean_baseline_acc = run_l1_selection_experiment_avg(dataset, noise_ratio=0.0, use_feature_selection=False, seeds=seeds)    
    noise_no_selection_acc = []
    noise_with_selection_acc = []

    for noise_ratio in noise_levels:
        acc_no_selection = run_l1_selection_experiment_avg(
            dataset,
            noise_ratio=noise_ratio,
            use_feature_selection=False,
            seeds=seeds
        )
        acc_with_selection = run_l1_selection_experiment_avg(
            dataset,
            noise_ratio=noise_ratio,
            use_feature_selection=True,
            k=original_num_features,
            seeds=seeds
        )

        noise_no_selection_acc.append(acc_no_selection)
        noise_with_selection_acc.append(acc_with_selection)

    print("Noise levels:", noise_levels)
    print("No noise, no feature selection:", clean_baseline_acc)
    print("With noise, no feature selection:", noise_no_selection_acc)
    print("With noise, with feature selection:", noise_with_selection_acc)

    plot_accuracy_vs_noise(noise_percent, clean_baseline_acc, noise_no_selection_acc, noise_with_selection_acc)

    k_values = [100, 300, 500, 700, 1000, original_num_features]
    fixed_noise_ratio = 1.0

    corrupted_baseline_acc = run_l1_selection_experiment_avg(
        dataset,
        noise_ratio=fixed_noise_ratio,
        use_feature_selection=False,
        seeds=seeds
    )

    k_selection_acc = []

    for k in k_values:
        acc = run_l1_selection_experiment_avg(
            dataset,
            noise_ratio=fixed_noise_ratio,
            use_feature_selection=True,
            k=k,
            seeds=seeds
        )
        k_selection_acc.append(acc)

    print("k values:", k_values)
    print("Clean baseline:", clean_baseline_acc)
    print("Corrupted baseline:", corrupted_baseline_acc)
    print("With noise, with feature selection over k:", k_selection_acc)

    plot_accuracy_vs_k(k_values, clean_baseline_acc, corrupted_baseline_acc, k_selection_acc)

if __name__ == "__main__":
    main()