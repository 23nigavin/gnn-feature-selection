# Script to run experiments
import numpy as np
import torch

from gnn import GCN, train, test
from noise import add_junk_features
from preprocessing_selection import select_top_k_features_l1
from util import aggregate_features

def run_l1_selection_experiment(dataset, noise_ratio=1.0, use_feature_selection=True, k=None, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

    graph = dataset[0].clone()
    feature_matrix = graph.x
    num_nodes, num_features = feature_matrix.shape
    num_junk_features = int(num_features * noise_ratio)

    if num_junk_features > 0:
        feature_matrix = add_junk_features(feature_matrix, num_junk_features)
        graph.x = feature_matrix
    else:
        graph.x = feature_matrix

    if use_feature_selection:
        if k is None:
            k = num_features

        x_agg = aggregate_features(graph.x, graph.edge_index)
        selected_indices = select_top_k_features_l1(x_agg, y=graph.y, train_mask=graph.train_mask, k=k)
        graph.x = graph.x[:, selected_indices]

    model = GCN(num_features=graph.x.shape[1], hidden_dim=16, num_classes=dataset.num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    for epoch in range(1, 201):
        loss = train(model, graph, optimizer)

    accuracy = test(model, graph)
    return accuracy

def run_l1_selection_experiment_avg(dataset, noise_ratio=1.0, use_feature_selection=True, k=None, seeds=None):
    if seeds is None:
        seeds = [0, 1, 2, 3, 4]

    accuracies = []
    for seed in seeds:
        acc = run_l1_selection_experiment(
            dataset,
            noise_ratio=noise_ratio,
            use_feature_selection=use_feature_selection,
            k=k,
            seed=seed
        )
        accuracies.append(acc)

    return float(np.mean(accuracies))