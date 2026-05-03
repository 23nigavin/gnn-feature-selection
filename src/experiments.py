# Script to run experiments
import numpy as np
import torch

from gnn import GCN, train, test
from noise import add_junk_features
from preprocessing_selection import (select_top_k_features_l1,select_features_permutation,select_features_correlation,select_features_mutual_info)
from util import aggregate_features
from autoencoder import BetterDenoisingAutoencoder, train_autoencoder, encode_features
from pca import apply_pca
from masked_gnn import MaskedGCN

def run_no_selection_baseline(dataset, noise_ratio=1.0, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

    graph = dataset[0].clone()
    feature_matrix = graph.x
    _, num_features = feature_matrix.shape
    num_junk_features = int(num_features * noise_ratio)

    if num_junk_features > 0:
        feature_matrix = add_junk_features(feature_matrix, num_junk_features)

    graph.x = feature_matrix

    model = GCN(num_features=graph.x.shape[1], hidden_dim=16, num_classes=dataset.num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    for epoch in range(1, 201):
        train(model, graph, optimizer)

    accuracy = test(model, graph)
    return accuracy

def run_no_selection_baseline_avg(dataset, noise_ratio=1.0, seeds=None):
    if seeds is None:
        seeds = [0, 1, 2, 3, 4]

    accuracies = []
    for seed in seeds:
        acc = run_no_selection_baseline(
            dataset,
            noise_ratio=noise_ratio,
            seed=seed,
        )
        accuracies.append(acc)

    return float(np.mean(accuracies))

def get_selection_fn(selection_method):
    if selection_method == "l1":
        return select_top_k_features_l1
    if selection_method == "permutation":
        return select_features_permutation
    if selection_method == "correlation":
        return select_features_correlation
    if selection_method == "mutual_info":
        return select_features_mutual_info

    raise ValueError(f"Unknown selection method: {selection_method}")

def run_preprocessing_selection_experiment(dataset,noise_ratio=1.0,k=None,selection_method="l1",seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

    graph = dataset[0].clone()
    feature_matrix = graph.x
    _, num_features = feature_matrix.shape
    num_junk_features = int(num_features * noise_ratio)

    if num_junk_features > 0:
        feature_matrix = add_junk_features(feature_matrix, num_junk_features)

    graph.x = feature_matrix

    if k is None:
        k = num_features

    selection_fn = get_selection_fn(selection_method)

    x_agg = aggregate_features(graph.x, graph.edge_index)
    selected_indices = selection_fn(x_agg, y=graph.y, train_mask=graph.train_mask, k=k)
    graph.x = graph.x[:, selected_indices]

    model = GCN(num_features=graph.x.shape[1], hidden_dim=16, num_classes=dataset.num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    for epoch in range(1, 201):
        train(model, graph, optimizer)

    accuracy = test(model, graph)
    return accuracy


def run_preprocessing_selection_experiment_avg(dataset,noise_ratio=1.0,k=None,selection_method="l1",seeds=None):
    if seeds is None:
        seeds = [0, 1, 2, 3, 4]

    accuracies = []
    for seed in seeds:
        acc = run_preprocessing_selection_experiment(
            dataset,
            noise_ratio=noise_ratio,
            k=k,
            selection_method=selection_method,
            seed=seed,
        )
        accuracies.append(acc)

    return float(np.mean(accuracies))

def run_autoencoder_experiment(
    dataset,
    noise_ratio=1.0,
    latent_dim=256,
    ae_hidden_dim=512,
    ae_dropout=0.3,
    ae_lr=1e-3,
    ae_weight_decay=1e-4,
    ae_epochs=200,
    ae_patience=20,
    seed=42,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    graph = dataset[0].clone()

    x_clean = graph.x
    _, num_features = x_clean.shape
    num_junk_features = int(num_features * noise_ratio)

    if num_junk_features > 0:
        x_noisy = add_junk_features(x_clean, num_junk_features)
    else:
        x_noisy = x_clean

    x_target = x_noisy

    ae_model = BetterDenoisingAutoencoder(
        input_dim=x_noisy.shape[1],
        latent_dim=latent_dim,
        target_dim=x_target.shape[1],
        hidden_dim=ae_hidden_dim,
        dropout=ae_dropout,
    )

    ae_mask = torch.ones_like(graph.train_mask, dtype=torch.bool)
    ae_model, history = train_autoencoder(
        model=ae_model,
        x_input=x_noisy,
        x_target=x_target,
        train_mask=ae_mask,
        val_mask=ae_mask,
        lr=ae_lr,
        weight_decay=ae_weight_decay,
        epochs=ae_epochs,
        patience=ae_patience,
        verbose=False,
        reconstruction_loss_type="bce",
    )

    z = encode_features(ae_model, x_noisy)
    graph.x = z

    model = GCN(
        num_features=graph.x.shape[1],
        hidden_dim=16,
        num_classes=dataset.num_classes,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    for epoch in range(1, 201):
        train(model, graph, optimizer)

    accuracy = test(model, graph)
    return accuracy

def run_autoencoder_experiment_avg(
    dataset,
    noise_ratio=1.0,
    latent_dim=256,
    seeds=None,
):
    if seeds is None:
        seeds = [0, 1, 2, 3, 4]

    accuracies = []
    for seed in seeds:
        acc = run_autoencoder_experiment(
            dataset,
            noise_ratio=noise_ratio,
            latent_dim=latent_dim,
            seed=seed,
        )
        accuracies.append(acc)

    return float(np.mean(accuracies))