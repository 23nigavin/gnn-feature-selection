# Script to run experiments
import numpy as np
import torch

from gnn import GCN, train, test
from noise import apply_noise
from preprocessing_selection import (select_top_k_features_l1, select_features_mutual_info)
from util import aggregate_features
from autoencoder import BetterDenoisingAutoencoder, train_autoencoder, encode_features
from pca import apply_pca
from masked_gnn import MaskedGCN

def mean_std(accuracies):
    # Small helper so every experiment reports results the same way.
    return float(np.mean(accuracies)), float(np.std(accuracies))

def run_no_selection_baseline(dataset, noise_ratio=1.0, noise_type="dense_junk", seed=42):
    """
    Run a GCN on the dataset with no feature selection, using the specified noise level and type.
    This serves as a baseline to compare against feature selection methods.
    Args:
        - dataset: The PyTorch Geometric dataset to use.
        - noise_ratio: The proportion of features to corrupt with noise.
        - noise_type: The type of noise to apply (e.g., "dense_junk").
        - seed: Random seed for reproducibility.
    Returns:
        - accuracy: The test accuracy of the GCN on the noisy dataset.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Work on a clone so the original dataset is not changed between runs.
    graph = dataset[0].clone()
    feature_matrix = graph.x

    # Add the chosen noise, then train the GCN on all resulting features.
    feature_matrix = apply_noise(feature_matrix, noise_type=noise_type, noise_level=noise_ratio)

    graph.x = feature_matrix

    model = GCN(num_features=graph.x.shape[1], hidden_dim=16, num_classes=dataset.num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    for epoch in range(1, 201):
        train(model, graph, optimizer)

    accuracy = test(model, graph)
    return accuracy

def run_no_selection_baseline_avg(dataset, noise_ratio=1.0, noise_type="dense_junk", seeds=None):
    """
    Run the no-selection baseline experiment multiple times with different random seeds and return the mean and standard deviation of the accuracies.
    Args:       
        - dataset: The PyTorch Geometric dataset to use.
        - noise_ratio: The proportion of features to corrupt with noise.
        - noise_type: The type of noise to apply (e.g., "dense_junk").
        - seeds: A list of random seeds to use for multiple runs. If None, defaults to [0, 1, 2, 3, 4].
    Returns:
        - A tuple (mean_accuracy, std_accuracy) representing the average test accuracy and its standard deviation across the runs.
    """
    if seeds is None:
        seeds = [0, 1, 2, 3, 4]

    # Run the same setup with different seeds and average the accuracies.
    accuracies = []
    for seed in seeds:
        acc = run_no_selection_baseline(
            dataset,
            noise_ratio=noise_ratio,
            noise_type=noise_type,
            seed=seed,
        )
        accuracies.append(acc)

    return mean_std(accuracies)

def get_selection_fn(selection_method):
    # Basic function to map from a method name to the corresponding feature selection function.
    if selection_method == "l1":
        return select_top_k_features_l1
    if selection_method == "mutual_info":
        return select_features_mutual_info

    raise ValueError(f"Unknown selection method: {selection_method}")

def run_preprocessing_selection_experiment(dataset, noise_ratio=1.0, noise_type="dense_junk", k=None, selection_method="l1", graph_aware=True, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

    # First corrupt the feature matrix, then select a smaller set of columns.
    graph = dataset[0].clone()
    feature_matrix = graph.x
    _, num_features = feature_matrix.shape
    feature_matrix = apply_noise(feature_matrix, noise_type=noise_type, noise_level=noise_ratio)

    graph.x = feature_matrix

    if k is None:
        k = num_features

    selection_fn = get_selection_fn(selection_method)

    # Graph-aware selection scores features after one neighborhood aggregation step.
    if graph_aware:
        scoring_features = aggregate_features(graph.x, graph.edge_index)
    else:
        scoring_features = graph.x

    # The selected indices are used on the original noisy features, not on the aggregated copy.
    selected_indices = selection_fn(scoring_features, y=graph.y, train_mask=graph.train_mask, k=k)
    graph.x = graph.x[:, selected_indices]

    model = GCN(num_features=graph.x.shape[1], hidden_dim=16, num_classes=dataset.num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    for epoch in range(1, 201):
        train(model, graph, optimizer)

    accuracy = test(model, graph)
    return accuracy

def run_preprocessing_selection_experiment_avg(dataset, noise_ratio=1.0, noise_type="dense_junk", k=None, selection_method="l1", graph_aware=True, seeds=None):
    """
    Run the preprocessing selection experiment multiple times with different random seeds and return the mean and standard deviation of the accuracies.
    Args:
        - dataset: The PyTorch Geometric dataset to use.
        - noise_ratio: The proportion of features to corrupt with noise.
        - noise_type: The type of noise to apply (e.g., "dense_junk").
        - k: The number of features to select. If None, defaults to the total number of features.
        - selection_method: The feature selection method to use (e.g., "l1" or "mutual_info").
        - graph_aware: Whether to use graph-aware features for selection.
        - seeds: A list of random seeds to use for multiple runs. If None, defaults to [0, 1, 2, 3, 4].
    Returns:
        - A tuple (mean_accuracy, std_accuracy) representing the average test accuracy and its standard deviation across the runs.
    """
    if seeds is None:
        seeds = [0, 1, 2, 3, 4]

    # Average over seeds so one lucky initialization does not decide the result.
    accuracies = []
    for seed in seeds:
        acc = run_preprocessing_selection_experiment(
            dataset,
            noise_ratio=noise_ratio,
            noise_type=noise_type,
            k=k,
            selection_method=selection_method,
            graph_aware=graph_aware,
            seed=seed,
        )
        accuracies.append(acc)

    return mean_std(accuracies)

def run_autoencoder_experiment(
    dataset,
    noise_ratio=1.0,
    noise_type="dense_junk",
    latent_dim=256,
    ae_hidden_dim=512,
    ae_dropout=0.3,
    ae_lr=1e-3,
    ae_weight_decay=1e-4,
    ae_epochs=200,
    ae_patience=20,
    seed=42,
):
    """
    Run the autoencoder feature learning experiment with the specified parameters and return the test accuracy.
    Args:
        - dataset: The PyTorch Geometric dataset to use.
        - noise_ratio: The proportion of features to corrupt with noise.
        - noise_type: The type of noise to apply (e.g., "dense_junk ").
        - latent_dim: The dimensionality of the autoencoder's latent space. Should be less than or equal to the number of input features.
        - ae_hidden_dim: The number of hidden units in the autoencoder's encoder and decoder layers.
        - ae_dropout: The dropout rate to apply in the autoencoder's encoder and decoder layers.
        - ae_lr: The learning rate for training the autoencoder.
        - ae_weight_decay: The weight decay (L2 regularization) for training the autoencoder.
        - ae_epochs: The maximum number of epochs to train the autoencoder.
        - ae_patience: The number of epochs with no improvement on the validation loss after which training will be stopped.
        - seed: Random seed for reproducibility.
    Returns:
        - accuracy: The test accuracy of the GCN trained on the autoencoder's latent features
    
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    graph = dataset[0].clone()

    # The autoencoder is trained on the noisy features and then its latent code is used by the GCN.
    x_clean = graph.x
    x_noisy = apply_noise(x_clean, noise_type=noise_type, noise_level=noise_ratio)
    x_target = x_noisy

    ae_model = BetterDenoisingAutoencoder(
        input_dim=x_noisy.shape[1],
        latent_dim=latent_dim,
        target_dim=x_target.shape[1],
        hidden_dim=ae_hidden_dim,
        dropout=ae_dropout,
    )

    ae_mask = torch.ones_like(graph.train_mask, dtype=torch.bool)

    # Train on all nodes because this is an unsupervised reconstruction task.
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

    # Replace the original features with the learned compressed representation.
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
    noise_type="dense_junk",
    latent_dim=256,
    seeds=None,
):
    if seeds is None:
        seeds = [0, 1, 2, 3, 4]

    # Repeat the autoencoder and GCN run for each seed.
    accuracies = []
    for seed in seeds:
        acc = run_autoencoder_experiment(
            dataset,
            noise_ratio=noise_ratio,
            noise_type=noise_type,
            latent_dim=latent_dim,
            seed=seed,
        )
        accuracies.append(acc)

    return mean_std(accuracies)

def run_pca_experiment(dataset, noise_ratio=1.0, noise_type="dense_junk", n_components=64, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

    # PCA is another compression baseline, but it is linear instead of neural.
    graph = dataset[0].clone()
    x_clean = graph.x
    x_noisy = apply_noise(x_clean, noise_type=noise_type, noise_level=noise_ratio)

    x_pca = apply_pca(feature_matrix=x_noisy, train_mask=graph.train_mask, n_components=n_components)
    graph.x = x_pca
    model = GCN(num_features=graph.x.shape[1], hidden_dim=16, num_classes=dataset.num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    for epoch in range(1, 201):
        train(model, graph, optimizer)

    accuracy = test(model, graph)
    return accuracy

def run_pca_experiment_avg(dataset, noise_ratio=1.0, noise_type="dense_junk", n_components=64, seeds=None):
    if seeds is None:
        seeds = [0, 1, 2, 3, 4]

    # Same PCA experiment, repeated for stability.
    accuracies = []
    for seed in seeds:
        acc = run_pca_experiment(
            dataset,
            noise_ratio=noise_ratio,
            noise_type=noise_type,
            n_components=n_components,
            seed=seed,
        )
        accuracies.append(acc)

    return mean_std(accuracies)

def run_learned_mask_experiment(dataset, noise_ratio=1.0, noise_type="dense_junk", mask_lambda=0.0, k=None, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

    # This method learns the feature mask while the GCN is training.
    graph = dataset[0].clone()
    feature_matrix = graph.x
    _, num_features = feature_matrix.shape
    if k is None:
        k = num_features

    feature_matrix = apply_noise(feature_matrix, noise_type=noise_type, noise_level=noise_ratio)

    graph.x = feature_matrix

    # MaskedGCN keeps only the top-k learned feature weights during the forward pass.
    model = MaskedGCN(num_features=graph.x.shape[1], hidden_dim=16, num_classes=dataset.num_classes, k=k)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    for epoch in range(1, 201):
        model.train()
        optimizer.zero_grad()

        out = model(graph.x, graph.edge_index)
        classification_loss = torch.nn.functional.cross_entropy(
            out[graph.train_mask],
            graph.y[graph.train_mask],
        )

        mask_penalty = model.mask_l1_penalty()
        # mask_lambda is currently usually 0, but this keeps the penalty option available.
        loss = classification_loss + mask_lambda * mask_penalty
        loss.backward()
        optimizer.step()

    accuracy = test(model, graph)
    return accuracy


def run_learned_mask_experiment_avg(dataset, noise_ratio=1.0, noise_type="dense_junk", mask_lambda=0.0, k=None, seeds=None):
    if seeds is None:
        seeds = [0, 1, 2, 3, 4]

    # Repeat the learned-mask run across seeds like the other methods.
    accuracies = []
    for seed in seeds:
        acc = run_learned_mask_experiment(
            dataset,
            noise_ratio=noise_ratio,
            noise_type=noise_type,
            mask_lambda=mask_lambda,
            k=k,
            seed=seed,
        )
        accuracies.append(acc)

    return mean_std(accuracies)
