import copy
import csv
import random
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


# Config

@dataclass
class Config:
    dataset_name: str = "Cora"         
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Noise / corruption
    junk_dims: int = 500
    mask_prob: float = 0.2
    noise_std: float = 0.1

    # AE
    ae_hidden_dim: int = 128
    latent_dim: int = 64
    ae_dropout: float = 0.3
    ae_lr: float = 1e-3
    ae_weight_decay: float = 1e-4
    ae_epochs: int = 200
    ae_patience: int = 20

    # GCN
    gcn_hidden_dim: int = 64
    gcn_lr: float = 1e-2
    gcn_weight_decay: float = 5e-4
    gcn_epochs: int = 300
    gcn_dropout: float = 0.5

    # Experiments
    junk_dim_list: tuple = (0, 100, 300, 500, 1000)
    seeds: tuple = (42, 52, 62)

    verbose: bool = True


# Utils

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == y).float().mean().item()


# Models

class BetterDenoisingAutoencoder(nn.Module):
    """
    Encoder sees noisy augmented input of size input_dim.
    Decoder reconstructs only original clean feature dimension of size target_dim.
    """
    def __init__(self, input_dim: int, latent_dim: int, target_dim: int,
                 hidden_dim: int = 128, dropout: float = 0.3):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, target_dim),
        )

    def forward(self, x_noisy: torch.Tensor):
        z = self.encoder(x_noisy)
        x_hat = self.decoder(z)
        return z, x_hat


class GCN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.5):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x


# Data

def load_dataset(dataset_name: str):
    dataset = Planetoid(root=f"data/{dataset_name}", name=dataset_name)
    return dataset, dataset[0]


def corrupt_features(
    x_clean: torch.Tensor,
    junk_dims: int,
    mask_prob: float,
    noise_std: float
):
    """
    Returns:
        x_noisy_aug : noisy input with junk dims
        x_clean     : original clean target (only original dimensions)
    """
    device = x_clean.device
    n, _ = x_clean.shape

    x_noisy = x_clean.clone()

    if mask_prob > 0:
        keep_mask = (torch.rand_like(x_noisy) > mask_prob).float()
        x_noisy = x_noisy * keep_mask

    if noise_std > 0:
        x_noisy = x_noisy + torch.randn_like(x_noisy) * noise_std

    if junk_dims > 0:
        junk = torch.randn(n, junk_dims, device=device)
        x_noisy_aug = torch.cat([x_noisy, junk], dim=1)
    else:
        x_noisy_aug = x_noisy

    return x_noisy_aug, x_clean


def aggregate_one_hop_features(x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    n = x.size(0)
    row, col = edge_index
    values = torch.ones(row.size(0), device=x.device)

    adj = torch.sparse_coo_tensor(
        indices=torch.stack([row, col], dim=0),
        values=values,
        size=(n, n)
    ).coalesce()

    self_idx = torch.arange(n, device=x.device)
    self_loops = torch.sparse_coo_tensor(
        indices=torch.stack([self_idx, self_idx], dim=0),
        values=torch.ones(n, device=x.device),
        size=(n, n)
    ).coalesce()

    adj = (adj + self_loops).coalesce()

    deg = torch.sparse.sum(adj, dim=1).to_dense().clamp(min=1).unsqueeze(1)
    x_agg = torch.sparse.mm(adj, x) / deg
    return x_agg


# AE training

def train_autoencoder(
    model,
    x_input,
    x_target,
    train_mask,
    val_mask,
    lr,
    weight_decay,
    epochs,
    patience,
    verbose=True
):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_state = None
    best_val_loss = float("inf")
    patience_counter = 0

    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        _, x_hat = model(x_input[train_mask])
        train_loss = F.mse_loss(x_hat, x_target[train_mask])
        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            _, x_hat_val = model(x_input[val_mask])
            val_loss = F.mse_loss(x_hat_val, x_target[val_mask])

        history["train_loss"].append(train_loss.item())
        history["val_loss"].append(val_loss.item())

        if val_loss.item() < best_val_loss - 1e-5:
            best_val_loss = val_loss.item()
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if verbose and (epoch == 1 or epoch % 25 == 0):
            print(
                f"[AE] Epoch {epoch:03d} | "
                f"Train Loss: {train_loss.item():.4f} | "
                f"Val Loss: {val_loss.item():.4f}"
            )

        if patience_counter >= patience:
            if verbose:
                print(f"[AE] Early stopping at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history


def encode_features(model, x_input):
    model.eval()
    with torch.no_grad():
        z = model.encoder(x_input)
        z = F.normalize(z, p=2, dim=1)   # helps stabilize downstream GCN
    return z


# GCN training

def train_gcn(
    model,
    x,
    edge_index,
    y,
    train_mask,
    val_mask,
    lr,
    weight_decay,
    epochs,
    verbose=True
):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_state = None
    best_val_acc = -1.0

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        out = model(x, edge_index)
        loss = F.cross_entropy(out[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            out_val = model(x, edge_index)
            val_acc = accuracy(out_val[val_mask], y[val_mask])

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())

        if verbose and (epoch == 1 or epoch % 25 == 0):
            train_acc = accuracy(out[train_mask], y[train_mask])
            print(
                f"[GCN] Epoch {epoch:03d} | "
                f"Loss: {loss.item():.4f} | "
                f"Train Acc: {train_acc:.4f} | "
                f"Val Acc: {val_acc:.4f}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


def evaluate_gcn(model, x, edge_index, y, mask):
    model.eval()
    with torch.no_grad():
        logits = model(x, edge_index)
        return accuracy(logits[mask], y[mask])


# Experiment runners

def run_baseline_noisy_gcn(data, config):
    x_noisy_aug, _ = corrupt_features(
        x_clean=data.x,
        junk_dims=config.junk_dims,
        mask_prob=config.mask_prob,
        noise_std=config.noise_std,
    )

    model = GCN(
        input_dim=x_noisy_aug.size(1),
        hidden_dim=config.gcn_hidden_dim,
        output_dim=int(data.y.max().item()) + 1,
        dropout=config.gcn_dropout,
    ).to(config.device)

    model = train_gcn(
        model=model,
        x=x_noisy_aug,
        edge_index=data.edge_index,
        y=data.y,
        train_mask=data.train_mask,
        val_mask=data.val_mask,
        lr=config.gcn_lr,
        weight_decay=config.gcn_weight_decay,
        epochs=config.gcn_epochs,
        verbose=config.verbose,
    )

    return evaluate_gcn(model, x_noisy_aug, data.edge_index, data.y, data.test_mask)


def run_dae_gcn(data, config, use_graph_aggregation=False):
    x_noisy_aug, x_clean_target = corrupt_features(
        x_clean=data.x,
        junk_dims=config.junk_dims,
        mask_prob=config.mask_prob,
        noise_std=config.noise_std,
    )

    ae_input = x_noisy_aug
    if use_graph_aggregation:
        ae_input = aggregate_one_hop_features(ae_input, data.edge_index)

    ae_model = BetterDenoisingAutoencoder(
        input_dim=ae_input.size(1),
        latent_dim=config.latent_dim,
        target_dim=data.x.size(1),
        hidden_dim=config.ae_hidden_dim,
        dropout=config.ae_dropout,
    ).to(config.device)

    ae_model, ae_history = train_autoencoder(
        model=ae_model,
        x_input=ae_input,
        x_target=x_clean_target,
        train_mask=data.train_mask,
        val_mask=data.val_mask,
        lr=config.ae_lr,
        weight_decay=config.ae_weight_decay,
        epochs=config.ae_epochs,
        patience=config.ae_patience,
        verbose=config.verbose,
    )

    z = encode_features(ae_model, ae_input)

    gcn_model = GCN(
        input_dim=z.size(1),
        hidden_dim=config.gcn_hidden_dim,
        output_dim=int(data.y.max().item()) + 1,
        dropout=config.gcn_dropout,
    ).to(config.device)

    gcn_model = train_gcn(
        model=gcn_model,
        x=z,
        edge_index=data.edge_index,
        y=data.y,
        train_mask=data.train_mask,
        val_mask=data.val_mask,
        lr=config.gcn_lr,
        weight_decay=config.gcn_weight_decay,
        epochs=config.gcn_epochs,
        verbose=config.verbose,
    )

    test_acc = evaluate_gcn(gcn_model, z, data.edge_index, data.y, data.test_mask)

    # Calculate final reconstruction error
    ae_model.eval()
    with torch.no_grad():
        _, x_hat = ae_model(ae_input[data.val_mask])
        final_reconstruction_mse = F.mse_loss(x_hat, x_clean_target[data.val_mask]).item()

    return test_acc, ae_history, final_reconstruction_mse


# Plotting / saving

def save_results_csv(results, filename="results.csv"):
    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "seed",
                "junk_dims",
                "baseline_acc",
                "dae_acc",
                "graph_dae_acc",
                "dae_mse",
                "graph_dae_mse",
            ],
        )
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved results to {filename}")


def plot_accuracy_vs_junk(results, filename="accuracy_vs_junkdims.png"):
    junk_values = sorted(list(set(r["junk_dims"] for r in results)))

    baseline_means = []
    dae_means = []
    graph_means = []

    baseline_stds = []
    dae_stds = []
    graph_stds = []

    for j in junk_values:
        subset = [r for r in results if r["junk_dims"] == j]

        baseline_vals = [r["baseline_acc"] for r in subset]
        dae_vals = [r["dae_acc"] for r in subset]
        graph_vals = [r["graph_dae_acc"] for r in subset]

        baseline_means.append(np.mean(baseline_vals))
        dae_means.append(np.mean(dae_vals))
        graph_means.append(np.mean(graph_vals))

        baseline_stds.append(np.std(baseline_vals))
        dae_stds.append(np.std(dae_vals))
        graph_stds.append(np.std(graph_vals))

    plt.figure(figsize=(8, 5))
    plt.plot(junk_values, baseline_means, marker="o", label="Baseline noisy GCN")
    plt.plot(junk_values, dae_means, marker="o", label="Vanilla DAE + GCN")
    plt.plot(junk_values, graph_means, marker="o", label="Graph-aware DAE + GCN")

    plt.fill_between(
        junk_values,
        np.array(baseline_means) - np.array(baseline_stds),
        np.array(baseline_means) + np.array(baseline_stds),
        alpha=0.15,
    )
    plt.fill_between(
        junk_values,
        np.array(dae_means) - np.array(dae_stds),
        np.array(dae_means) + np.array(dae_stds),
        alpha=0.15,
    )
    plt.fill_between(
        junk_values,
        np.array(graph_means) - np.array(graph_stds),
        np.array(graph_means) + np.array(graph_stds),
        alpha=0.15,
    )

    plt.xlabel("Number of junk features appended")
    plt.ylabel("Test accuracy")
    plt.title("Robustness to Feature Flooding on Cora")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()

    print(f"Saved figure to {filename}")


def plot_ae_history(history, filename="ae_loss_curve.png"):
    plt.figure(figsize=(7, 4.5))
    plt.plot(history["train_loss"], label="Train reconstruction loss")
    plt.plot(history["val_loss"], label="Validation reconstruction loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.title("Vanilla DAE training curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()
    print(f"Saved AE curve to {filename}")


def plot_feature_reconstruction_quality(results, filename="reconstruction_quality.png"):
    """Plot reconstruction MSE vs junk dimensions"""
    junk_values = sorted(list(set(r["junk_dims"] for r in results)))

    dae_mses = []
    graph_dae_mses = []

    for j in junk_values:
        subset = [r for r in results if r["junk_dims"] == j]
        dae_mses.append(np.mean([r["dae_mse"] for r in subset]))
        graph_dae_mses.append(np.mean([r["graph_dae_mse"] for r in subset]))

    plt.figure(figsize=(8, 5))
    plt.plot(junk_values, dae_mses, marker='o', color='blue', label='Vanilla DAE MSE')
    plt.plot(junk_values, graph_dae_mses, marker='s', color='green', label='Graph-aware DAE MSE')
    plt.xlabel("Number of junk features")
    plt.ylabel("Final validation reconstruction MSE")
    plt.title("Autoencoder Reconstruction Quality vs Noise Level")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()
    print(f"Saved reconstruction quality plot to {filename}")


def plot_latent_space_embeddings(data, z, filename="latent_embeddings.png"):
    """Visualize latent space using t-SNE"""
    # Use only test nodes for cleaner visualization
    test_indices = data.test_mask.nonzero().squeeze()
    z_test = z[test_indices].cpu().numpy()
    y_test = data.y[test_indices].cpu().numpy()

    # t-SNE for dimensionality reduction to 2D
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    z_2d = tsne.fit_transform(z_test)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(z_2d[:, 0], z_2d[:, 1], c=y_test, cmap='tab10', alpha=0.7, s=50)
    plt.colorbar(scatter, label='Class label')
    plt.xlabel("t-SNE dimension 1")
    plt.ylabel("t-SNE dimension 2")
    plt.title("Latent Space Embeddings (t-SNE) - Test Nodes")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()
    print(f"Saved latent space visualization to {filename}")


def plot_feature_importance_analysis(data, ae_model, x_input, filename="feature_importance.png"):
    """Analyze which original features are most reconstructed"""
    ae_model.eval()
    with torch.no_grad():
        _, x_hat = ae_model(x_input)

    # Calculate reconstruction error per feature dimension
    reconstruction_error = F.mse_loss(x_hat, data.x, reduction='none').mean(dim=0).cpu().numpy()

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.bar(range(len(reconstruction_error)), reconstruction_error, alpha=0.7)
    plt.xlabel("Feature dimension")
    plt.ylabel("Reconstruction MSE")
    plt.title("Reconstruction Error per Feature Dimension")
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    # Show distribution of reconstruction errors
    plt.hist(reconstruction_error, bins=30, alpha=0.7, edgecolor='black')
    plt.xlabel("Reconstruction MSE")
    plt.ylabel("Frequency")
    plt.title("Distribution of Feature Reconstruction Errors")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()
    print(f"Saved feature importance analysis to {filename}")


def plot_corruption_effect_analysis(data, config, filename="corruption_effects.png"):
    """Compare different types of corruption effects"""
    clean_features = data.x

    # Test different corruption levels
    corruption_levels = [0, 0.2, 0.5, 0.8]
    effects = []

    for level in corruption_levels:
        # Test masking effect
        x_masked, _ = corrupt_features(clean_features, 0, level, 0)
        mask_mse = F.mse_loss(x_masked, clean_features).item()

        # Test noise effect
        x_noisy, _ = corrupt_features(clean_features, 0, 0, level)
        noise_mse = F.mse_loss(x_noisy, clean_features).item()

        # Test junk features effect (normalized by feature count)
        x_junk, _ = corrupt_features(clean_features, int(clean_features.size(1) * level), 0, 0)
        junk_mse = F.mse_loss(x_junk[:, :clean_features.size(1)], clean_features).item()

        effects.append({
            'corruption_level': level,
            'mask_mse': mask_mse,
            'noise_mse': noise_mse,
            'junk_mse': junk_mse
        })

    levels = [e['corruption_level'] for e in effects]
    mask_mses = [e['mask_mse'] for e in effects]
    noise_mses = [e['noise_mse'] for e in effects]
    junk_mses = [e['junk_mse'] for e in effects]

    plt.figure(figsize=(10, 6))
    plt.plot(levels, mask_mses, marker='o', label='Feature Masking', linewidth=2)
    plt.plot(levels, noise_mses, marker='s', label='Gaussian Noise', linewidth=2)
    plt.plot(levels, junk_mses, marker='^', label='Junk Features', linewidth=2)
    plt.xlabel("Corruption Level")
    plt.ylabel("MSE from Clean Features")
    plt.title("Effect of Different Corruption Types on Feature Quality")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()
    print(f"Saved corruption effects analysis to {filename}")


def plot_combined_training_history(ae_history, gcn_history, filename="combined_training.png"):
    """Plot both AE and GCN training curves together"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # AE training
    ax1.plot(ae_history["train_loss"], label="AE Train Loss", color='blue')
    ax1.plot(ae_history["val_loss"], label="AE Val Loss", color='red')
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("MSE Loss")
    ax1.set_title("Autoencoder Training")
    ax1.legend()
    ax1.grid(True, alpha=0.3)


# Main experiment

def main():
    config = Config()

    dataset, data = load_dataset(config.dataset_name)
    data = data.to(config.device)

    print(f"Dataset: {config.dataset_name}")
    print(f"Nodes: {data.num_nodes}")
    print(f"Edges: {data.num_edges}")
    print(f"Original feature dim: {data.num_node_features}")
    print(f"Classes: {dataset.num_classes}")
    print("-" * 60)

    results = []
    last_ae_history = None

    for seed in config.seeds:
        set_seed(seed)
        print(f"\n================ Seed {seed} ================\n")

        for junk_dims in config.junk_dim_list:
            print(f"\n----- junk_dims = {junk_dims} -----")
            config.junk_dims = junk_dims

            baseline_acc = run_baseline_noisy_gcn(data, config)
            dae_acc, ae_history, dae_mse = run_dae_gcn(data, config, use_graph_aggregation=False)
            graph_dae_acc, _, graph_dae_mse = run_dae_gcn(data, config, use_graph_aggregation=True)

            last_ae_history = ae_history

            row = {
                "seed": seed,
                "junk_dims": junk_dims,
                "baseline_acc": baseline_acc,
                "dae_acc": dae_acc,
                "graph_dae_acc": graph_dae_acc,
                "dae_mse": dae_mse,
                "graph_dae_mse": graph_dae_mse,
            }
            results.append(row)
            print(row)

    save_results_csv(results, "results.csv")
    plot_accuracy_vs_junk(results, "accuracy_vs_junkdims.png")
    plot_feature_reconstruction_quality(results, "reconstruction_quality.png")

    if last_ae_history is not None:
        plot_ae_history(last_ae_history, "ae_loss_curve.png")

    # Additional visualizations
    if len(results) > 0:
        plot_corruption_effect_analysis(data, config, "corruption_effects.png")

    # Generate visualizations using the last trained model
    if last_ae_history is not None:
        # We need to retrain a model to get the latent embeddings
        # For demonstration, let's train one more model with moderate noise
        config_temp = Config()
        config_temp.junk_dims = 300  # Moderate noise level

        x_noisy_aug, x_clean_target = corrupt_features(
            x_clean=data.x,
            junk_dims=config_temp.junk_dims,
            mask_prob=config_temp.mask_prob,
            noise_std=config_temp.noise_std,
        )

        ae_model_vis = BetterDenoisingAutoencoder(
            input_dim=x_noisy_aug.size(1),
            latent_dim=config_temp.latent_dim,
            target_dim=data.x.size(1),
            hidden_dim=config_temp.ae_hidden_dim,
            dropout=config_temp.ae_dropout,
        ).to(config.device)

        ae_model_vis, _ = train_autoencoder(
            model=ae_model_vis,
            x_input=x_noisy_aug,
            x_target=x_clean_target,
            train_mask=data.train_mask,
            val_mask=data.val_mask,
            lr=config_temp.ae_lr,
            weight_decay=config_temp.ae_weight_decay,
            epochs=config_temp.ae_epochs,
            patience=config_temp.ae_patience,
            verbose=False,  # Less verbose for visualization
        )

        z_vis = encode_features(ae_model_vis, x_noisy_aug)

        # Generate additional plots
        plot_latent_space_embeddings(data, z_vis, "latent_embeddings.png")
        plot_feature_importance_analysis(data, ae_model_vis, x_noisy_aug, "feature_importance.png")
        plot_combined_training_history(last_ae_history, None, "combined_training.png")

    print("\nFinished all experiments and visualizations.")


if __name__ == "__main__":
    main()