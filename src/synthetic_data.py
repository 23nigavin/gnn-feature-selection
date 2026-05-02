# synthetic_data.py

import numpy as np
import torch
from torch_geometric.data import Data


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


# ============================================================
# GRAPH GENERATION (Stochastic Block Model)
# ============================================================

def make_sbm_adjacency(num_classes, nodes_per_class, p_in, p_out, seed):
    rng = np.random.default_rng(seed)
    n = num_classes * nodes_per_class

    labels = np.repeat(np.arange(num_classes), nodes_per_class)
    adj = np.zeros((n, n), dtype=np.float32)

    for i in range(n):
        for j in range(i + 1, n):
            if labels[i] == labels[j]:
                if rng.random() < p_in:
                    adj[i, j] = 1
                    adj[j, i] = 1
            else:
                if rng.random() < p_out:
                    adj[i, j] = 1
                    adj[j, i] = 1

    return adj, labels


# ============================================================
# FEATURE GENERATION
# ============================================================

def make_informative_features(labels, dim, class_sep, noise_std, seed):
    rng = np.random.default_rng(seed)
    n = len(labels)
    num_classes = len(np.unique(labels))

    # class-specific means
    means = rng.normal(0, 1, size=(num_classes, dim))
    means = class_sep * means

    x = np.zeros((n, dim), dtype=np.float32)

    for i, c in enumerate(labels):
        x[i] = means[c] + rng.normal(0, noise_std, size=dim)

    return x


def add_junk_features(x_info, junk_dim, seed):
    rng = np.random.default_rng(seed)

    if junk_dim <= 0:
        return x_info.astype(np.float32)

    junk = rng.normal(0, 1, size=(x_info.shape[0], junk_dim))
    return np.concatenate([x_info, junk], axis=1).astype(np.float32)


def corrupt_features(x, mask_prob, noise_std, seed):
    rng = np.random.default_rng(seed)
    x_corrupt = x.copy()

    if mask_prob > 0:
        mask = rng.random(x.shape) > mask_prob
        x_corrupt *= mask

    if noise_std > 0:
        x_corrupt += rng.normal(0, noise_std, size=x.shape)

    return x_corrupt.astype(np.float32)


# ============================================================
# UTILITIES
# ============================================================

def adjacency_to_edge_index(adj):
    src, dst = np.nonzero(adj)
    return torch.tensor([src, dst], dtype=torch.long)


def make_masks(labels, train_per_class, val_per_class, seed):
    rng = np.random.default_rng(seed)
    n = len(labels)

    train_mask = np.zeros(n, dtype=bool)
    val_mask = np.zeros(n, dtype=bool)
    test_mask = np.zeros(n, dtype=bool)

    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        rng.shuffle(idx)

        train_idx = idx[:train_per_class]
        val_idx = idx[train_per_class:train_per_class + val_per_class]
        test_idx = idx[train_per_class + val_per_class:]

        train_mask[train_idx] = True
        val_mask[val_idx] = True
        test_mask[test_idx] = True

    return (
        torch.tensor(train_mask),
        torch.tensor(val_mask),
        torch.tensor(test_mask),
    )


# ============================================================
# MAIN FUNCTION
# ============================================================

def make_synthetic_graph_data(
    num_classes=3,
    nodes_per_class=100,
    informative_dim=10,
    junk_dim=100,
    p_in=0.08,
    p_out=0.01,
    class_sep=2.0,
    informative_noise_std=0.5,
    corruption_mask_prob=0.0,
    corruption_noise_std=0.0,
    train_per_class=20,
    val_per_class=30,
    seed=42,
):
    """
    Returns a PyTorch Geometric Data object.
    """

    set_seed(seed)

    # Graph
    adj, labels = make_sbm_adjacency(
        num_classes, nodes_per_class, p_in, p_out, seed
    )

    # Features
    x_info = make_informative_features(
        labels,
        informative_dim,
        class_sep,
        informative_noise_std,
        seed,
    )

    x_clean = add_junk_features(x_info, junk_dim, seed)

    x_corrupt = corrupt_features(
        x_clean,
        corruption_mask_prob,
        corruption_noise_std,
        seed,
    )

    # Convert to PyG format
    edge_index = adjacency_to_edge_index(adj)

    train_mask, val_mask, test_mask = make_masks(
        labels,
        train_per_class,
        val_per_class,
        seed,
    )

    data = Data(
        x=torch.tensor(x_corrupt, dtype=torch.float),
        x_clean=torch.tensor(x_clean, dtype=torch.float),
        edge_index=edge_index,
        y=torch.tensor(labels, dtype=torch.long),
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )

    return data