import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
from torch_geometric.utils import add_self_loops, degree
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif, SelectKBest, f_classif
import matplotlib.pyplot as plt
import random
from dataclasses import dataclass

@dataclass
class Config:
    dataset_name: str = "Cora"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    junk_dims: int = 500
    k_values: tuple = (100, 300, 500, 700, 1000)
    seeds: tuple = (42, 52, 62)

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# GNN Model
class GCN(torch.nn.Module):
    def __init__(self, num_features, hidden_dim, num_classes):
        super().__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

def train_gcn(model, x, edge_index, y, train_mask, val_mask, lr=0.01, weight_decay=5e-4, epochs=200):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_acc = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(x, edge_index)
        loss = F.cross_entropy(out[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            out_val = model(x, edge_index)
            preds = out_val.argmax(dim=1)
            acc = (preds[val_mask] == y[val_mask]).float().mean()

        if acc > best_acc:
            best_acc = acc
            best_state = model.state_dict()

    if best_state:
        model.load_state_dict(best_state)
    return model

def evaluate_gcn(model, x, edge_index, y, mask):
    model.eval()
    with torch.no_grad():
        out = model(x, edge_index)
        preds = out.argmax(dim=1)
        return (preds[mask] == y[mask]).float().mean().item()

# Feature corruption
def add_junk_features(x, num_junk_features):
    num_nodes = x.size(0)
    junk = torch.randn(num_nodes, num_junk_features, device=x.device)
    return torch.cat([x, junk], dim=1)

# Feature aggregation
def aggregate_features(x, edge_index):
    num_nodes = x.size(0)
    edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
    row, col = edge_index
    deg = degree(col, num_nodes=num_nodes, dtype=x.dtype)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0
    norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

    x_agg = torch.zeros_like(x)
    for i in range(edge_index.size(1)):
        src = row[i]
        dst = col[i]
        x_agg[dst] += norm[i] * x[src]

    return x_agg

# Feature selection methods
def select_features_l1(x_agg, y, train_mask, k):
    x_train = x_agg[train_mask].cpu().numpy()
    y_train = y[train_mask].cpu().numpy()

    clf = LogisticRegression(
        penalty="l1",
        solver="saga",
        max_iter=5000,
        C=1.0,
        random_state=42
    )
    clf.fit(x_train, y_train)

    coef = clf.coef_
    feature_scores = np.sum(np.abs(coef), axis=0)
    top_k_indices = np.argsort(feature_scores)[-k:]
    return np.sort(top_k_indices)

def select_features_permutation(x_agg, y, train_mask, k):
    x_train = x_agg[train_mask].cpu().numpy()
    y_train = y[train_mask].cpu().numpy()

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(x_train, y_train)

    perm_importance = permutation_importance(clf, x_train, y_train, n_repeats=10, random_state=42)
    feature_scores = perm_importance.importances_mean
    top_k_indices = np.argsort(feature_scores)[-k:]
    return np.sort(top_k_indices)

def select_features_correlation(x_agg, y, train_mask, k):
    x_train = x_agg[train_mask].cpu().numpy()
    y_train = y[train_mask].cpu().numpy()

    # Use ANOVA F-test for feature selection
    selector = SelectKBest(score_func=f_classif, k=k)
    selector.fit(x_train, y_train)
    top_k_indices = selector.get_support(indices=True)
    return np.sort(top_k_indices)

def select_features_mutual_info(x_agg, y, train_mask, k):
    x_train = x_agg[train_mask].cpu().numpy()
    y_train = y[train_mask].cpu().numpy()

    # Mutual information
    selector = SelectKBest(score_func=mutual_info_classif, k=k)
    selector.fit(x_train, y_train)
    top_k_indices = selector.get_support(indices=True)
    return np.sort(top_k_indices)

# Experiment runner
def run_experiment(data, method, k, use_aggregation=False):
    # Add junk features
    x_noisy = add_junk_features(data.x, Config.junk_dims)

    # Aggregate if requested
    x_input = aggregate_features(x_noisy, data.edge_index) if use_aggregation else x_noisy

    # Select features
    if method == "l1":
        selected_indices = select_features_l1(x_input, data.y, data.train_mask, k)
    elif method == "permutation":
        selected_indices = select_features_permutation(x_input, data.y, data.train_mask, k)
    elif method == "correlation":
        selected_indices = select_features_correlation(x_input, data.y, data.train_mask, k)
    elif method == "mutual_info":
        selected_indices = select_features_mutual_info(x_input, data.y, data.train_mask, k)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Apply selection
    x_selected = x_input[:, selected_indices]

    # Train GCN
    model = GCN(x_selected.size(1), 64, int(data.y.max().item()) + 1).to(Config.device)
    model = train_gcn(model, x_selected, data.edge_index, data.y, data.train_mask, data.val_mask)

    # Evaluate
    test_acc = evaluate_gcn(model, x_selected, data.edge_index, data.y, data.test_mask)
    return test_acc

def main():
    dataset = Planetoid(root="data/Cora", name="Cora")
    data = dataset[0].to(Config.device)

    methods = ["l1", "permutation", "correlation", "mutual_info"]
    aggregation_options = [False, True]  # Original vs aggregated features
    results = []

    for seed in Config.seeds:
        set_seed(seed)
        print(f"\n=== Seed {seed} ===")

        for method in methods:
            for use_agg in aggregation_options:
                for k in Config.k_values:
                    acc = run_experiment(data, method, k, use_agg)
                    results.append({
                        "seed": seed,
                        "method": method,
                        "aggregation": use_agg,
                        "k": k,
                        "accuracy": acc
                    })
                    print(".4f")

    # Save results
    import csv
    with open("feature_selection_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["seed", "method", "aggregation", "k", "accuracy"])
        writer.writeheader()
        writer.writerows(results)

    print("Feature selection experiments completed!")

if __name__ == "__main__":
    main()