import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
import numpy as np
from sklearn.linear_model import LogisticRegression
from torch_geometric.utils import add_self_loops, degree
import matplotlib.pyplot as plt

#simple 2 layer gnn
class GCN(torch.nn.Module):
    def __init__(self, num_features, hidden_dim, num_classes):
        super().__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index) #potentially add more layers later
        return x

#trains model
def train(model, graph, optimizer):
    model.train()
    optimizer.zero_grad()
    out = model(graph.x, graph.edge_index)
    loss = F.cross_entropy(out[graph.train_mask], graph.y[graph.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

#tests model
def test(model, graph):
    model.eval()
    out = model(graph.x, graph.edge_index)
    pred = out.argmax(dim=1)

    correct = pred[graph.test_mask] == graph.y[graph.test_mask]
    accuracy = int(correct.sum()) / int(graph.test_mask.sum())
    return accuracy

"""
appends junk features to feature nodes

args:
x, feature matrix, [number of nodes, number of features]
num_junk_features, number of junk columns to append
"""
def add_junk_features(x, num_junk_features):
    num_nodes = x.size(0)
    junk = torch.randint(0, 2, (num_nodes, num_junk_features), device=x.device, dtype=x.dtype)
    return torch.cat([x, junk], dim=1)

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

def select_top_k_features_l1(x_agg, y, train_mask, k):
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
    top_k_indices = np.sort(top_k_indices)
    return top_k_indices

def run_experiment(dataset, noise_ratio=1.0, use_feature_selection=True, k=None, seed=42):
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
        selected_indices = select_top_k_features_l1(x_agg=x_agg, y=graph.y, train_mask=graph.train_mask, k=k)
        graph.x = graph.x[:, selected_indices]

    model = GCN(num_features=graph.x.shape[1], hidden_dim=16, num_classes=dataset.num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    for epoch in range(1, 201):
        loss = train(model, graph, optimizer)

    accuracy = test(model, graph)
    return accuracy

def run_experiment_avg(dataset, noise_ratio=1.0, use_feature_selection=True, k=None, seeds=None):
    if seeds is None:
        seeds = [0, 1, 2, 3, 4]

    accuracies = []
    for seed in seeds:
        acc = run_experiment(
            dataset,
            noise_ratio=noise_ratio,
            use_feature_selection=use_feature_selection,
            k=k,
            seed=seed
        )
        accuracies.append(acc)

    return float(np.mean(accuracies))

def main():
    dataset = Planetoid(root="data/Planetoid", name="Cora")
    original_num_features = dataset[0].x.shape[1]
    seeds = [0, 1, 2, 3, 4]

    noise_levels = [0.0, .25, .5, .75, 1.0, 1.25, 1.5]
    noise_percent = [100 * n for n in noise_levels]
    clean_baseline_acc = run_experiment_avg(dataset, noise_ratio=0.0, use_feature_selection=False, seeds=seeds)    
    noise_no_selection_acc = []
    noise_with_selection_acc = []

    for noise_ratio in noise_levels:
        acc_no_selection = run_experiment_avg(
            dataset,
            noise_ratio=noise_ratio,
            use_feature_selection=False,
            seeds=seeds
        )
        acc_with_selection = run_experiment_avg(
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

    plt.figure(figsize=(8, 5))
    plt.plot(noise_percent, [clean_baseline_acc] * len(noise_levels), marker='o', label='No noise, no feature selection')
    plt.plot(noise_percent, noise_no_selection_acc, marker='o', label='With noise, no feature selection')
    plt.plot(noise_percent, noise_with_selection_acc, marker='o', label='With noise, with feature selection')
    plt.xlabel("Extra junk features added (% of original feature count)")
    plt.ylabel("Test accuracy")
    plt.title("Accuracy vs. noise level")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("accuracy_vs_noise.png")
    plt.show()

    k_values = [100, 300, 500, 700, 1000, original_num_features]
    fixed_noise_ratio = 1.0

    corrupted_baseline_acc = run_experiment_avg(
        dataset,
        noise_ratio=fixed_noise_ratio,
        use_feature_selection=False,
        seeds=seeds
    )

    k_selection_acc = []

    for k in k_values:
        acc = run_experiment_avg(
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

    plt.figure(figsize=(8, 5))
    plt.plot(k_values, [clean_baseline_acc] * len(k_values), marker='o', label='No noise, no feature selection')
    plt.plot(k_values, [corrupted_baseline_acc] * len(k_values), marker='o', label='With noise, no feature selection')
    plt.plot(k_values, k_selection_acc, marker='o', label='With noise, with feature selection')
    plt.xlabel("Number of selected features (k)")
    plt.ylabel("Test accuracy")
    plt.title("Accuracy vs. number of selected features")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("accuracy_vs_k.png")
    plt.show()


if __name__ == "__main__":
    main()