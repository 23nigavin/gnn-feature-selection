import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
import numpy as np
from sklearn.linear_model import LogisticRegression
from torch_geometric.utils import add_self_loops, degree

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

def main():
    dataset = Planetoid(root="data/Planetoid", name="Cora")
    graph = dataset[0]
    feature_matrix = graph.x
    num_nodes, num_features = feature_matrix.shape
    num_junk_features = int(num_features * 1.0)

    print(f"Original feature shape: {feature_matrix.shape}")

    feature_matrix = add_junk_features(feature_matrix, num_junk_features)
    graph.x = feature_matrix
    
    print(f"Corrupted feature shape: {feature_matrix.shape}")
    print("Noise injection complete.")

    x_agg = aggregate_features(graph.x, graph.edge_index)
    selected_indices = select_top_k_features_l1(x_agg=x_agg, y=graph.y, train_mask=graph.train_mask, k=num_features)
    graph.x = graph.x[:, selected_indices]

    print(f"Selected feature shape: {graph.x.shape}")
    print("Feature selection complete.")

    model = GCN(num_features=graph.x.shape[1], hidden_dim=16, num_classes=dataset.num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    for epoch in range(1, 201):
        loss = train(model, graph, optimizer)
        if epoch % 20 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")

    accuracy = test(model, graph)
    print(f"Test accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()