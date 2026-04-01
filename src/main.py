import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv

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
        x = self.conv2(x, edge_index)
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