import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

# simple 2 layer GCN
class GCN(torch.nn.Module):
    def __init__(self, num_features, hidden_dim, num_classes, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, num_classes)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
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