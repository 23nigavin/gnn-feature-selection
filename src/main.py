# load cora
# corrupt features
# run one feature selection method
# train GNN
# print/save results

import torch
from torch_geometric.datasets import Planetoid

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


if __name__ == "__main__":
    main()