# Utility functions

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