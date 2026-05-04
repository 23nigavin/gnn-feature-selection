# All noise functions exist here
import torch

"""
Simply appends junk features to feature nodes

args:
x, feature matrix, [number of nodes, number of features]
num_junk_features, number of junk columns to append
"""
def add_junk_features(x, num_junk_features):
    num_nodes = x.size(0)
    junk = torch.randint(0, 2, (num_nodes, num_junk_features), device=x.device, dtype=x.dtype)
    return torch.cat([x, junk], dim=1)

def feature_dropout(x, drop_prob):
    if drop_prob <= 0:
        return x

    x_corrupt = x.clone()
    ones = x_corrupt == 1
    drop = torch.rand_like(x_corrupt) < drop_prob
    x_corrupt[ones & drop] = 0
    return x_corrupt

def bit_flip_features(x, flip_prob):
    if flip_prob <= 0:
        return x

    x_corrupt = x.clone()
    flip = torch.rand_like(x_corrupt) < flip_prob
    x_corrupt[flip] = 1 - x_corrupt[flip]
    return x_corrupt

def apply_noise(x, noise_type="dense_junk", noise_level=1.0):
    _, num_features = x.shape

    if noise_type == "dense_junk":
        num_junk_features = int(num_features * noise_level)
        return add_junk_features(x, num_junk_features)
    if noise_type == "feature_dropout":
        return feature_dropout(x, noise_level)
    if noise_type == "bit_flip":
        return bit_flip_features(x, noise_level)
    raise ValueError(f"Unknown noise_type: {noise_type}")