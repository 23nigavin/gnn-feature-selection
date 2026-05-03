import torch
from sklearn.decomposition import PCA

def apply_pca(feature_matrix, train_mask, n_components):
    x_train = feature_matrix[train_mask].cpu().numpy()
    x_all = feature_matrix.cpu().numpy()
    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(x_train)
    x_pca = pca.transform(x_all)
    return torch.tensor(x_pca, dtype=feature_matrix.dtype, device=feature_matrix.device)