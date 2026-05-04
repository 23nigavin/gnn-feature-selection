import torch
from sklearn.decomposition import PCA

def apply_pca(feature_matrix, train_mask, n_components):
    """
    Fits PCA on all nodes and return transductive PCA features.

    The train_mask argument is kept so the call site matches supervised
    preprocessing methods, but PCA is unsupervised and uses no labels.
    
    """
    x_all = feature_matrix.cpu().numpy()
    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(x_all)
    x_pca = pca.transform(x_all)

    # The PCA output is a numpy array, so convert it back to a PyTorch tensor with the same dtype and device as the input features.
    return torch.tensor(x_pca, dtype=feature_matrix.dtype, device=feature_matrix.device)
