# All preprocessing feature selection methods exist here

# L1 logistic regression feature scoring
def select_top_k_features_l1(feature_matrix, y, train_mask, k):
    x_train = feature_matrix[train_mask].cpu().numpy()
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