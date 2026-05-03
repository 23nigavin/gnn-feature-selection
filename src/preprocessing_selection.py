# All preprocessing feature selection methods exist here
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.inspection import permutation_importance

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

def select_features_permutation(feature_matrix, y, train_mask, k):
    x_train = feature_matrix[train_mask].cpu().numpy()
    y_train = y[train_mask].cpu().numpy()

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(x_train, y_train)

    perm_importance = permutation_importance(clf, x_train, y_train, n_repeats=10, random_state=42)
    feature_scores = perm_importance.importances_mean
    top_k_indices = np.argsort(feature_scores)[-k:]
    return np.sort(top_k_indices)

def select_features_correlation(feature_matrix, y, train_mask, k):
    x_train = feature_matrix[train_mask].cpu().numpy()
    y_train = y[train_mask].cpu().numpy()

    selector = SelectKBest(score_func=f_classif, k=k)
    selector.fit(x_train, y_train)
    top_k_indices = selector.get_support(indices=True)
    return np.sort(top_k_indices)

def select_features_mutual_info(feature_matrix, y, train_mask, k):
    x_train = feature_matrix[train_mask].cpu().numpy()
    y_train = y[train_mask].cpu().numpy()

    selector = SelectKBest(score_func=mutual_info_classif, k=k)
    selector.fit(x_train, y_train)
    top_k_indices = selector.get_support(indices=True)
    return np.sort(top_k_indices)