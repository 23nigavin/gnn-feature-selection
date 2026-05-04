# All preprocessing feature selection methods exist here
import numpy as np
from sklearn import __version__ as SKLEARN_VERSION
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.inspection import permutation_importance

def _sklearn_version_at_least(major, minor):
    version_parts = SKLEARN_VERSION.split(".")
    try:
        current_major = int(version_parts[0])
        current_minor = int(version_parts[1])
    except (IndexError, ValueError):
        return False

    return (current_major, current_minor) >= (major, minor)


def make_l1_logistic_regression():
    """
    Build an L1-style logistic regression model without triggering newer
    sklearn deprecation warnings.
    """
    common_args = {
        "solver": "saga",
        "max_iter": 10000,
        "tol": 1e-3,
        "C": 1.0,
        "random_state": 42,
    }

    if _sklearn_version_at_least(1, 8):
        return LogisticRegression(l1_ratio=1.0, **common_args)

    return LogisticRegression(penalty="l1", **common_args)


# L1 logistic regression feature scoring
def select_top_k_features_l1(feature_matrix, y, train_mask, k):
    x_train = feature_matrix[train_mask].cpu().numpy()
    y_train = y[train_mask].cpu().numpy()

    clf = make_l1_logistic_regression()
    clf.fit(x_train, y_train)

    coef = clf.coef_
    feature_scores = np.sum(np.abs(coef), axis=0)
    top_k_indices = np.argsort(feature_scores)[-k:]
    top_k_indices = np.sort(top_k_indices)
    return top_k_indices

def select_features_mutual_info(feature_matrix, y, train_mask, k):
    x_train = feature_matrix[train_mask].cpu().numpy()
    y_train = y[train_mask].cpu().numpy()

    selector = SelectKBest(score_func=mutual_info_classif, k=k)
    selector.fit(x_train, y_train)
    top_k_indices = selector.get_support(indices=True)
    return np.sort(top_k_indices)
