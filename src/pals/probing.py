"""Linear probe training, evaluation, and cross-domain transfer."""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import cross_val_score


def prepare_probe_data(pos_acts_list, neg_acts_list, layer):
    """Stack activations at a specific layer into X, y arrays.

    Args:
        pos_acts_list: List of dicts, each layer_idx -> tensor.
        neg_acts_list: Same format.
        layer: Which layer to extract.

    Returns:
        X: (n_samples, hidden_dim) numpy array.
        y: (n_samples,) numpy array, 1 for positive, 0 for negative.
    """
    pos = np.stack([a[layer].numpy() for a in pos_acts_list])
    neg = np.stack([a[layer].numpy() for a in neg_acts_list])
    X = np.concatenate([pos, neg], axis=0)
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    return X, y


def train_probe(X, y, C=1.0):
    """Train a logistic regression probe."""
    clf = LogisticRegression(C=C, max_iter=1000, solver="lbfgs")
    clf.fit(X, y)
    return clf


def evaluate_probe(probe, X, y):
    """Evaluate probe accuracy and AUC."""
    y_pred = probe.predict(X)
    y_prob = probe.predict_proba(X)[:, 1]
    acc = accuracy_score(y, y_pred)
    try:
        auc = roc_auc_score(y, y_prob)
    except ValueError:
        auc = float("nan")
    return {"accuracy": acc, "auc": auc}


def cross_val_probe(X, y, C=1.0, cv=5):
    """Cross-validated probe accuracy."""
    clf = LogisticRegression(C=C, max_iter=1000, solver="lbfgs")
    n_cv = min(cv, min(np.sum(y == 0), np.sum(y == 1)))
    if n_cv < 2:
        return {"mean_accuracy": float("nan"), "std_accuracy": float("nan")}
    scores = cross_val_score(clf, X, y, cv=n_cv, scoring="accuracy")
    return {"mean_accuracy": scores.mean(), "std_accuracy": scores.std()}


def cross_domain_probing(source_pos, source_neg, target_pos, target_neg, layers):
    """Train probe on source domain, evaluate on target domain, per layer.

    Returns:
        dict[int, dict]: layer_idx -> {"accuracy": float, "auc": float}
    """
    results = {}
    for layer in layers:
        X_train, y_train = prepare_probe_data(source_pos, source_neg, layer)
        X_test, y_test = prepare_probe_data(target_pos, target_neg, layer)
        probe = train_probe(X_train, y_train)
        results[layer] = evaluate_probe(probe, X_test, y_test)
    return results


def within_domain_probing(pos_acts, neg_acts, layers, cv=5):
    """Cross-validated probing within a single domain, per layer.

    Returns:
        dict[int, dict]: layer_idx -> {"mean_accuracy", "std_accuracy"}
    """
    results = {}
    for layer in layers:
        X, y = prepare_probe_data(pos_acts, neg_acts, layer)
        results[layer] = cross_val_probe(X, y, cv=cv)
    return results
