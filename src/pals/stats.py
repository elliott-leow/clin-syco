"""Statistical tests: permutation tests, bootstrap CIs, effect sizes."""

import numpy as np
import torch
import torch.nn.functional as F


def permutation_test_means(data_a, data_b, n_perms=10000, seed=42):
    """Two-sample permutation test on difference of means.

    Returns:
        dict with "observed", "p_value", "null_distribution".
    """
    rng = np.random.RandomState(seed)
    a, b = np.asarray(data_a), np.asarray(data_b)
    observed = a.mean() - b.mean()
    combined = np.concatenate([a, b])
    n_a = len(a)

    null = np.empty(n_perms)
    for i in range(n_perms):
        perm = rng.permutation(combined)
        null[i] = perm[:n_a].mean() - perm[n_a:].mean()

    p_value = (np.abs(null) >= np.abs(observed)).mean()
    return {"observed": observed, "p_value": p_value, "null_distribution": null}


def bootstrap_ci(data, statistic=np.mean, n_bootstrap=10000, ci=0.95, seed=42):
    """Bootstrap confidence interval.

    Returns:
        (lower, upper) tuple.
    """
    rng = np.random.RandomState(seed)
    data = np.asarray(data)
    boot_stats = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sample = rng.choice(data, size=len(data), replace=True)
        boot_stats[i] = statistic(sample)

    alpha = (1 - ci) / 2
    return float(np.percentile(boot_stats, 100 * alpha)), \
           float(np.percentile(boot_stats, 100 * (1 - alpha)))


def cohen_d(a, b):
    """Cohen's d effect size."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    n_a, n_b = len(a), len(b)
    pooled_std = np.sqrt(
        ((n_a - 1) * a.std(ddof=1)**2 + (n_b - 1) * b.std(ddof=1)**2)
        / (n_a + n_b - 2)
    )
    if pooled_std == 0:
        return 0.0
    return float((a.mean() - b.mean()) / pooled_std)


def direction_similarity_permutation(pos_a, neg_a, pos_b, neg_b,
                                     layer, n_perms=5000, seed=42):
    """Test if cosine similarity between two contrastive directions is above chance.

    Permutes positive/negative labels within each condition to build a null
    distribution of cosine similarities.

    Args:
        pos_a, neg_a: Lists of activation dicts for condition A.
        pos_b, neg_b: Lists of activation dicts for condition B.
        layer: Which layer to test.
        n_perms: Number of permutations.

    Returns:
        dict with "observed_cos", "p_value", "null_distribution".
    """
    rng = np.random.RandomState(seed)

    def mean_diff_direction(pos_list, neg_list):
        pos_t = torch.stack([a[layer] for a in pos_list])
        neg_t = torch.stack([a[layer] for a in neg_list])
        diff = pos_t.mean(0) - neg_t.mean(0)
        return F.normalize(diff, dim=0)

    observed_dir_a = mean_diff_direction(pos_a, neg_a)
    observed_dir_b = mean_diff_direction(pos_b, neg_b)
    observed_cos = F.cosine_similarity(
        observed_dir_a.unsqueeze(0), observed_dir_b.unsqueeze(0)
    ).item()

    all_a = pos_a + neg_a
    all_b = pos_b + neg_b
    n_pos_a = len(pos_a)
    n_pos_b = len(pos_b)

    null_cos = np.empty(n_perms)
    for i in range(n_perms):
        perm_a = rng.permutation(len(all_a))
        perm_pos_a = [all_a[j] for j in perm_a[:n_pos_a]]
        perm_neg_a = [all_a[j] for j in perm_a[n_pos_a:]]

        perm_b = rng.permutation(len(all_b))
        perm_pos_b = [all_b[j] for j in perm_b[:n_pos_b]]
        perm_neg_b = [all_b[j] for j in perm_b[n_pos_b:]]

        dir_a = mean_diff_direction(perm_pos_a, perm_neg_a)
        dir_b = mean_diff_direction(perm_pos_b, perm_neg_b)
        null_cos[i] = F.cosine_similarity(
            dir_a.unsqueeze(0), dir_b.unsqueeze(0)
        ).item()

    p_value = (np.abs(null_cos) >= np.abs(observed_cos)).mean()
    return {
        "observed_cos": observed_cos,
        "p_value": float(p_value),
        "null_distribution": null_cos,
    }
