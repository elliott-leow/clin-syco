"""Contrastive direction computation and projection."""

import torch
import torch.nn.functional as F


def compute_contrastive_direction(pos_acts_list, neg_acts_list):
    """Compute the mean-difference direction per layer.

    Direction = normalize(mean(positive) - mean(negative)).

    Args:
        pos_acts_list: List of dicts, each layer_idx -> (hidden_dim,) tensor.
        neg_acts_list: Same format.

    Returns:
        dict[int, Tensor]: layer_idx -> (hidden_dim,) unit direction vector.
    """
    layers = sorted(pos_acts_list[0].keys())
    directions = {}

    for layer in layers:
        pos_stack = torch.stack([a[layer] for a in pos_acts_list])
        neg_stack = torch.stack([a[layer] for a in neg_acts_list])
        diff = pos_stack.mean(0) - neg_stack.mean(0)
        directions[layer] = F.normalize(diff, dim=0)

    return directions


def cosine_similarity_by_layer(dir_a, dir_b):
    """Cosine similarity between two direction dicts, per layer.

    Returns:
        dict[int, float]: layer_idx -> cosine similarity.
    """
    layers = sorted(set(dir_a.keys()) & set(dir_b.keys()))
    return {
        l: F.cosine_similarity(
            dir_a[l].unsqueeze(0), dir_b[l].unsqueeze(0)
        ).item()
        for l in layers
    }


def project_onto_direction(activation, direction):
    """Scalar projection of an activation vector onto a direction."""
    return (activation @ direction).item()


def project_acts_by_layer(acts_list, directions):
    """Project a list of activation dicts onto directions, per layer.

    Returns:
        dict[int, list[float]]: layer_idx -> list of scalar projections.
    """
    layers = sorted(directions.keys())
    projections = {l: [] for l in layers}
    for acts in acts_list:
        for l in layers:
            projections[l].append((acts[l] @ directions[l]).item())
    return projections


def compute_direction_from_split(all_acts, labels):
    """Compute direction from labeled activations (1 = positive, 0 = negative).

    Useful when acts come from a mixed source rather than pre-separated lists.
    """
    layers = sorted(all_acts[0].keys())
    directions = {}

    for layer in layers:
        pos = torch.stack([a[layer] for a, lab in zip(all_acts, labels) if lab == 1])
        neg = torch.stack([a[layer] for a, lab in zip(all_acts, labels) if lab == 0])
        diff = pos.mean(0) - neg.mean(0)
        directions[layer] = F.normalize(diff, dim=0)

    return directions
