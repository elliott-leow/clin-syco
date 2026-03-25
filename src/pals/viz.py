"""Visualization functions for PALS experiments."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")


def plot_cosine_similarity_by_layer(similarities_dict, title="Direction Cosine Similarity by Layer",
                                    save_path=None):
    """Plot cosine similarity curves for multiple direction pairs.

    Args:
        similarities_dict: dict mapping label -> {layer: cos_sim} dict.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    for label, sims in similarities_dict.items():
        layers = sorted(sims.keys())
        values = [sims[l] for l in layers]
        ax.plot(layers, values, marker="o", markersize=4, label=label)

    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Cosine Similarity")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_probe_transfer(results_dict, title="Cross-Domain Probe Transfer",
                        metric="accuracy", save_path=None):
    """Plot probe accuracy/AUC across layers for multiple transfer conditions.

    Args:
        results_dict: dict mapping label -> {layer: {"accuracy": ..., "auc": ...}}.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    for label, layer_results in results_dict.items():
        layers = sorted(layer_results.keys())
        values = [layer_results[l][metric] for l in layers]
        ax.plot(layers, values, marker="o", markersize=4, label=label)

    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Chance")
    ax.set_xlabel("Layer")
    ax.set_ylabel(metric.capitalize())
    ax.set_title(title)
    ax.legend()
    ax.set_ylim(0.3, 1.05)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_logit_lens_signals(signals_dict, title="Logit Lens: Correct Answer Signal by Layer",
                            save_path=None):
    """Plot layer-wise correct-answer signal for multiple conditions.

    Args:
        signals_dict: dict mapping condition_label -> {"layers": [...], "mean": {...}, "std": {...}}.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    colors = plt.cm.tab10.colors
    for i, (label, data) in enumerate(signals_dict.items()):
        layers = data["layers"]
        means = [data["mean"][l] for l in layers]
        stds = [data["std"][l] for l in layers]
        means_arr = np.array(means)
        stds_arr = np.array(stds)

        ax.plot(layers, means, marker="o", markersize=4, label=label, color=colors[i])
        ax.fill_between(layers, means_arr - stds_arr, means_arr + stds_arr,
                        alpha=0.15, color=colors[i])

    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Signal (log P(therapeutic) - log P(sycophantic))")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_direction_evolution(checkpoint_sims, title="Empathy-Sycophancy Direction Alignment Across Training",
                             save_path=None):
    """Plot how cosine similarity between empathy and sycophancy directions
    evolves across training checkpoints.

    Args:
        checkpoint_sims: dict mapping checkpoint_name -> {layer: cos_sim}.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left panel: per-layer curves for each checkpoint
    ax = axes[0]
    for ckpt_name, sims in checkpoint_sims.items():
        layers = sorted(sims.keys())
        values = [sims[l] for l in layers]
        ax.plot(layers, values, marker="o", markersize=4, label=ckpt_name)

    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Cosine Similarity")
    ax.set_title("Per-Layer Alignment")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right panel: mean similarity across layers per checkpoint
    ax = axes[1]
    ckpt_names = list(checkpoint_sims.keys())
    mean_sims = []
    for name in ckpt_names:
        sims = checkpoint_sims[name]
        mean_sims.append(np.mean(list(sims.values())))

    ax.bar(ckpt_names, mean_sims, color=plt.cm.tab10.colors[:len(ckpt_names)])
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_ylabel("Mean Cosine Similarity")
    ax.set_title("Mean Alignment by Checkpoint")
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(title, fontsize=13)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_projection_histograms(projections_pos, projections_neg, layer,
                               labels=("Sycophantic", "Therapeutic"),
                               title=None, save_path=None):
    """Plot histograms of activation projections onto a direction for a single layer.

    Args:
        projections_pos: list of scalar projections for positive class.
        projections_neg: list of scalar projections for negative class.
    """
    fig, ax = plt.subplots(figsize=(8, 4))

    ax.hist(projections_pos, bins=15, alpha=0.6, label=labels[0], color="coral")
    ax.hist(projections_neg, bins=15, alpha=0.6, label=labels[1], color="steelblue")
    ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Projection onto sycophancy direction")
    ax.set_ylabel("Count")
    ax.set_title(title or f"Layer {layer}: Projection Distribution")
    ax.legend()
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig
