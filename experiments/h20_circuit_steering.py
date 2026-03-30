"""Hypothesis 20: Circuit-Based Steering — Can Causal Knowledge Fix Steering?

Uses the causally-identified circuit from H19 to construct better steering
vectors. Compares single-layer mean-diff (current H12 approach) against
multi-layer coordinated steering and PCA-subspace steering.
"""

import json
import os
import gc
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from pals.extraction import batch_extract_contrastive
from pals.directions import compute_contrastive_direction
from pals.steering import (apply_steering, apply_multi_layer_steering,
                            apply_subspace_steering, evaluate_steering_effect)
from pals.models import get_device


def run(model, tokenizer, stimuli_dir, output_dir, layers=None, n_stimuli=15,
        n_pca_components=5, alphas=None, critical_layers=None):
    os.makedirs(output_dir, exist_ok=True)

    if alphas is None:
        alphas = [2.0, 4.0, 8.0]

    with open(os.path.join(stimuli_dir, "cognitive_distortions.json")) as f:
        all_clinical = json.load(f)
    clinical = all_clinical[:n_stimuli]

    n_layers = model.config.num_hidden_layers
    device = get_device(model)
    model_dtype = next(model.parameters()).dtype

    if critical_layers is None:
        if layers is not None:
            critical_layers = layers[len(layers)//2:]
        else:
            critical_layers = list(range(2 * n_layers // 3, n_layers))

    print(f"\n=== H20: Circuit-Based Steering ===")
    print(f"Critical layers: {critical_layers}")
    print(f"PCA components: {n_pca_components}")

    # extract contrastive activations at critical layers
    print("Extracting contrastive activations...")
    clin_pos, clin_neg = batch_extract_contrastive(
        model, tokenizer, all_clinical,
        "sycophantic_completion", "therapeutic_completion",
        layers=critical_layers, desc="Clinical"
    )

    # compute per-layer directions
    directions = compute_contrastive_direction(clin_pos, clin_neg)

    # compute per-layer PCA directions
    pca_dirs = {}
    for l in critical_layers:
        diffs = torch.stack([clin_pos[i][l] - clin_neg[i][l] for i in range(len(clin_pos))])
        diffs = diffs - diffs.mean(0)
        U, S, Vh = torch.linalg.svd(diffs, full_matrices=False)
        k = min(n_pca_components, len(S))
        pca_dirs[l] = Vh[:k]  # (k, hidden_dim)

    steer_layer = critical_layers[len(critical_layers) // 2]
    print(f"Primary steering layer: {steer_layer}")

    results = {"methods": {}, "alphas": alphas, "steer_layer": steer_layer,
               "critical_layers": critical_layers}

    # Method 1: single-layer mean-diff (baseline, same as H12)
    print("\n--- Method 1: Single-layer mean-diff ---")
    for alpha in alphas:
        direction = directions[steer_layer].to(device=device, dtype=model_dtype)

        def make_ctx():
            return apply_steering(model, steer_layer, direction, alpha)

        effect = evaluate_steering_effect(model, tokenizer, clinical, make_ctx,
                                           random_layer=steer_layer, random_alpha=alpha)
        results["methods"][f"single_layer_alpha{alpha}"] = effect
        print(f"  alpha={alpha}: shift={effect['mean_shift']:+.4f}, z={effect['z_score']:.2f}")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Method 2: multi-layer coordinated steering
    print("\n--- Method 2: Multi-layer coordinated ---")
    for alpha in alphas:
        triples = [(l, directions[l].to(device=device, dtype=model_dtype), alpha)
                    for l in critical_layers]

        def make_ctx():
            return apply_multi_layer_steering(model, triples)

        effect = evaluate_steering_effect(model, tokenizer, clinical, make_ctx,
                                           random_layer=steer_layer, random_alpha=alpha)
        results["methods"][f"multi_layer_alpha{alpha}"] = effect
        print(f"  alpha={alpha}: shift={effect['mean_shift']:+.4f}, z={effect['z_score']:.2f}")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Method 3: PCA subspace steering at primary layer
    print("\n--- Method 3: PCA subspace ---")
    for alpha in alphas:
        pca_matrix = pca_dirs[steer_layer].to(device=device, dtype=model_dtype)
        pca_alphas = torch.full((pca_matrix.shape[0],), alpha,
                                 device=device, dtype=model_dtype)

        def make_ctx():
            return apply_subspace_steering(model, steer_layer, pca_matrix, pca_alphas)

        effect = evaluate_steering_effect(model, tokenizer, clinical, make_ctx,
                                           random_layer=steer_layer, random_alpha=alpha)
        results["methods"][f"pca_subspace_alpha{alpha}"] = effect
        print(f"  alpha={alpha}: shift={effect['mean_shift']:+.4f}, z={effect['z_score']:.2f}")

    with open(os.path.join(output_dir, "h20_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    methods = ['single_layer', 'multi_layer', 'pca_subspace']
    colors = ['#2980b9', '#c0392b', '#27ae60']
    x = np.arange(len(alphas))
    w = 0.25
    for mi, (method, color) in enumerate(zip(methods, colors)):
        shifts = [results["methods"][f"{method}_alpha{a}"]["mean_shift"] for a in alphas]
        ax.bar(x + mi * w, shifts, w, color=color, label=method.replace('_', ' ').title())

    ax.set_xticks(x + w)
    ax.set_xticklabels([f'α={a}' for a in alphas])
    ax.axhline(0, color='gray', ls=':', alpha=0.4)
    ax.set(ylabel='Mean shift (+ = more therapeutic)',
           title='H20: Steering Method Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "h20_circuit_steering.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    print(f"\nH20 results saved to {output_dir}")
    return results
