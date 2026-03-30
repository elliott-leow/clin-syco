"""Hypothesis 21: Nonlinear Steering — Is the Circuit Nonlinear?

Tests whether a learned nonlinear transformation produces better steering
than linear activation addition. If so, the sycophancy circuit has nonlinear
structure that mean-difference directions cannot capture.
"""

import json
import os
import gc
import numpy as np
import torch

from pals.steering import (train_nonlinear_steering, apply_steering,
                            evaluate_steering_effect)
from pals.extraction import batch_extract_contrastive
from pals.directions import compute_contrastive_direction
from pals.models import get_device


def run(model, tokenizer, stimuli_dir, output_dir, layers=None, n_stimuli=15,
        hidden_size=32, epochs=30, lr=1e-3, steer_layer=None):
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(stimuli_dir, "cognitive_distortions.json")) as f:
        all_clinical = json.load(f)
    clinical = all_clinical[:n_stimuli]

    n_layers = model.config.num_hidden_layers
    device = get_device(model)
    model_dtype = next(model.parameters()).dtype

    # pick steering layer (prefer caller-specified, fall back to 2/3 depth)
    if steer_layer is None:
        steer_layer = 2 * n_layers // 3
        if layers is not None and steer_layer not in layers:
            steer_layer = min(layers, key=lambda l: abs(l - steer_layer))

    print(f"\n=== H21: Nonlinear Steering ===")
    print(f"Steering layer: {steer_layer}")
    print(f"MLP hidden size: {hidden_size}")
    print(f"Training: {epochs} epochs, lr={lr}")

    # train nonlinear steering MLP
    print("\nTraining nonlinear steering MLP...")
    mlp, train_log = train_nonlinear_steering(
        model, tokenizer, clinical, steer_layer,
        hidden_size=hidden_size, epochs=epochs, lr=lr
    )
    print(f"  Final train loss: {train_log[-1]:.4f}")

    # evaluate nonlinear steering
    print("\nEvaluating nonlinear steering...")

    def make_nonlinear_ctx():
        from pals.steering import apply_nonlinear_steering
        return apply_nonlinear_steering(model, steer_layer, mlp)

    nonlinear_effect = evaluate_steering_effect(model, tokenizer, clinical, make_nonlinear_ctx,
                                                random_layer=steer_layer, random_alpha=8.0)
    print(f"  Nonlinear: shift={nonlinear_effect['mean_shift']:+.4f}, z={nonlinear_effect['z_score']:.2f}")

    # linear baseline for comparison
    print("\nLinear baseline...")
    clin_pos, clin_neg = batch_extract_contrastive(
        model, tokenizer, all_clinical,
        "sycophantic_completion", "therapeutic_completion",
        layers=[steer_layer], desc="Linear baseline"
    )
    direction = compute_contrastive_direction(clin_pos, clin_neg)[steer_layer]
    direction_gpu = direction.to(device=device, dtype=model_dtype)

    for alpha in [4.0, 8.0]:
        def make_linear_ctx(a=alpha):
            return apply_steering(model, steer_layer, direction_gpu, a)

        linear_effect = evaluate_steering_effect(model, tokenizer, clinical, make_linear_ctx,
                                                  random_layer=steer_layer, random_alpha=alpha)
        print(f"  Linear alpha={alpha}: shift={linear_effect['mean_shift']:+.4f}, z={linear_effect['z_score']:.2f}")

    results = {
        "steer_layer": steer_layer,
        "hidden_size": hidden_size,
        "epochs": epochs,
        "train_loss_final": float(train_log[-1]),
        "train_loss_curve": [float(x) for x in train_log],
        "nonlinear_effect": nonlinear_effect,
    }

    with open(os.path.join(output_dir, "h21_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.plot(train_log, color='#c0392b', lw=1.5)
    ax1.set(xlabel='Epoch', ylabel='Loss', title='MLP training loss')
    ax1.grid(True, alpha=0.3)

    methods = ['Nonlinear MLP']
    shifts = [nonlinear_effect['mean_shift']]
    zscores = [nonlinear_effect['z_score']]
    colors = ['#c0392b']

    ax2.bar(range(len(methods)), shifts, color=colors, width=0.5)
    ax2.axhline(0, color='gray', ls=':', alpha=0.4)
    ax2.set_xticks(range(len(methods)))
    ax2.set_xticklabels(methods)
    ax2.set(ylabel='Mean shift', title='H21: Nonlinear vs Linear Steering')
    for i, (s, z) in enumerate(zip(shifts, zscores)):
        ax2.text(i, s + 0.005, f'z={z:.1f}', ha='center', fontsize=8)
    ax2.grid(True, alpha=0.3, axis='y')

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "h21_nonlinear_steering.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    print(f"\nH21 results saved to {output_dir}")
    return results
