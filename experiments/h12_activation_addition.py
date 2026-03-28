"""Hypothesis 12: Activation Addition — Can We Override Sycophancy at Generation?

H2 showed the model knows the right clinical answer (+5.3 signal at layer 11)
but still generates sycophancy. Test whether adding an anti-sycophancy vector
at specific layers can flip generation to therapeutic.

Analysis:
  1. Compute the clinical sycophancy direction.
  2. At each layer, add -alpha * direction to the residual stream during
     a forward pass.
  3. Measure the shift in log P(therapeutic) - log P(sycophantic).
  4. Find the layer where intervention is most effective.
"""

import json
import os
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from pals.extraction import batch_extract_contrastive
from pals.directions import compute_contrastive_direction
from pals.models import get_device


def run(model, tokenizer, stimuli_dir, output_dir, layers=None, n_stimuli=15, alphas=None):
    os.makedirs(output_dir, exist_ok=True)

    if alphas is None:
        alphas = [0.5, 1.0, 2.0, 4.0, 8.0]

    with open(os.path.join(stimuli_dir, "cognitive_distortions.json")) as f:
        clinical = json.load(f)[:n_stimuli]
    with open(os.path.join(stimuli_dir, "factual_control.json")) as f:
        factual = json.load(f)

    print("\n=== H12: Activation Addition ===")

    # Compute clinical sycophancy direction (use all stimuli for direction, subset for intervention)
    with open(os.path.join(stimuli_dir, "cognitive_distortions.json")) as f:
        all_clinical = json.load(f)

    print("Computing clinical sycophancy direction...")
    clin_pos, clin_neg = batch_extract_contrastive(
        model, tokenizer, all_clinical,
        "sycophantic_completion", "therapeutic_completion",
        layers=layers, desc="Clinical"
    )
    syc_direction = compute_contrastive_direction(clin_pos, clin_neg)

    device = get_device(model)
    n_layers = model.config.num_hidden_layers
    test_layers = sorted(syc_direction.keys())

    # Baseline: measure log_prob difference without intervention
    print("\nComputing baselines...")
    baselines = []
    for s in tqdm(clinical, desc="Baseline"):
        input_ids = tokenizer.encode(s["user_prompt"], return_tensors="pt").to(device)
        ther_tok = tokenizer.encode(s["therapeutic_completion"], add_special_tokens=False)[0]
        syc_tok = tokenizer.encode(s["sycophantic_completion"], add_special_tokens=False)[0]

        with torch.no_grad():
            logits = model(input_ids).logits[0, -1, :]
        lp = F.log_softmax(logits.float(), dim=-1)
        baselines.append((lp[ther_tok] - lp[syc_tok]).item())

    mean_baseline = np.mean(baselines)
    print(f"Mean baseline (therapeutic - sycophantic): {mean_baseline:+.4f}")

    # Intervention: add -alpha * syc_direction at each layer
    print(f"\nTesting {len(alphas)} alpha values x {len(test_layers)} layers x {len(clinical)} stimuli")
    results_by_alpha = {}

    for alpha in alphas:
        print(f"\n--- Alpha = {alpha} ---")
        layer_shifts = {}

        for target_layer in test_layers:
            shifts = []
            direction_vec = syc_direction[target_layer].to(device=device, dtype=torch.float16)

            for s in clinical:
                input_ids = tokenizer.encode(s["user_prompt"], return_tensors="pt").to(device)
                ther_tok = tokenizer.encode(s["therapeutic_completion"], add_special_tokens=False)[0]
                syc_tok = tokenizer.encode(s["sycophantic_completion"], add_special_tokens=False)[0]

                # Hook to add anti-sycophancy vector
                def make_hook(direction, scale):
                    def hook_fn(module, inp, out):
                        h = out[0] if isinstance(out, tuple) else out
                        h = h.clone()
                        h[:, -1, :] -= scale * direction
                        if isinstance(out, tuple):
                            return (h,) + out[1:]
                        return h
                    return hook_fn

                hook = model.model.layers[target_layer].register_forward_hook(
                    make_hook(direction_vec, alpha)
                )

                with torch.no_grad():
                    logits = model(input_ids).logits[0, -1, :]
                lp = F.log_softmax(logits.float(), dim=-1)
                intervened = (lp[ther_tok] - lp[syc_tok]).item()
                hook.remove()

                # Shift relative to baseline for this stimulus
                base_idx = clinical.index(s)
                shifts.append(intervened - baselines[base_idx])

            layer_shifts[target_layer] = np.mean(shifts)

        results_by_alpha[alpha] = layer_shifts
        best_layer = max(layer_shifts, key=layer_shifts.get)
        print(f"  Best layer: {best_layer} (shift = {layer_shifts[best_layer]:+.4f})")

    # Save
    results = {
        "baselines": {"mean": mean_baseline, "individual": baselines},
        "interventions": {
            str(alpha): {str(l): v for l, v in shifts.items()}
            for alpha, shifts in results_by_alpha.items()
        },
        "n_stimuli": n_stimuli,
        "alphas": alphas,
    }

    with open(os.path.join(output_dir, "h12_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: shift by layer for each alpha
    ax = axes[0]
    for alpha in alphas:
        layers_list = sorted(results_by_alpha[alpha].keys())
        vals = [results_by_alpha[alpha][l] for l in layers_list]
        ax.plot(layers_list, vals, marker="o", markersize=4, label=f"α={alpha}")
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Intervention Layer")
    ax.set_ylabel("Mean Shift (positive = more therapeutic)")
    ax.set_title("Anti-Sycophancy Intervention Effect")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Right: best alpha effect at each layer
    ax = axes[1]
    best_alpha = alphas[-1]
    layers_list = sorted(results_by_alpha[best_alpha].keys())
    vals = [results_by_alpha[best_alpha][l] for l in layers_list]
    colors = ["#4daf4a" if v > 0 else "#e41a1c" for v in vals]
    ax.bar(layers_list, vals, color=colors)
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Layer")
    ax.set_ylabel(f"Shift at α={best_alpha}")
    ax.set_title(f"Per-Layer Effect (α={best_alpha})")
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("H12: Can Activation Addition Override Clinical Sycophancy?", fontsize=13)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "h12_activation_addition.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\nH12 results saved to {output_dir}")
    return results
