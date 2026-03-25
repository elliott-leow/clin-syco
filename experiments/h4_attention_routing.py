"""Hypothesis 4: The Representation-Generation Dissociation.

If the model's residual stream encodes the correct clinical answer (H2)
but still generates sycophancy, specific attention heads in late layers
may route generation away from the correct representation.

Analysis:
  1. For each head in late layers (last third), ablate it and measure
     the shift in sycophantic vs therapeutic log-probability.
  2. Heads whose ablation shifts output toward therapeutic = "sycophancy routing heads."
  3. Test whether these heads overlap with heads identified in factual sycophancy.
"""

import json
import os
import numpy as np

from pals.attention import find_sycophancy_routing_heads, ablate_heads_and_measure


def run(model, tokenizer, stimuli_dir, output_dir, n_stimuli=10):
    """Run H4 experiment.

    Args:
        n_stimuli: Number of stimuli to use per condition (subset for speed).
    """
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(stimuli_dir, "cognitive_distortions.json")) as f:
        clinical = json.load(f)[:n_stimuli]
    with open(os.path.join(stimuli_dir, "factual_control.json")) as f:
        factual = json.load(f)[:n_stimuli]

    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    # Test last third of layers
    late_layers = list(range(2 * n_layers // 3, n_layers))

    print(f"\n=== H4: Attention Head Routing ===")
    print(f"Testing {len(late_layers)} layers × {n_heads} heads × {n_stimuli} stimuli")

    # Clinical sycophancy routing heads
    print("\nScanning clinical sycophancy routing heads...")
    clinical_effects = find_sycophancy_routing_heads(
        model, tokenizer, clinical, late_layers, n_heads, desc="Clinical"
    )

    # Factual sycophancy routing heads
    print("\nScanning factual sycophancy routing heads...")
    factual_effects = find_sycophancy_routing_heads(
        model, tokenizer, factual, late_layers, n_heads, desc="Factual"
    )

    # Find top sycophancy-promoting heads (most negative shift = most sycophantic)
    clinical_sorted = sorted(clinical_effects.items(), key=lambda x: x[1])
    factual_sorted = sorted(factual_effects.items(), key=lambda x: x[1])

    print("\nTop 10 sycophancy-promoting heads (clinical):")
    for (l, h), shift in clinical_sorted[:10]:
        print(f"  Layer {l}, Head {h}: shift = {shift:+.4f}")

    print("\nTop 10 sycophancy-promoting heads (factual):")
    for (l, h), shift in factual_sorted[:10]:
        print(f"  Layer {l}, Head {h}: shift = {shift:+.4f}")

    # Overlap analysis
    top_k = min(10, len(clinical_sorted))
    clinical_top = set(k for k, v in clinical_sorted[:top_k])
    factual_top = set(k for k, v in factual_sorted[:top_k])
    overlap = clinical_top & factual_top

    print(f"\nOverlap in top-{top_k} sycophancy heads: {len(overlap)}/{top_k}")
    if overlap:
        print(f"  Shared heads: {overlap}")

    # Save results
    results = {
        "clinical_head_effects": {f"L{l}H{h}": v for (l, h), v in clinical_effects.items()},
        "factual_head_effects": {f"L{l}H{h}": v for (l, h), v in factual_effects.items()},
        "clinical_top10": [{"layer": l, "head": h, "shift": s} for (l, h), s in clinical_sorted[:10]],
        "factual_top10": [{"layer": l, "head": h, "shift": s} for (l, h), s in factual_sorted[:10]],
        "overlap_count": len(overlap),
        "overlap_heads": [list(x) for x in overlap],
    }

    with open(os.path.join(output_dir, "h4_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Heatmap plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, effects, title in [
        (axes[0], clinical_effects, "Clinical Sycophancy"),
        (axes[1], factual_effects, "Factual Sycophancy"),
    ]:
        matrix = np.zeros((len(late_layers), n_heads))
        for i, l in enumerate(late_layers):
            for h in range(n_heads):
                matrix[i, h] = effects.get((l, h), 0.0)

        im = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-0.5, vmax=0.5)
        ax.set_xlabel("Head")
        ax.set_ylabel("Layer")
        ax.set_yticks(range(len(late_layers)))
        ax.set_yticklabels(late_layers)
        ax.set_title(f"H4: {title}\nHead Ablation Effect")
        plt.colorbar(im, ax=ax, label="Shift (neg = sycophancy promoting)")

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "h4_head_ablation.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\nH4 results saved to {output_dir}")
    return results
