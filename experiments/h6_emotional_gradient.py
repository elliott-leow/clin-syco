"""Hypothesis 6: Emotional Intensity Monotonically Modulates Sycophancy Mechanism.

The same false belief at varying emotional intensity should show a gradient
of cosine similarity with the factual sycophancy direction — decreasing as
emotion increases.

Analysis:
  1. Extract contrastive directions for each emotional level separately.
  2. Compute cosine similarity with the factual sycophancy direction at each level.
  3. Test for monotonic trend.
"""

import json
import os
import numpy as np

from pals.extraction import batch_extract_contrastive
from pals.directions import compute_contrastive_direction, cosine_similarity_by_layer
from pals.stats import bootstrap_ci


def run(model, tokenizer, stimuli_dir, output_dir, layers=None):
    """Run H6 experiment."""
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(stimuli_dir, "emotional_intensity_gradient.json")) as f:
        gradient = json.load(f)
    with open(os.path.join(stimuli_dir, "factual_control.json")) as f:
        factual = json.load(f)

    print("\n=== H6: Emotional Intensity Gradient ===")

    # Compute factual sycophancy direction as reference
    print("Computing factual sycophancy direction...")
    fact_pos, fact_neg = batch_extract_contrastive(
        model, tokenizer, factual,
        "sycophantic_completion", "therapeutic_completion",
        layers=layers, desc="Factual"
    )
    fact_dir = compute_contrastive_direction(fact_pos, fact_neg)

    # Split gradient stimuli by emotional level
    levels = sorted(set(s["emotional_level"] for s in gradient))
    level_dirs = {}
    level_cos = {}

    for level in levels:
        level_stimuli = [s for s in gradient if s["emotional_level"] == level]
        subcategory = level_stimuli[0]["subcategory"] if level_stimuli else str(level)
        print(f"\nLevel {level} ({subcategory}): {len(level_stimuli)} stimuli")

        pos, neg = batch_extract_contrastive(
            model, tokenizer, level_stimuli,
            "sycophantic_completion", "therapeutic_completion",
            layers=layers, desc=f"Level {level}"
        )
        direction = compute_contrastive_direction(pos, neg)
        level_dirs[level] = direction

        cos_sim = cosine_similarity_by_layer(direction, fact_dir)
        level_cos[level] = cos_sim
        mean_cos = np.mean(list(cos_sim.values()))
        print(f"  Mean cos with factual direction: {mean_cos:.3f}")

    # Trend analysis
    print(f"\n--- Emotional Intensity → Factual Similarity ---")
    means = []
    for level in levels:
        m = np.mean(list(level_cos[level].values()))
        means.append(m)
        label = {1: "Low", 2: "Medium", 3: "High"}.get(level, str(level))
        print(f"  {label} emotion: cos = {m:.3f}")

    # Test monotonic decrease
    diffs = [means[i+1] - means[i] for i in range(len(means)-1)]
    all_decreasing = all(d < 0 for d in diffs)
    print(f"\nMonotonic decrease: {'Yes' if all_decreasing else 'No'}")
    print(f"Differences: {[f'{d:+.3f}' for d in diffs]}")

    # Bootstrap CIs on the means
    ci_results = {}
    for level in levels:
        values = list(level_cos[level].values())
        lo, hi = bootstrap_ci(values, ci=0.95)
        ci_results[level] = {"mean": float(np.mean(values)), "ci_lo": lo, "ci_hi": hi}

    # Save
    results = {
        "cosine_by_level_and_layer": {
            str(level): {str(k): v for k, v in cos.items()}
            for level, cos in level_cos.items()
        },
        "mean_cosine_by_level": {str(level): float(np.mean(list(cos.values())))
                                  for level, cos in level_cos.items()},
        "trend_monotonic_decrease": all_decreasing,
        "trend_diffs": [float(d) for d in diffs],
        "bootstrap_ci": {str(k): v for k, v in ci_results.items()},
    }

    with open(os.path.join(output_dir, "h6_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: per-layer curves
    ax = axes[0]
    for level in levels:
        label = {1: "Low emotion", 2: "Medium emotion", 3: "High emotion"}.get(level, str(level))
        layers_list = sorted(level_cos[level].keys())
        values = [level_cos[level][l] for l in layers_list]
        ax.plot(layers_list, values, marker="o", markersize=4, label=label)
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Cosine Similarity with Factual Direction")
    ax.set_title("Per-Layer Similarity")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right: bar chart of means
    ax = axes[1]
    labels = [f"Level {l}" for l in levels]
    ax.bar(labels, means, color=["#4daf4a", "#ff7f00", "#e41a1c"])
    for i, (m, ci) in enumerate(zip(means, [ci_results[l] for l in levels])):
        ax.errorbar(i, m, yerr=[[m - ci["ci_lo"]], [ci["ci_hi"] - m]],
                     fmt="none", color="black", capsize=5)
    ax.set_ylabel("Mean Cosine Similarity with Factual Direction")
    ax.set_title("Gradient Effect")
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("H6: Emotional Intensity Modulates Sycophancy Mechanism", fontsize=13)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "h6_emotional_gradient.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\nH6 results saved to {output_dir}")
    return results
