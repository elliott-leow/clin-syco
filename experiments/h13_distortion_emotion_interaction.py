"""Hypothesis 13: Distortion × Emotion Interaction.

H6 showed emotion modulates sycophancy mechanism. H7 showed distortion types vary.
Do these interact? Test the emotional gradient within each distortion type.

If catastrophizing (most factual-like) has a flat gradient and personalization
(least factual-like) has a steep gradient, it means emotion only reshapes
the mechanism for distortions that aren't already factual-like.
"""

import json
import os
import numpy as np
from collections import defaultdict

from pals.extraction import batch_extract_contrastive
from pals.directions import compute_contrastive_direction, cosine_similarity_by_layer
from pals.stats import bootstrap_ci


def run(model, tokenizer, stimuli_dir, output_dir, layers=None):
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(stimuli_dir, "emotional_intensity_gradient.json")) as f:
        gradient = json.load(f)
    with open(os.path.join(stimuli_dir, "factual_control.json")) as f:
        factual = json.load(f)

    print("\n=== H13: Distortion × Emotion Interaction ===")

    # Compute factual sycophancy direction as reference
    print("Computing factual sycophancy direction...")
    fact_pos, fact_neg = batch_extract_contrastive(
        model, tokenizer, factual,
        "sycophantic_completion", "therapeutic_completion",
        layers=layers, desc="Factual"
    )
    fact_dir = compute_contrastive_direction(fact_pos, fact_neg)

    # Group gradient stimuli by subcategory AND emotional level
    by_subcat_level = defaultdict(list)
    for s in gradient:
        key = (s["subcategory"], s["emotional_level"])
        by_subcat_level[key].append(s)

    subcategories = sorted(set(s["subcategory"] for s in gradient))
    levels = sorted(set(s["emotional_level"] for s in gradient))

    print(f"Subcategories: {subcategories}")
    print(f"Levels: {levels}")

    # For each subcategory × level combination, compute direction and cos with factual
    interaction = {}
    for subcat in subcategories:
        interaction[subcat] = {}
        for level in levels:
            key = (subcat, level)
            stimuli = by_subcat_level.get(key, [])
            if len(stimuli) < 2:
                print(f"  Skipping {subcat} × level {level} ({len(stimuli)} stimuli)")
                continue

            pos, neg = batch_extract_contrastive(
                model, tokenizer, stimuli,
                "sycophantic_completion", "therapeutic_completion",
                layers=layers, desc=f"{subcat[:8]}×L{level}"
            )
            direction = compute_contrastive_direction(pos, neg)
            cos = cosine_similarity_by_layer(direction, fact_dir)
            mean_cos = float(np.mean(list(cos.values())))
            interaction[subcat][level] = mean_cos
            print(f"  {subcat} × level {level}: cos = {mean_cos:.3f}")

    # Compute gradient slope per subcategory
    print(f"\nGradient slopes (change in cos per emotion level):")
    slopes = {}
    for subcat in subcategories:
        if len(interaction.get(subcat, {})) >= 2:
            vals = [interaction[subcat].get(l, None) for l in levels]
            vals = [v for v in vals if v is not None]
            if len(vals) >= 2:
                slope = float(vals[-1] - vals[0])
                slopes[subcat] = slope
                print(f"  {subcat:30s}: {slope:+.3f} ({'steeper' if abs(slope) > 0.1 else 'flat'})")

    # Save
    results = {
        "interaction": {
            subcat: {str(l): v for l, v in lvl_data.items()}
            for subcat, lvl_data in interaction.items()
        },
        "slopes": slopes,
    }

    with open(os.path.join(output_dir, "h13_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Plot: heatmap of subcategory × level
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: line plot per subcategory
    ax = axes[0]
    for subcat in subcategories:
        if subcat in interaction and len(interaction[subcat]) >= 2:
            lvls = sorted(interaction[subcat].keys())
            vals = [interaction[subcat][l] for l in lvls]
            ax.plot(lvls, vals, marker="o", markersize=5, label=subcat[:12])
    ax.set_xlabel("Emotional Intensity Level")
    ax.set_ylabel("Cosine Similarity with Factual Direction")
    ax.set_title("Emotional Gradient by Distortion Type")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # Right: slope bar chart
    ax = axes[1]
    if slopes:
        sorted_slopes = sorted(slopes.items(), key=lambda x: x[1])
        names = [s[:15] for s, _ in sorted_slopes]
        vals = [v for _, v in sorted_slopes]
        colors = ["#e41a1c" if v < -0.1 else "#4daf4a" if v > -0.05 else "#ff7f00" for v in vals]
        ax.barh(range(len(names)), vals, color=colors)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xlabel("Gradient Slope (high - low emotion)")
        ax.set_title("Emotion Sensitivity by Distortion Type")
        ax.axvline(0, color="gray", linestyle="--", alpha=0.5)
        ax.grid(True, alpha=0.3, axis="x")

    fig.suptitle("H13: Distortion × Emotion Interaction", fontsize=13)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "h13_distortion_emotion.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\nH13 results saved to {output_dir}")
    return results
