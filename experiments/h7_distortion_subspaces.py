"""Hypothesis 7: Distortion-Specific Subspaces Within Clinical Sycophancy.

Different cognitive distortion types may have different representational
signatures. Test whether distortion types cluster into meaningful groups.

Analysis:
  1. Compute contrastive directions per distortion subcategory.
  2. Pairwise cosine similarity matrix between distortion-type directions.
  3. Cosine similarity of each distortion type with the factual sycophancy direction.
  4. Hierarchical clustering to identify distortion families.

Note: Per-subcategory directions are computed from small samples (often 2-3
stimuli each). Directions estimated from fewer than 5 stimuli should be
treated as exploratory — they may not generalize and are sensitive to
individual stimulus idiosyncrasies.
"""

import json
import os
import numpy as np
from collections import defaultdict

from pals.extraction import batch_extract_contrastive
from pals.directions import compute_contrastive_direction, cosine_similarity_by_layer
from pals.decomposition import pairwise_cosine_matrix


def run(model, tokenizer, stimuli_dir, output_dir, layers=None, target_layer=None):
    """Run H7 experiment.

    Args:
        target_layer: Which layer to use for the pairwise matrix. If None,
                      uses 2/3 through the model (where clinical-specific
                      processing was strongest in H1).
    """
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(stimuli_dir, "cognitive_distortions.json")) as f:
        clinical = json.load(f)
    with open(os.path.join(stimuli_dir, "factual_control.json")) as f:
        factual = json.load(f)

    n_layers = model.config.num_hidden_layers
    if target_layer is None:
        target_layer = 2 * n_layers // 3

    print(f"\n=== H7: Distortion-Specific Subspaces ===")
    print(f"Target layer for analysis: {target_layer}")

    # Group clinical stimuli by subcategory
    by_subcat = defaultdict(list)
    for s in clinical:
        by_subcat[s["subcategory"]].append(s)

    print(f"Distortion types found: {len(by_subcat)}")
    for k, v in sorted(by_subcat.items()):
        print(f"  {k}: {len(v)} stimuli")

    # Compute factual direction
    print("\nComputing factual sycophancy direction...")
    fact_pos, fact_neg = batch_extract_contrastive(
        model, tokenizer, factual,
        "sycophantic_completion", "therapeutic_completion",
        layers=layers, desc="Factual"
    )
    fact_dir = compute_contrastive_direction(fact_pos, fact_neg)

    # Compute per-distortion directions
    distortion_dirs = {}
    distortion_cos_with_factual = {}

    for subcat, stimuli in sorted(by_subcat.items()):
        if len(stimuli) < 2:
            print(f"  Skipping {subcat} (only {len(stimuli)} stimuli)")
            continue

        if len(stimuli) < 5:
            print(f"  WARNING: {subcat} has only {len(stimuli)} stimuli — "
                  f"direction estimate is exploratory and may not generalize.")

        print(f"\nProcessing {subcat} ({len(stimuli)} stimuli)...")
        pos, neg = batch_extract_contrastive(
            model, tokenizer, stimuli,
            "sycophantic_completion", "therapeutic_completion",
            layers=layers, desc=subcat
        )
        direction = compute_contrastive_direction(pos, neg)
        distortion_dirs[subcat] = direction

        cos = cosine_similarity_by_layer(direction, fact_dir)
        distortion_cos_with_factual[subcat] = cos
        print(f"  cos with factual at layer {target_layer}: {cos.get(target_layer, 'N/A'):.3f}")

    # Pairwise cosine matrix at target layer
    print(f"\nComputing pairwise cosine matrix at layer {target_layer}...")
    target_dirs = {name: dirs[target_layer] for name, dirs in distortion_dirs.items()
                   if target_layer in dirs}
    names, matrix = pairwise_cosine_matrix(target_dirs)

    print(f"\nPairwise cosine matrix ({len(names)} distortion types):")
    header = "              " + "  ".join(f"{n[:6]:>6}" for n in names)
    print(header)
    for i, name in enumerate(names):
        row = f"{name[:12]:>12}  " + "  ".join(f"{matrix[i,j]:+.3f}" for j in range(len(names)))
        print(row)

    # Factual similarity ranking
    print(f"\nDistortion types ranked by similarity to factual direction (layer {target_layer}):")
    ranked = sorted(distortion_cos_with_factual.items(),
                    key=lambda x: x[1].get(target_layer, 0), reverse=True)
    for subcat, cos in ranked:
        print(f"  {subcat:30s} cos = {cos.get(target_layer, 0):+.3f}")

    # Save
    results = {
        "target_layer": target_layer,
        "distortion_types": names,
        "pairwise_matrix": matrix.tolist(),
        "factual_similarity": {
            subcat: {str(k): v for k, v in cos.items()}
            for subcat, cos in distortion_cos_with_factual.items()
        },
        "factual_similarity_at_target": {
            subcat: cos.get(target_layer, None)
            for subcat, cos in distortion_cos_with_factual.items()
        },
    }

    with open(os.path.join(output_dir, "h7_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Plots
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Left: pairwise cosine heatmap
    ax = axes[0]
    im = ax.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n[:10] for n in names], rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([n[:10] for n in names], fontsize=7)
    ax.set_title(f"Pairwise Cosine Similarity (Layer {target_layer})")
    plt.colorbar(im, ax=ax, shrink=0.8)

    # Right: bar chart of factual similarity
    ax = axes[1]
    subcats = [s for s, _ in ranked]
    vals = [c.get(target_layer, 0) for _, c in ranked]
    colors = ["#e41a1c" if v < 0.1 else "#4daf4a" if v > 0.3 else "#ff7f00" for v in vals]
    ax.barh(range(len(subcats)), vals, color=colors)
    ax.set_yticks(range(len(subcats)))
    ax.set_yticklabels([s[:20] for s in subcats], fontsize=7)
    ax.set_xlabel("Cosine Similarity with Factual Direction")
    ax.set_title(f"Factual Similarity by Distortion Type (Layer {target_layer})")
    ax.axvline(0, color="gray", linestyle="--", alpha=0.5)
    ax.grid(True, alpha=0.3, axis="x")

    fig.suptitle("H7: Distortion-Specific Subspaces", fontsize=13)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "h7_distortion_subspaces.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\nH7 results saved to {output_dir}")
    return results
