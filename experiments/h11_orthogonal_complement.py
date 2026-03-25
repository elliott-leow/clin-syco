"""Hypothesis 11: Orthogonal Complement — What Do Clinical's Extra Dimensions Encode?

H1 showed clinical sycophancy CONTAINS factual sycophancy (asymmetric probe
transfer). Project out the factual direction and analyze the residual:
  1. Does the orthogonal complement predict emotional intensity level?
  2. Does it predict distortion type?
  3. How does it relate to the empathy direction?

This tells us what clinical sycophancy IS beyond simple agreement.
"""

import json
import os
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict

from pals.extraction import batch_extract_contrastive
from pals.directions import compute_contrastive_direction, cosine_similarity_by_layer
from pals.decomposition import residualize
from pals.probing import prepare_probe_data, train_probe, evaluate_probe, within_domain_probing
from pals.stats import bootstrap_ci


def run(model, tokenizer, stimuli_dir, output_dir, layers=None):
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(stimuli_dir, "cognitive_distortions.json")) as f:
        clinical = json.load(f)
    with open(os.path.join(stimuli_dir, "factual_control.json")) as f:
        factual = json.load(f)
    with open(os.path.join(stimuli_dir, "high_emotion_general.json")) as f:
        emotion = json.load(f)
    with open(os.path.join(stimuli_dir, "emotional_intensity_gradient.json")) as f:
        gradient = json.load(f)

    print("\n=== H11: Orthogonal Complement Analysis ===")

    # Compute directions
    print("Computing clinical sycophancy direction...")
    clin_pos, clin_neg = batch_extract_contrastive(
        model, tokenizer, clinical,
        "sycophantic_completion", "therapeutic_completion",
        layers=layers, desc="Clinical"
    )
    clinical_dir = compute_contrastive_direction(clin_pos, clin_neg)

    print("Computing factual sycophancy direction...")
    fact_pos, fact_neg = batch_extract_contrastive(
        model, tokenizer, factual,
        "sycophantic_completion", "therapeutic_completion",
        layers=layers, desc="Factual"
    )
    factual_dir = compute_contrastive_direction(fact_pos, fact_neg)

    print("Computing empathy direction...")
    emp_pos, emp_neg = batch_extract_contrastive(
        model, tokenizer, emotion,
        "therapeutic_completion", "cold_completion",
        layers=layers, desc="Empathy"
    )
    empathy_dir = compute_contrastive_direction(emp_pos, emp_neg)

    # Compute orthogonal complement: clinical minus factual projection
    all_layers = sorted(clinical_dir.keys())
    ortho_dir = {}
    for l in all_layers:
        resid = residualize(clinical_dir[l], factual_dir[l])
        ortho_dir[l] = F.normalize(resid, dim=0)

    # 1. Cosine similarity: orthogonal complement vs empathy
    cos_ortho_empathy = cosine_similarity_by_layer(ortho_dir, empathy_dir)
    cos_clinical_empathy = cosine_similarity_by_layer(clinical_dir, empathy_dir)
    print(f"\nOrthogonal complement vs empathy: mean cos = {np.mean(list(cos_ortho_empathy.values())):.3f}")
    print(f"Full clinical vs empathy:         mean cos = {np.mean(list(cos_clinical_empathy.values())):.3f}")

    # 2. Can the orthogonal complement predict emotional intensity?
    print("\nTesting: does orthogonal complement predict emotional intensity?")
    levels = sorted(set(s["emotional_level"] for s in gradient))

    # Extract gradient stimulus activations
    level_acts = {}
    for level in levels:
        level_stimuli = [s for s in gradient if s["emotional_level"] == level]
        pos, neg = batch_extract_contrastive(
            model, tokenizer, level_stimuli,
            "sycophantic_completion", "therapeutic_completion",
            layers=layers, desc=f"Gradient L{level}"
        )
        level_acts[level] = (pos, neg)

    # Use middle layer for this analysis
    mid_layer = all_layers[len(all_layers) // 2]

    # Project gradient activations onto orthogonal complement
    # Only use POSITIVE (sycophantic) completions, at mid_layer
    ortho_proj_by_level = {}
    for level in levels:
        pos, neg = level_acts[level]
        projs = []
        for acts in pos:  # Only sycophantic completions
            projs.append((acts[mid_layer] @ ortho_dir[mid_layer]).item())
        ortho_proj_by_level[level] = np.mean(projs)

    print("Mean projection onto orthogonal complement by emotion level:")
    for level in levels:
        label = {1: "Low", 2: "Medium", 3: "High"}.get(level, str(level))
        print(f"  {label}: {ortho_proj_by_level[level]:+.4f}")

    # 3. Can the orthogonal complement predict distortion type?
    print("\nTesting: does orthogonal complement discriminate distortion types?")
    by_subcat = defaultdict(list)
    for s in clinical:
        by_subcat[s["subcategory"]].append(s)

    # Compute mean projection onto orthogonal complement per distortion type
    subcat_projs = {}
    for subcat in sorted(by_subcat.keys()):
        stimuli = by_subcat[subcat]
        projs = []
        for s_idx, s in enumerate(stimuli):
            # Use pre-extracted clinical activations (indices match)
            idx = next(i for i, cs in enumerate(clinical) if cs["id"] == s["id"])
            proj_pos = (clin_pos[idx][mid_layer] @ ortho_dir[mid_layer]).item()
            proj_neg = (clin_neg[idx][mid_layer] @ ortho_dir[mid_layer]).item()
            projs.extend([proj_pos, proj_neg])
        subcat_projs[subcat] = np.mean(projs)

    print(f"Distortion type projections onto orthogonal complement (layer {mid_layer}):")
    for subcat, proj in sorted(subcat_projs.items(), key=lambda x: -abs(x[1])):
        print(f"  {subcat:30s}: {proj:+.4f}")

    # Spread analysis: how much do distortion types vary on this direction?
    proj_values = list(subcat_projs.values())
    proj_range = max(proj_values) - min(proj_values)
    proj_std = np.std(proj_values)

    print(f"\nDistortion type spread on orthogonal complement:")
    print(f"  Range: {proj_range:.4f}, Std: {proj_std:.4f}")

    # Save
    results = {
        "cos_orthogonal_vs_empathy": {str(k): v for k, v in cos_ortho_empathy.items()},
        "cos_clinical_vs_empathy": {str(k): v for k, v in cos_clinical_empathy.items()},
        "mean_cos_ortho_empathy": float(np.mean(list(cos_ortho_empathy.values()))),
        "mean_cos_clinical_empathy": float(np.mean(list(cos_clinical_empathy.values()))),
        "emotion_level_projections": {str(k): float(v) for k, v in ortho_proj_by_level.items()},
        "distortion_projections": {k: float(v) for k, v in subcat_projs.items()},
        "distortion_spread": {"range": float(proj_range), "std": float(proj_std)},
        "analysis_layer": mid_layer,
    }

    with open(os.path.join(output_dir, "h11_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Left: orthogonal vs empathy cosine by layer
    ax = axes[0]
    layers_list = sorted(cos_ortho_empathy.keys())
    ax.plot(layers_list, [cos_ortho_empathy[l] for l in layers_list],
            marker="o", markersize=4, label="Orthogonal comp vs Empathy", color="#ff7f00")
    ax.plot(layers_list, [cos_clinical_empathy[l] for l in layers_list],
            marker="s", markersize=4, label="Full clinical vs Empathy", color="#377eb8")
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Cosine Similarity")
    ax.set_title("Orthogonal Complement vs Empathy")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Middle: emotion level projections
    ax = axes[1]
    level_labels = [f"Level {l}" for l in levels]
    level_vals = [ortho_proj_by_level[l] for l in levels]
    ax.bar(level_labels, level_vals, color=["#4daf4a", "#ff7f00", "#e41a1c"])
    ax.set_ylabel("Mean Projection")
    ax.set_title("Emotion Level on Orthogonal Complement")
    ax.grid(True, alpha=0.3, axis="y")

    # Right: distortion type projections
    ax = axes[2]
    subcats_sorted = sorted(subcat_projs.items(), key=lambda x: -x[1])
    names = [s[:12] for s, _ in subcats_sorted]
    vals = [v for _, v in subcats_sorted]
    ax.barh(range(len(names)), vals, color="#984ea3")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel("Projection")
    ax.set_title(f"Distortion Types (Layer {mid_layer})")
    ax.axvline(0, color="gray", linestyle="--", alpha=0.5)
    ax.grid(True, alpha=0.3, axis="x")

    fig.suptitle("H11: What Does Clinical Sycophancy Encode Beyond Factual Agreement?", fontsize=13)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "h11_orthogonal_complement.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\nH11 results saved to {output_dir}")
    return results
