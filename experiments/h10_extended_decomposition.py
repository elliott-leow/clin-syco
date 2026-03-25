"""Hypothesis 10: Extended Decomposition — What IS the 83% Residual?

H9 found clinical sycophancy is 83% unique (not empathy + factual agreement).
Decompose against 5 reference directions to identify what the residual contains:
  1. Empathy (therapeutic vs cold on proportionate distress)
  2. Factual sycophancy (sycophantic vs therapeutic on factual errors)
  3. Conflict-avoidance (sycophantic vs cold on factual errors)
  4. Clinical warmth (therapeutic vs cold on cognitive distortions)
  5. Framing acceptance (sycophantic vs cold on cognitive distortions)

If conflict-avoidance explains a large chunk, clinical sycophancy = "don't disagree
with distressed people." If clinical warmth dominates, it's empathy-in-context.
If the residual stays >60%, it's genuinely novel.
"""

import json
import os
import numpy as np

from pals.extraction import batch_extract_contrastive
from pals.directions import compute_contrastive_direction
from pals.decomposition import decompose_by_layer
from pals.stats import bootstrap_ci


def run(model, tokenizer, stimuli_dir, output_dir, layers=None):
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(stimuli_dir, "cognitive_distortions.json")) as f:
        clinical = json.load(f)
    with open(os.path.join(stimuli_dir, "factual_control.json")) as f:
        factual = json.load(f)
    with open(os.path.join(stimuli_dir, "high_emotion_general.json")) as f:
        emotion = json.load(f)

    print("\n=== H10: Extended Decomposition ===")

    # Target: clinical sycophancy direction
    print("Computing clinical sycophancy direction...")
    clin_pos, clin_neg = batch_extract_contrastive(
        model, tokenizer, clinical,
        "sycophantic_completion", "therapeutic_completion",
        layers=layers, desc="Clinical syc"
    )
    clinical_dir = compute_contrastive_direction(clin_pos, clin_neg)

    # Reference direction 1: empathy (therapeutic vs cold on emotion stimuli)
    print("Computing empathy direction...")
    emp_pos, emp_neg = batch_extract_contrastive(
        model, tokenizer, emotion,
        "therapeutic_completion", "cold_completion",
        layers=layers, desc="Empathy"
    )
    empathy_dir = compute_contrastive_direction(emp_pos, emp_neg)

    # Reference direction 2: factual sycophancy
    print("Computing factual sycophancy direction...")
    fact_pos, fact_neg = batch_extract_contrastive(
        model, tokenizer, factual,
        "sycophantic_completion", "therapeutic_completion",
        layers=layers, desc="Factual syc"
    )
    factual_dir = compute_contrastive_direction(fact_pos, fact_neg)

    # Reference direction 3: conflict-avoidance (sycophantic vs cold on factual)
    print("Computing conflict-avoidance direction...")
    ca_pos, ca_neg = batch_extract_contrastive(
        model, tokenizer, factual,
        "sycophantic_completion", "cold_completion",
        layers=layers, desc="Conflict-avoid"
    )
    conflict_avoid_dir = compute_contrastive_direction(ca_pos, ca_neg)

    # Reference direction 4: clinical warmth (therapeutic vs cold on clinical)
    print("Computing clinical warmth direction...")
    cw_pos, cw_neg = batch_extract_contrastive(
        model, tokenizer, clinical,
        "therapeutic_completion", "cold_completion",
        layers=layers, desc="Clinical warmth"
    )
    clinical_warmth_dir = compute_contrastive_direction(cw_pos, cw_neg)

    # Reference direction 5: framing acceptance (sycophantic vs cold on clinical)
    print("Computing framing acceptance direction...")
    fa_pos, fa_neg = batch_extract_contrastive(
        model, tokenizer, clinical,
        "sycophantic_completion", "cold_completion",
        layers=layers, desc="Framing accept"
    )
    framing_accept_dir = compute_contrastive_direction(fa_pos, fa_neg)

    # Decompose: 2-component (original H9) vs 5-component
    print("\nDecomposing with 2 components (H9 replication)...")
    decomp_2 = decompose_by_layer(
        clinical_dir,
        {"empathy": empathy_dir, "factual_sycophancy": factual_dir}
    )

    print("Decomposing with 5 components...")
    decomp_5 = decompose_by_layer(
        clinical_dir,
        {
            "empathy": empathy_dir,
            "factual_sycophancy": factual_dir,
            "conflict_avoidance": conflict_avoid_dir,
            "clinical_warmth": clinical_warmth_dir,
            "framing_acceptance": framing_accept_dir,
        }
    )

    # Summarize
    all_layers = sorted(decomp_5.keys())

    print(f"\n{'Layer':>6}  {'Empathy':>8}  {'Factual':>8}  {'Conflict':>8}  {'ClinWrm':>8}  {'Frame':>8}  {'Resid':>8}")
    for layer in all_layers:
        d = decomp_5[layer]
        ve = d["variance_explained"]
        res = d["residual_variance_fraction"]
        print(f"{layer:>6}  {ve['empathy']:>7.1%}  {ve['factual_sycophancy']:>7.1%}  "
              f"{ve['conflict_avoidance']:>7.1%}  {ve['clinical_warmth']:>7.1%}  "
              f"{ve['framing_acceptance']:>7.1%}  {res:>7.1%}")

    mean_ve_5 = {}
    for comp in ["empathy", "factual_sycophancy", "conflict_avoidance",
                 "clinical_warmth", "framing_acceptance"]:
        mean_ve_5[comp] = float(np.mean([decomp_5[l]["variance_explained"][comp] for l in all_layers]))
    mean_res_5 = float(np.mean([decomp_5[l]["residual_variance_fraction"] for l in all_layers]))
    mean_res_2 = float(np.mean([decomp_2[l]["residual_variance_fraction"] for l in all_layers]))

    print(f"\nMean variance explained (5-component):")
    for comp, ve in sorted(mean_ve_5.items(), key=lambda x: -x[1]):
        print(f"  {comp:25s}: {ve:.1%}")
    print(f"  {'residual':25s}: {mean_res_5:.1%}")
    print(f"\nResidual reduction: {mean_res_2:.1%} (2-comp) -> {mean_res_5:.1%} (5-comp) = {mean_res_2 - mean_res_5:+.1%}")

    # Save
    results = {
        "decomposition_2comp": {
            str(l): {
                "variance_explained": decomp_2[l]["variance_explained"],
                "unique_variance_explained": decomp_2[l]["unique_variance_explained"],
                "residual": decomp_2[l]["residual_variance_fraction"],
            } for l in all_layers
        },
        "decomposition_5comp": {
            str(l): {
                "variance_explained": decomp_5[l]["variance_explained"],
                "unique_variance_explained": decomp_5[l]["unique_variance_explained"],
                "residual": decomp_5[l]["residual_variance_fraction"],
            } for l in all_layers
        },
        "mean_variance_explained_5comp": mean_ve_5,
        "mean_residual_2comp": mean_res_2,
        "mean_residual_5comp": mean_res_5,
    }

    with open(os.path.join(output_dir, "h10_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Plot: stacked bar comparing 2-comp vs 5-comp
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: 5-component stacked area
    ax = axes[0]
    comp_names = ["empathy", "factual_sycophancy", "conflict_avoidance",
                  "clinical_warmth", "framing_acceptance"]
    comp_colors = ["#ff7f00", "#377eb8", "#e41a1c", "#4daf4a", "#984ea3"]
    vals = [[decomp_5[l]["variance_explained"].get(c, 0) for l in all_layers] for c in comp_names]
    res_vals = [decomp_5[l]["residual_variance_fraction"] for l in all_layers]
    vals.append(res_vals)
    comp_names_plot = ["Empathy", "Factual Syc", "Conflict Avoid", "Clinical Warmth", "Frame Accept"]
    comp_colors.append("#999999")
    comp_names_plot.append("Residual")
    ax.stackplot(all_layers, *vals, labels=comp_names_plot, colors=comp_colors, alpha=0.8)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Variance Explained")
    ax.set_title("5-Component Decomposition")
    ax.legend(loc="upper right", fontsize=7)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    # Right: comparison bar chart (mean across layers)
    ax = axes[1]
    labels_2 = ["Empathy", "Factual", "Residual"]
    mean_ve_2 = {
        "empathy": np.mean([decomp_2[l]["variance_explained"]["empathy"] for l in all_layers]),
        "factual": np.mean([decomp_2[l]["variance_explained"]["factual_sycophancy"] for l in all_layers]),
    }
    vals_2 = [mean_ve_2["empathy"], mean_ve_2["factual"], mean_res_2]
    vals_5_bar = [mean_ve_5[c] for c in ["empathy", "factual_sycophancy", "conflict_avoidance",
                                          "clinical_warmth", "framing_acceptance"]]
    vals_5_bar.append(mean_res_5)
    labels_5 = ["Empathy", "Factual", "Conflict\nAvoid", "Clinical\nWarmth", "Frame\nAccept", "Residual"]

    x2 = np.arange(len(labels_2))
    x5 = np.arange(len(labels_5))
    ax.bar(x2 - 0.15, vals_2, 0.3, label="2-component (H9)", color="#377eb8", alpha=0.7)
    colors_5 = comp_colors
    for i, (v, c) in enumerate(zip(vals_5_bar, colors_5)):
        ax.bar(i + 0.15 if i < 2 else i + 0.85, v, 0.3, color=c, alpha=0.7,
               label="5-component" if i == 0 else None)

    all_x = list(range(len(labels_5)))
    ax.set_xticks(all_x)
    ax.set_xticklabels(labels_5, fontsize=8)
    ax.set_ylabel("Mean Variance Explained")
    ax.set_title(f"Residual: {mean_res_2:.0%} (2-comp) → {mean_res_5:.0%} (5-comp)")
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("H10: Extended Decomposition — What IS the 83% Residual?", fontsize=13)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "h10_extended_decomposition.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\nH10 results saved to {output_dir}")
    return results
