"""Hypothesis 9: The Warmth Tax — Does Clinical Sycophancy Reduce to Empathy + Agreement?

Decompose the clinical sycophancy direction into:
  - Component along empathy direction
  - Component along factual sycophancy direction
  - Unique residual

If most of what's "unique" about clinical sycophancy IS empathy, that's
evidence that warmth and agreement are conflated.

Analysis:
  1. Compute empathy, factual sycophancy, and clinical sycophancy directions.
  2. Decompose clinical sycophancy into empathy + factual + residual.
  3. Measure variance explained by each component.

Note: This experiment tests whether the warmth tax pattern exists (i.e.,
whether clinical sycophancy decomposes primarily into empathy and factual
agreement components), not whether safety training caused it. The causal
claim would require cross-checkpoint decomposition showing the pattern
emerges or strengthens during training.
"""

import json
import os
import numpy as np

from pals.extraction import batch_extract_contrastive
from pals.directions import compute_contrastive_direction
from pals.decomposition import decompose_by_layer


def run(model, tokenizer, stimuli_dir, output_dir, layers=None):
    """Run H9 experiment."""
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(stimuli_dir, "cognitive_distortions.json")) as f:
        clinical = json.load(f)
    with open(os.path.join(stimuli_dir, "factual_control.json")) as f:
        factual = json.load(f)
    with open(os.path.join(stimuli_dir, "high_emotion_general.json")) as f:
        emotion = json.load(f)

    print("\n=== H9: Warmth Tax Decomposition ===")

    # 1. Clinical sycophancy direction: sycophantic vs therapeutic (from clinical)
    print("Computing clinical sycophancy direction...")
    clin_pos, clin_neg = batch_extract_contrastive(
        model, tokenizer, clinical,
        "sycophantic_completion", "therapeutic_completion",
        layers=layers, desc="Clinical syc"
    )
    clinical_dir = compute_contrastive_direction(clin_pos, clin_neg)

    # 2. Factual sycophancy direction: sycophantic vs therapeutic (from factual)
    print("Computing factual sycophancy direction...")
    fact_pos, fact_neg = batch_extract_contrastive(
        model, tokenizer, factual,
        "sycophantic_completion", "therapeutic_completion",
        layers=layers, desc="Factual syc"
    )
    factual_dir = compute_contrastive_direction(fact_pos, fact_neg)

    # 3. Empathy direction: therapeutic vs cold (from emotion)
    print("Computing empathy direction...")
    emp_pos, emp_neg = batch_extract_contrastive(
        model, tokenizer, emotion,
        "therapeutic_completion", "cold_completion",
        layers=layers, desc="Empathy"
    )
    empathy_dir = compute_contrastive_direction(emp_pos, emp_neg)

    # 4. Decompose clinical sycophancy into empathy + factual + residual
    print("\nDecomposing clinical sycophancy direction...")
    decomp = decompose_by_layer(
        clinical_dir,
        {"empathy": empathy_dir, "factual_sycophancy": factual_dir}
    )

    # Print results
    all_layers = sorted(decomp.keys())
    print(f"\n{'Layer':>6}  {'Empathy':>10}  {'Factual':>10}  {'Residual':>10}  {'Emp(u)':>8}  {'Fact(u)':>8}  {'Res%':>8}")
    for layer in all_layers:
        d = decomp[layer]
        emp_ve = d["unique_variance_explained"]["empathy"]
        fact_ve = d["unique_variance_explained"]["factual_sycophancy"]
        res_ve = d["residual_variance_fraction"]
        emp_proj = d["projections"]["empathy"]
        fact_proj = d["projections"]["factual_sycophancy"]
        res_norm = d["residual_norm"]
        print(f"{layer:>6}  {emp_proj:>+10.3f}  {fact_proj:>+10.3f}  {res_norm:>10.3f}"
              f"  {emp_ve:>7.1%}  {fact_ve:>7.1%}  {res_ve:>7.1%}")

    # Summary
    mean_emp_ve = float(np.mean([decomp[l]["unique_variance_explained"]["empathy"] for l in all_layers]))
    mean_fact_ve = float(np.mean([decomp[l]["unique_variance_explained"]["factual_sycophancy"] for l in all_layers]))
    mean_res_ve = float(np.mean([decomp[l]["residual_variance_fraction"] for l in all_layers]))

    print(f"\nMean unique variance explained across layers:")
    print(f"  Empathy component (unique):           {mean_emp_ve:.1%}")
    print(f"  Factual sycophancy component (unique): {mean_fact_ve:.1%}")
    print(f"  Residual:                              {mean_res_ve:.1%}")

    if mean_emp_ve > mean_fact_ve and mean_emp_ve > 0.15:
        print("\n-> WARMTH TAX: Clinical sycophancy is dominated by empathy component (unique variance).")
    elif mean_res_ve > 0.6:
        print("\n-> NOVEL MECHANISM: Clinical sycophancy has large unique component.")
    else:
        print("\n-> MIXED: Clinical sycophancy draws from both empathy and factual agreement.")

    # Save
    results = {
        "decomposition_by_layer": {
            str(l): {
                "empathy_projection": decomp[l]["projections"]["empathy"],
                "factual_projection": decomp[l]["projections"]["factual_sycophancy"],
                "empathy_variance_explained": decomp[l]["variance_explained"]["empathy"],
                "factual_variance_explained": decomp[l]["variance_explained"]["factual_sycophancy"],
                "empathy_unique_ve": decomp[l]["unique_variance_explained"]["empathy"],
                "factual_unique_ve": decomp[l]["unique_variance_explained"]["factual_sycophancy"],
                "residual_variance_fraction": decomp[l]["residual_variance_fraction"],
                "residual_norm": decomp[l]["residual_norm"],
            }
            for l in all_layers
        },
        "mean_unique_variance_explained": {
            "empathy": mean_emp_ve,
            "factual_sycophancy": mean_fact_ve,
            "residual": mean_res_ve,
        },
    }

    with open(os.path.join(output_dir, "h9_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Plot: stacked area chart of variance explained
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: variance explained stacked
    ax = axes[0]
    emp_vals = [decomp[l]["variance_explained"]["empathy"] for l in all_layers]
    fact_vals = [decomp[l]["variance_explained"]["factual_sycophancy"] for l in all_layers]
    res_vals = [decomp[l]["residual_variance_fraction"] for l in all_layers]

    ax.stackplot(all_layers, emp_vals, fact_vals, res_vals,
                 labels=["Empathy", "Factual Sycophancy", "Unique Residual"],
                 colors=["#ff7f00", "#377eb8", "#999999"], alpha=0.8)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Variance Explained")
    ax.set_title("Decomposition of Clinical Sycophancy")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    # Right: projection magnitudes
    ax = axes[1]
    ax.plot(all_layers, [decomp[l]["projections"]["empathy"] for l in all_layers],
            marker="o", markersize=4, label="Empathy", color="#ff7f00")
    ax.plot(all_layers, [decomp[l]["projections"]["factual_sycophancy"] for l in all_layers],
            marker="s", markersize=4, label="Factual Sycophancy", color="#377eb8")
    ax.plot(all_layers, [decomp[l]["residual_norm"] for l in all_layers],
            marker="^", markersize=4, label="Residual Norm", color="#999999")
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Projection / Norm")
    ax.set_title("Component Magnitudes")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle("H9: Warmth Tax — Clinical Sycophancy Direction Decomposition", fontsize=13)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "h9_warmth_tax.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\nH9 results saved to {output_dir}")
    return results
