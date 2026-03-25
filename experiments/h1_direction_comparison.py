"""Hypothesis 1: Is clinical sycophancy a different direction from factual sycophancy?

Analysis:
  1. Extract contrastive directions for clinical (cognitive distortion),
     clinical-bridge, and factual sycophancy.
  2. Compute cosine similarity between direction pairs at each layer.
  3. Cross-domain probe transfer: train on one domain, test on another.
  4. Permutation test for significance of direction similarity.
"""

import json
import os
import numpy as np

from pals.extraction import batch_extract_contrastive
from pals.directions import compute_contrastive_direction, cosine_similarity_by_layer
from pals.probing import cross_domain_probing, within_domain_probing
from pals.stats import direction_similarity_permutation, bootstrap_ci
from pals.viz import plot_cosine_similarity_by_layer, plot_probe_transfer, plot_projection_histograms


def run(model, tokenizer, stimuli_dir, output_dir, layers=None):
    """Run H1 experiment.

    Args:
        model: Loaded HuggingFace model.
        tokenizer: Loaded tokenizer.
        stimuli_dir: Path to directory containing stimulus JSON files.
        output_dir: Path to save results.
        layers: Specific layers to analyze, or None for all.
    """
    os.makedirs(output_dir, exist_ok=True)

    # --- Load stimuli ---
    with open(os.path.join(stimuli_dir, "cognitive_distortions.json")) as f:
        clinical = json.load(f)
    with open(os.path.join(stimuli_dir, "clinical_bridge.json")) as f:
        bridge = json.load(f)
    with open(os.path.join(stimuli_dir, "factual_control.json")) as f:
        factual = json.load(f)

    # --- Extract contrastive activations ---
    print("\n=== H1: Direction Comparison ===")
    print("Extracting clinical sycophancy activations...")
    clin_pos, clin_neg = batch_extract_contrastive(
        model, tokenizer, clinical,
        "sycophantic_completion", "therapeutic_completion",
        layers=layers, desc="Clinical"
    )

    print("Extracting bridge sycophancy activations...")
    bridge_pos, bridge_neg = batch_extract_contrastive(
        model, tokenizer, bridge,
        "sycophantic_completion", "therapeutic_completion",
        layers=layers, desc="Bridge"
    )

    print("Extracting factual sycophancy activations...")
    fact_pos, fact_neg = batch_extract_contrastive(
        model, tokenizer, factual,
        "sycophantic_completion", "therapeutic_completion",
        layers=layers, desc="Factual"
    )

    # --- Compute contrastive directions ---
    print("Computing contrastive directions...")
    clin_dir = compute_contrastive_direction(clin_pos, clin_neg)
    bridge_dir = compute_contrastive_direction(bridge_pos, bridge_neg)
    fact_dir = compute_contrastive_direction(fact_pos, fact_neg)

    # --- Cosine similarity between directions ---
    cos_clin_fact = cosine_similarity_by_layer(clin_dir, fact_dir)
    cos_bridge_fact = cosine_similarity_by_layer(bridge_dir, fact_dir)
    cos_clin_bridge = cosine_similarity_by_layer(clin_dir, bridge_dir)

    all_layers = sorted(cos_clin_fact.keys())
    print(f"\nCosine similarities (mean across layers):")
    print(f"  Clinical vs Factual:  {np.mean(list(cos_clin_fact.values())):.3f}")
    print(f"  Bridge vs Factual:    {np.mean(list(cos_bridge_fact.values())):.3f}")
    print(f"  Clinical vs Bridge:   {np.mean(list(cos_clin_bridge.values())):.3f}")

    # --- Cross-domain probing ---
    print("\nRunning cross-domain probing...")
    probe_fact_to_clin = cross_domain_probing(
        fact_pos, fact_neg, clin_pos, clin_neg, all_layers
    )
    probe_clin_to_fact = cross_domain_probing(
        clin_pos, clin_neg, fact_pos, fact_neg, all_layers
    )
    probe_fact_to_bridge = cross_domain_probing(
        fact_pos, fact_neg, bridge_pos, bridge_neg, all_layers
    )
    probe_bridge_to_fact = cross_domain_probing(
        bridge_pos, bridge_neg, fact_pos, fact_neg, all_layers
    )

    # Within-domain baselines
    probe_within_clin = within_domain_probing(clin_pos, clin_neg, all_layers)
    probe_within_fact = within_domain_probing(fact_pos, fact_neg, all_layers)

    print(f"\nCross-domain probe accuracy (mean across layers):")
    print(f"  Factual -> Clinical: {np.mean([v['accuracy'] for v in probe_fact_to_clin.values()]):.3f}")
    print(f"  Clinical -> Factual: {np.mean([v['accuracy'] for v in probe_clin_to_fact.values()]):.3f}")
    print(f"  Factual -> Bridge:   {np.mean([v['accuracy'] for v in probe_fact_to_bridge.values()]):.3f}")
    print(f"  Bridge -> Factual:   {np.mean([v['accuracy'] for v in probe_bridge_to_fact.values()]):.3f}")

    # --- Permutation tests on key layers ---
    print("\nRunning permutation tests (middle and late layers)...")
    test_layers = [all_layers[len(all_layers) // 4],
                   all_layers[len(all_layers) // 2],
                   all_layers[3 * len(all_layers) // 4]]
    perm_results = {}
    for l in test_layers:
        res = direction_similarity_permutation(
            clin_pos, clin_neg, fact_pos, fact_neg, l, n_perms=2000
        )
        perm_results[l] = res
        print(f"  Layer {l}: cos={res['observed_cos']:.3f}, p={res['p_value']:.4f}")

    # --- Save results ---
    results = {
        "cosine_similarities": {
            "clinical_vs_factual": {str(k): v for k, v in cos_clin_fact.items()},
            "bridge_vs_factual": {str(k): v for k, v in cos_bridge_fact.items()},
            "clinical_vs_bridge": {str(k): v for k, v in cos_clin_bridge.items()},
        },
        "cross_domain_probing": {
            "factual_to_clinical": {str(k): v for k, v in probe_fact_to_clin.items()},
            "clinical_to_factual": {str(k): v for k, v in probe_clin_to_fact.items()},
            "factual_to_bridge": {str(k): v for k, v in probe_fact_to_bridge.items()},
            "bridge_to_factual": {str(k): v for k, v in probe_bridge_to_fact.items()},
        },
        "within_domain_probing": {
            "clinical": {str(k): v for k, v in probe_within_clin.items()},
            "factual": {str(k): v for k, v in probe_within_fact.items()},
        },
        "permutation_tests": {
            str(l): {"observed_cos": r["observed_cos"], "p_value": r["p_value"]}
            for l, r in perm_results.items()
        },
    }

    with open(os.path.join(output_dir, "h1_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # --- Plots ---
    plot_cosine_similarity_by_layer(
        {
            "Clinical vs Factual": cos_clin_fact,
            "Bridge vs Factual": cos_bridge_fact,
            "Clinical vs Bridge": cos_clin_bridge,
        },
        title="H1: Sycophancy Direction Similarity Across Domains",
        save_path=os.path.join(output_dir, "h1_cosine_similarity.png"),
    )

    plot_probe_transfer(
        {
            "Factual -> Clinical": probe_fact_to_clin,
            "Clinical -> Factual": probe_clin_to_fact,
            "Within Clinical (CV)": {l: {"accuracy": v["mean_accuracy"], "auc": 0}
                                     for l, v in probe_within_clin.items()},
            "Within Factual (CV)": {l: {"accuracy": v["mean_accuracy"], "auc": 0}
                                    for l, v in probe_within_fact.items()},
        },
        title="H1: Sycophancy Probe Transfer",
        save_path=os.path.join(output_dir, "h1_probe_transfer.png"),
    )

    print(f"\nH1 results saved to {output_dir}")
    return results
