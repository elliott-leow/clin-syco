"""Hypothesis 3: Does preference optimization conflate empathy and sycophancy?

Analysis:
  1. At each checkpoint (base -> SFT -> DPO), extract:
     - Empathy direction: therapeutic vs cold (from high_emotion_general)
       Isolates warmth/validation since both are appropriate in content.
     - Sycophancy direction: sycophantic vs therapeutic (from cognitive_distortions)
       Isolates frame-acceptance since both are warm in tone.
  2. Compute cosine similarity between empathy and sycophancy directions.
  3. Track how this similarity evolves across training stages.
  4. If DPO conflates: similarity increases base -> SFT -> DPO.

Note: With OLMo-2 (base/instruct only), this experiment tests whether
alignment training in general changes empathy-sycophancy alignment. The
DPO-specific claim (that DPO *specifically* conflates empathy and sycophancy)
requires OLMo-3 base/SFT/DPO checkpoints, which are available on Colab.
"""

import json
import os
import numpy as np
import torch

from pals.models import load_model, load_tokenizer, OLMO2_CHECKPOINTS, OLMO3_CHECKPOINTS
from pals.extraction import batch_extract_contrastive
from pals.directions import compute_contrastive_direction, cosine_similarity_by_layer
from pals.stats import bootstrap_ci
from pals.viz import plot_direction_evolution, plot_cosine_similarity_by_layer


def run_single_checkpoint(model, tokenizer, emotion_stimuli, sycophancy_stimuli,
                          layers=None):
    """Compute empathy and sycophancy directions for a single checkpoint.

    Returns:
        (empathy_direction, sycophancy_direction, cosine_sims)
    """
    # Empathy direction: therapeutic (warm) vs cold (no warmth)
    # Both respond to proportionate distress, differing only in warmth.
    emp_pos, emp_neg = batch_extract_contrastive(
        model, tokenizer, emotion_stimuli,
        "therapeutic_completion", "cold_completion",
        layers=layers, desc="Empathy"
    )
    empathy_dir = compute_contrastive_direction(emp_pos, emp_neg)

    # Sycophancy direction: sycophantic (agrees) vs therapeutic (challenges)
    # Both are warm, differing only in frame-acceptance.
    syc_pos, syc_neg = batch_extract_contrastive(
        model, tokenizer, sycophancy_stimuli,
        "sycophantic_completion", "therapeutic_completion",
        layers=layers, desc="Sycophancy"
    )
    sycophancy_dir = compute_contrastive_direction(syc_pos, syc_neg)

    cos_sims = cosine_similarity_by_layer(empathy_dir, sycophancy_dir)
    return empathy_dir, sycophancy_dir, cos_sims


def run(model, tokenizer, stimuli_dir, output_dir, checkpoints=None,
        device="cpu", quantize_4bit=False, layers=None):
    """Run H3 experiment across multiple checkpoints.

    Args:
        model: Not used if checkpoints is provided (loads its own models).
        tokenizer: Not used if checkpoints is provided.
        stimuli_dir: Path to stimulus JSON files.
        output_dir: Path to save results.
        checkpoints: Dict mapping name -> model_id. Defaults to OLMO2_CHECKPOINTS.
        device: Device for model loading.
        quantize_4bit: Whether to quantize.
        layers: Specific layers or None for all.
    """
    os.makedirs(output_dir, exist_ok=True)

    if checkpoints is None:
        checkpoints = OLMO2_CHECKPOINTS

    # --- Load stimuli ---
    with open(os.path.join(stimuli_dir, "high_emotion_general.json")) as f:
        emotion_stimuli = json.load(f)
    with open(os.path.join(stimuli_dir, "cognitive_distortions.json")) as f:
        sycophancy_stimuli = json.load(f)

    print("\n=== H3: Empathy-Sycophancy Conflation Across Training ===")

    # --- Run on each checkpoint ---
    all_cos_sims = {}
    all_mean_sims = {}

    for ckpt_name, model_id in checkpoints.items():
        print(f"\n--- Checkpoint: {ckpt_name} ({model_id}) ---")

        ckpt_model = load_model(model_id, device=device, quantize_4bit=quantize_4bit)
        ckpt_tokenizer = load_tokenizer(model_id)

        emp_dir, syc_dir, cos_sims = run_single_checkpoint(
            ckpt_model, ckpt_tokenizer, emotion_stimuli, sycophancy_stimuli,
            layers=layers
        )

        all_cos_sims[ckpt_name] = cos_sims
        mean_sim = float(np.mean(list(cos_sims.values())))
        all_mean_sims[ckpt_name] = mean_sim

        print(f"  Mean empathy-sycophancy cosine similarity: {mean_sim:.4f}")

        # Free memory and disk (HF cache can be 60-130GB per checkpoint)
        del ckpt_model, ckpt_tokenizer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        import gc; gc.collect()
        import shutil
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache/huggingface/hub",
                                 "models--" + model_id.replace("/", "--"))
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
            print(f"  Cleared HF cache for {model_id}")

    # --- Trend analysis ---
    ckpt_names = list(checkpoints.keys())
    mean_values = [all_mean_sims[n] for n in ckpt_names]

    print(f"\n--- Trend: Empathy-Sycophancy Alignment ---")
    for name, val in zip(ckpt_names, mean_values):
        print(f"  {name:12s}: {val:+.4f}")

    if len(mean_values) >= 2:
        trend = mean_values[-1] - mean_values[0]
        print(f"  Change ({ckpt_names[0]} -> {ckpt_names[-1]}): {trend:+.4f}")
        if trend > 0.05:
            print("  -> CONFLATION: Empathy and sycophancy directions become more aligned.")
        elif trend < -0.05:
            print("  -> SEPARATION: Training separates empathy from sycophancy.")
        else:
            print("  -> STABLE: No clear trend in alignment.")

    # --- Bootstrap CIs on mean similarity ---
    print(f"\nBootstrap CIs on mean similarity per checkpoint:")
    ci_results = {}
    for ckpt_name, cos_sims in all_cos_sims.items():
        values = list(cos_sims.values())
        lo, hi = bootstrap_ci(values, ci=0.95)
        ci_results[ckpt_name] = {"mean": float(np.mean(values)), "ci_lo": lo, "ci_hi": hi}
        print(f"  {ckpt_name}: {np.mean(values):.4f} [{lo:.4f}, {hi:.4f}]")

    # --- Save results ---
    results = {
        "cosine_similarities_by_layer": {
            name: {str(k): v for k, v in sims.items()}
            for name, sims in all_cos_sims.items()
        },
        "mean_similarities": {str(k): v for k, v in all_mean_sims.items()},
        "bootstrap_ci": ci_results,
        "checkpoints": checkpoints,
    }

    with open(os.path.join(output_dir, "h3_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # --- Plots ---
    plot_direction_evolution(
        all_cos_sims,
        title="H3: Empathy-Sycophancy Direction Alignment Across Training",
        save_path=os.path.join(output_dir, "h3_direction_evolution.png"),
    )

    print(f"\nH3 results saved to {output_dir}")
    return results
