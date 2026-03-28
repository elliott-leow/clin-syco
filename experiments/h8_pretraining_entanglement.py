"""Hypothesis 8: Empathy-Sycophancy Entanglement Is Pre-Trained.

If the cos ~0.35 alignment between empathy and sycophancy exists even
in the base model, it may originate during pretraining. OLMo publishes
intermediate pretraining checkpoints via git branches.

Analysis:
  Track empathy-sycophancy cosine similarity across pretraining checkpoints
  (by specifying different revisions/steps of the base model).

  For local testing: compare base vs instruct (same as H3).
  For Colab: can use stage1-stepXXX branches for intermediate pretraining.

Note: With base/instruct checkpoints only, this experiment tests whether
empathy-sycophancy alignment exists BEFORE alignment training — which is
consistent with, but does not prove, pretraining origin. The alignment
could also arise from architectural priors or tokenizer structure. True
evidence of pretraining origin requires intermediate pretraining checkpoints
(e.g., OLMo git revisions at different training steps) to show the
entanglement emerging during pretraining itself.
"""

import json
import os
import numpy as np
import torch

from pals.models import load_model, load_tokenizer, OLMO2_CHECKPOINTS, OLMO3_CHECKPOINTS
from pals.extraction import batch_extract_contrastive
from pals.directions import compute_contrastive_direction, cosine_similarity_by_layer
from pals.stats import bootstrap_ci


def run_checkpoint(model, tokenizer, emotion_stimuli, sycophancy_stimuli, layers=None):
    """Compute empathy-sycophancy alignment for one checkpoint."""
    emp_pos, emp_neg = batch_extract_contrastive(
        model, tokenizer, emotion_stimuli,
        "therapeutic_completion", "cold_completion",
        layers=layers, desc="Empathy"
    )
    empathy_dir = compute_contrastive_direction(emp_pos, emp_neg)

    syc_pos, syc_neg = batch_extract_contrastive(
        model, tokenizer, sycophancy_stimuli,
        "sycophantic_completion", "therapeutic_completion",
        layers=layers, desc="Sycophancy"
    )
    sycophancy_dir = compute_contrastive_direction(syc_pos, syc_neg)

    cos_sims = cosine_similarity_by_layer(empathy_dir, sycophancy_dir)
    return empathy_dir, sycophancy_dir, cos_sims


def run(model, tokenizer, stimuli_dir, output_dir,
        checkpoints=None, device="cpu", quantize_4bit=False, layers=None):
    """Run H8: pretraining entanglement analysis.

    Args:
        checkpoints: Dict of name -> model_id (or model_id:revision).
    """
    os.makedirs(output_dir, exist_ok=True)

    if checkpoints is None:
        checkpoints = OLMO2_CHECKPOINTS

    with open(os.path.join(stimuli_dir, "high_emotion_general.json")) as f:
        emotion_stimuli = json.load(f)
    with open(os.path.join(stimuli_dir, "cognitive_distortions.json")) as f:
        sycophancy_stimuli = json.load(f)

    print("\n=== H8: Pretraining Entanglement ===")

    all_cos_sims = {}
    all_mean_sims = {}

    for ckpt_name, model_id in checkpoints.items():
        # Support model_id:revision format
        revision = None
        if ":" in model_id and not model_id.startswith("/"):
            model_id, revision = model_id.rsplit(":", 1)

        print(f"\n--- {ckpt_name}: {model_id}" + (f" (rev: {revision})" if revision else "") + " ---")

        from transformers import AutoModelForCausalLM, AutoTokenizer
        kwargs = {"torch_dtype": torch.float32, "attn_implementation": "eager"}
        if quantize_4bit:
            from transformers import BitsAndBytesConfig
            kwargs = {"quantization_config": BitsAndBytesConfig(load_in_4bit=True,
                      bnb_4bit_compute_dtype=torch.float16), "device_map": "auto",
                      "attn_implementation": "sdpa"}
        elif device == "cuda":
            kwargs = {"torch_dtype": torch.float16, "device_map": "auto",
                      "attn_implementation": "sdpa"}

        if revision:
            kwargs["revision"] = revision

        ckpt_model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        ckpt_model.eval()
        ckpt_tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
        if ckpt_tok.pad_token is None:
            ckpt_tok.pad_token = ckpt_tok.eos_token

        _, _, cos_sims = run_checkpoint(
            ckpt_model, ckpt_tok, emotion_stimuli, sycophancy_stimuli, layers
        )
        all_cos_sims[ckpt_name] = cos_sims
        mean_sim = float(np.mean(list(cos_sims.values())))
        all_mean_sims[ckpt_name] = mean_sim
        print(f"  Mean empathy-sycophancy cos: {mean_sim:.4f}")

        del ckpt_model, ckpt_tok
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        import gc; gc.collect()
        import shutil
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache/huggingface/hub",
                                 "models--" + model_id.replace("/", "--"))
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
            print(f"  Cleared HF cache for {model_id}")

    # Trend
    print(f"\n--- Entanglement Across Checkpoints ---")
    for name in checkpoints:
        print(f"  {name:20s}: {all_mean_sims[name]:+.4f}")

    # Bootstrap CIs
    ci_results = {}
    for name, cos_sims in all_cos_sims.items():
        values = list(cos_sims.values())
        lo, hi = bootstrap_ci(values, ci=0.95)
        ci_results[name] = {"mean": float(np.mean(values)), "ci_lo": lo, "ci_hi": hi}

    # Save
    results = {
        "cosine_by_layer": {
            name: {str(k): v for k, v in sims.items()}
            for name, sims in all_cos_sims.items()
        },
        "mean_similarities": all_mean_sims,
        "bootstrap_ci": ci_results,
        "checkpoints": {k: v for k, v in checkpoints.items()},
    }

    with open(os.path.join(output_dir, "h8_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pals.viz import plot_direction_evolution

    plot_direction_evolution(
        all_cos_sims,
        title="H8: Empathy-Sycophancy Entanglement Across Checkpoints",
        save_path=os.path.join(output_dir, "h8_pretraining_entanglement.png"),
    )

    print(f"\nH8 results saved to {output_dir}")
    return results
