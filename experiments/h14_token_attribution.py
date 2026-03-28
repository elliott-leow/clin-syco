"""Hypothesis 14: Token Attribution — Which Prompt Tokens Trigger Sycophancy?

Compute which input tokens most strongly activate the clinical sycophancy
direction. Uses projection-based attribution: for each token position, measure
the absolute projection of its hidden state onto the sycophancy direction.

If emotion words ("devastated", "terrified") dominate, the trigger is affect.
If first-person pronouns ("I feel", "I think") dominate, it's self-reference.
If cognitive distortion markers ("always", "never", "everything") dominate,
the model is detecting the distortion itself.
"""

import json
import os
import numpy as np
import torch
from collections import Counter

from pals.extraction import batch_extract_contrastive
from pals.directions import compute_contrastive_direction
from pals.models import get_device


def run(model, tokenizer, stimuli_dir, output_dir, layers=None, n_stimuli=15, target_layer=None):
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(stimuli_dir, "cognitive_distortions.json")) as f:
        all_clinical = json.load(f)
    clinical = all_clinical[:n_stimuli]
    with open(os.path.join(stimuli_dir, "factual_control.json")) as f:
        factual = json.load(f)[:n_stimuli]

    n_layers = model.config.num_hidden_layers
    if target_layer is None:
        target_layer = 2 * n_layers // 3
    # Snap to nearest extracted layer if using a subset
    if layers is not None and target_layer not in set(layers):
        target_layer = min(layers, key=lambda l: abs(l - target_layer))

    print(f"\n=== H14: Token Attribution ===")
    print(f"Target layer: {target_layer}")

    # Compute sycophancy direction (use all clinical stimuli for stable direction)
    print("Computing clinical sycophancy direction...")
    clin_pos, clin_neg = batch_extract_contrastive(
        model, tokenizer, all_clinical,
        "sycophantic_completion", "therapeutic_completion",
        layers=layers, desc="Clinical"
    )
    syc_direction = compute_contrastive_direction(clin_pos, clin_neg)
    target_dir = syc_direction[target_layer]

    device = get_device(model)

    # Attribution via input gradient
    print("\nComputing token attributions...")
    all_attributions = []
    token_importance_counter = Counter()
    token_count = Counter()

    from pals.extraction import extract_activations
    import gc
    target_dir_gpu = target_dir.to(device)

    for si, s in enumerate(clinical):
        input_ids = tokenizer.encode(s["user_prompt"], return_tensors="pt").to(device)
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

        with torch.no_grad():
            # Extract per-token hidden states at target layer (seq_len, hidden_dim)
            acts = extract_activations(model, input_ids, layers=[target_layer])
            hidden = acts[target_layer]  # (seq_len, hidden_dim) on CPU

            # Attribution = absolute projection of each token's hidden state
            # onto the sycophancy direction. Tokens whose representations align
            # strongly with the sycophancy direction are the triggers.
            projections = (hidden @ target_dir.unsqueeze(1)).squeeze().abs().numpy()
            attr = projections / (projections.sum() + 1e-10)

        all_attributions.append({
            "tokens": tokens,
            "attributions": attr.tolist(),
            "prompt": s["user_prompt"][:100],
        })

        for tok, imp in zip(tokens, attr):
            clean_tok = tok.replace("▁", "").replace("Ġ", "").lower().strip()
            if len(clean_tok) > 1:
                token_importance_counter[clean_tok] += imp
                token_count[clean_tok] += 1

        if (si + 1) % 5 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    # Compute mean importance per unique token
    mean_importance = {}
    for tok in token_importance_counter:
        if token_count[tok] >= 2:  # Appeared in at least 2 prompts
            mean_importance[tok] = token_importance_counter[tok] / token_count[tok]

    # Top tokens
    top_tokens = sorted(mean_importance.items(), key=lambda x: -x[1])[:30]
    print(f"\nTop 30 tokens by mean attribution (across {len(all_attributions)} prompts):")
    for tok, imp in top_tokens:
        print(f"  {tok:20s}: {imp:.4f} (appeared {token_count[tok]}x)")

    # Categorize top tokens
    emotion_words = {"feel", "feeling", "scared", "terrified", "devastated", "horrible",
                     "awful", "anxious", "depressed", "worried", "upset", "angry",
                     "hopeless", "worthless", "ashamed", "guilty", "afraid", "panic"}
    distortion_markers = {"always", "never", "everything", "nothing", "everyone",
                          "nobody", "completely", "totally", "impossible", "ruined",
                          "worst", "failure", "disaster", "catastrophe"}
    self_ref = {"i", "me", "my", "mine", "myself"}

    top_token_set = {tok for tok, _ in top_tokens}
    emotion_hits = top_token_set & emotion_words
    distortion_hits = top_token_set & distortion_markers
    self_ref_hits = top_token_set & self_ref

    print(f"\nCategory analysis of top-30 tokens:")
    print(f"  Emotion words: {len(emotion_hits)} — {emotion_hits}")
    print(f"  Distortion markers: {len(distortion_hits)} — {distortion_hits}")
    print(f"  Self-reference: {len(self_ref_hits)} — {self_ref_hits}")

    # Save
    results = {
        "target_layer": target_layer,
        "top_tokens": [{"token": t, "mean_attribution": float(v), "count": token_count[t]}
                       for t, v in top_tokens],
        "category_hits": {
            "emotion_words": list(emotion_hits),
            "distortion_markers": list(distortion_hits),
            "self_reference": list(self_ref_hits),
        },
        "per_prompt_attributions": all_attributions[:5],  # Save first 5 for inspection
    }

    with open(os.path.join(output_dir, "h14_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: top 20 tokens bar chart
    ax = axes[0]
    top20 = top_tokens[:20]
    names = [t for t, _ in top20]
    vals = [v for _, v in top20]
    colors = []
    for t, _ in top20:
        if t in emotion_words:
            colors.append("#e41a1c")
        elif t in distortion_markers:
            colors.append("#377eb8")
        elif t in self_ref:
            colors.append("#4daf4a")
        else:
            colors.append("#999999")
    ax.barh(range(len(names)), vals, color=colors)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Mean Attribution Score")
    ax.set_title("Top 20 Sycophancy-Triggering Tokens")
    # Legend
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color="#e41a1c", label="Emotion"),
        Patch(color="#377eb8", label="Distortion marker"),
        Patch(color="#4daf4a", label="Self-reference"),
        Patch(color="#999999", label="Other"),
    ], fontsize=8)
    ax.grid(True, alpha=0.3, axis="x")

    # Right: example attribution heatmap (first prompt)
    ax = axes[1]
    if all_attributions:
        ex = all_attributions[0]
        tokens = ex["tokens"][:30]
        attrs = ex["attributions"][:30]
        ax.barh(range(len(tokens)), attrs, color="#ff7f00")
        ax.set_yticks(range(len(tokens)))
        ax.set_yticklabels(tokens, fontsize=6)
        ax.set_xlabel("Attribution")
        ax.set_title(f"Example: {ex['prompt'][:50]}...")
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis="x")

    fig.suptitle("H14: Token Attribution — What Triggers Clinical Sycophancy?", fontsize=13)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "h14_token_attribution.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\nH14 results saved to {output_dir}")
    return results
