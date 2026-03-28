"""Hypothesis 15: Full Clinical Sycophancy Circuit Decomposition.

The 63% residual from H10 means most of clinical sycophancy is unexplained
by known reference directions. This experiment uses data-driven methods to
characterize the FULL circuit:

1. Logit lens on the sycophancy direction — project through unembedding at
   each layer to see which TOKENS the direction points toward. This reveals
   what the direction "means" in vocabulary space at each processing stage.

2. Logit lens on the residual direction — same analysis on the 63% that's
   unexplained. What tokens does the unknown component point toward?

3. Layer-wise variance growth — at which layers does the clinical sycophancy
   direction gain its norm? Where is the signal computed?

4. Per-stimulus residual clustering — the residual might not be one thing.
   Compute per-stimulus residuals and cluster them.

5. Divergence mapping — at which layer does clinical sycophancy diverge
   from factual sycophancy? Map the "branching point."
"""

import json
import os
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict

from pals.extraction import batch_extract_contrastive
from pals.directions import compute_contrastive_direction, cosine_similarity_by_layer
from pals.decomposition import decompose_by_layer
from pals.models import get_device


def direction_to_tokens(model, direction, top_k=20):
    """Project a direction through the unembedding matrix to get top tokens.

    Returns list of (token_id, logit_value) tuples.
    """
    norm = model.model.norm
    lm_head = model.lm_head

    # Apply final layernorm then unembedding
    device = get_device(model)
    model_dtype = next(model.parameters()).dtype
    d = direction.unsqueeze(0).unsqueeze(0).to(device=device, dtype=model_dtype)
    normed = norm(d)
    logits = lm_head(normed).squeeze().float().cpu()

    # Top tokens in the direction
    top_vals, top_ids = torch.topk(logits, top_k)
    # Bottom tokens (anti-direction)
    bot_vals, bot_ids = torch.topk(-logits, top_k)

    return {
        "top": [(idx.item(), val.item()) for idx, val in zip(top_ids, top_vals)],
        "bottom": [(idx.item(), -val.item()) for idx, val in zip(bot_ids, bot_vals)],
    }


def run(model, tokenizer, stimuli_dir, output_dir, layers=None):
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(stimuli_dir, "cognitive_distortions.json")) as f:
        clinical = json.load(f)
    with open(os.path.join(stimuli_dir, "factual_control.json")) as f:
        factual = json.load(f)
    with open(os.path.join(stimuli_dir, "high_emotion_general.json")) as f:
        emotion = json.load(f)

    print("\n=== H15: Full Circuit Decomposition ===")

    # --- Extract all directions ---
    print("Extracting clinical sycophancy activations...")
    clin_pos, clin_neg = batch_extract_contrastive(
        model, tokenizer, clinical,
        "sycophantic_completion", "therapeutic_completion",
        layers=layers, desc="Clinical"
    )
    clinical_dir = compute_contrastive_direction(clin_pos, clin_neg)

    print("Extracting factual sycophancy activations...")
    fact_pos, fact_neg = batch_extract_contrastive(
        model, tokenizer, factual,
        "sycophantic_completion", "therapeutic_completion",
        layers=layers, desc="Factual"
    )
    factual_dir = compute_contrastive_direction(fact_pos, fact_neg)

    print("Extracting empathy activations...")
    emp_pos, emp_neg = batch_extract_contrastive(
        model, tokenizer, emotion,
        "therapeutic_completion", "cold_completion",
        layers=layers, desc="Empathy"
    )
    empathy_dir = compute_contrastive_direction(emp_pos, emp_neg)

    # Framing acceptance and conflict avoidance
    print("Extracting framing acceptance activations...")
    fa_pos, fa_neg = batch_extract_contrastive(
        model, tokenizer, clinical,
        "sycophantic_completion", "cold_completion",
        layers=layers, desc="Frame accept"
    )
    framing_dir = compute_contrastive_direction(fa_pos, fa_neg)

    ca_pos, ca_neg = batch_extract_contrastive(
        model, tokenizer, factual,
        "sycophantic_completion", "cold_completion",
        layers=layers, desc="Conflict avoid"
    )
    conflict_dir = compute_contrastive_direction(ca_pos, ca_neg)

    cw_pos, cw_neg = batch_extract_contrastive(
        model, tokenizer, clinical,
        "therapeutic_completion", "cold_completion",
        layers=layers, desc="Clinical warmth"
    )
    clinical_warmth_dir = compute_contrastive_direction(cw_pos, cw_neg)

    all_layers = sorted(clinical_dir.keys())

    # =========================================================
    # ANALYSIS 1: Logit lens on clinical sycophancy direction
    # =========================================================
    print("\n--- Analysis 1: Logit Lens on Clinical Sycophancy Direction ---")
    direction_tokens_by_layer = {}
    for layer in all_layers:
        result = direction_to_tokens(model, clinical_dir[layer])
        top_words = [tokenizer.decode([tid]).strip() for tid, _ in result["top"][:10]]
        bot_words = [tokenizer.decode([tid]).strip() for tid, _ in result["bottom"][:10]]
        direction_tokens_by_layer[layer] = {
            "sycophantic_tokens": top_words,
            "therapeutic_tokens": bot_words,
        }
        print(f"  Layer {layer:2d}  syc→ {top_words[:5]}  |  ther→ {bot_words[:5]}")

    # =========================================================
    # ANALYSIS 2: Compute residual and logit-lens it
    # =========================================================
    print("\n--- Analysis 2: Logit Lens on the 63% Residual ---")
    components = {
        "empathy": empathy_dir,
        "factual_sycophancy": factual_dir,
        "conflict_avoidance": conflict_dir,
        "clinical_warmth": clinical_warmth_dir,
        "framing_acceptance": framing_dir,
    }

    residual_dir = {}
    for layer in all_layers:
        residual = clinical_dir[layer].clone()
        for name, comp_dirs in components.items():
            comp_normed = F.normalize(comp_dirs[layer], dim=0)
            proj = (residual @ comp_normed) * comp_normed
            residual = residual - proj
        residual_dir[layer] = F.normalize(residual, dim=0)

    residual_tokens_by_layer = {}
    for layer in all_layers:
        result = direction_to_tokens(model, residual_dir[layer])
        top_words = [tokenizer.decode([tid]).strip() for tid, _ in result["top"][:10]]
        bot_words = [tokenizer.decode([tid]).strip() for tid, _ in result["bottom"][:10]]
        residual_tokens_by_layer[layer] = {
            "positive_tokens": top_words,
            "negative_tokens": bot_words,
        }
        print(f"  Layer {layer:2d}  +residual→ {top_words[:5]}  |  -residual→ {bot_words[:5]}")

    # =========================================================
    # ANALYSIS 3: Layer-wise variance growth
    # =========================================================
    print("\n--- Analysis 3: Layer-wise Variance Growth ---")
    # Compute per-stimulus sycophancy signal at each layer
    layer_norms = {}
    layer_cos_with_final = {}
    final_layer = all_layers[-1]

    for layer in all_layers:
        # Mean difference vector (not normalized) to get actual magnitude
        pos_stack = torch.stack([a[layer] for a in clin_pos])
        neg_stack = torch.stack([a[layer] for a in clin_neg])
        diff = pos_stack.mean(0) - neg_stack.mean(0)
        layer_norms[layer] = diff.norm().item()

        # Cosine with final layer direction
        cos = F.cosine_similarity(
            clinical_dir[layer].unsqueeze(0),
            clinical_dir[final_layer].unsqueeze(0)
        ).item()
        layer_cos_with_final[layer] = cos

    print(f"  {'Layer':>6}  {'Norm':>8}  {'Δ Norm':>8}  {'Cos w/ final':>12}")
    prev_norm = 0
    for layer in all_layers:
        delta = layer_norms[layer] - prev_norm
        print(f"  {layer:>6}  {layer_norms[layer]:>8.3f}  {delta:>+8.3f}  {layer_cos_with_final[layer]:>12.3f}")
        prev_norm = layer_norms[layer]

    # =========================================================
    # ANALYSIS 4: Per-stimulus residual clustering
    # =========================================================
    print("\n--- Analysis 4: Per-stimulus Residual Analysis ---")
    # Use a representative mid-late layer
    analysis_layer = all_layers[2 * len(all_layers) // 3]
    print(f"  Analysis layer: {analysis_layer}")

    per_stimulus_residuals = []
    per_stimulus_labels = []
    for i in range(len(clin_pos)):
        # Per-stimulus direction
        stim_dir = clin_pos[i][analysis_layer] - clin_neg[i][analysis_layer]
        stim_dir_norm = F.normalize(stim_dir, dim=0)

        # Remove known components
        residual = stim_dir_norm.clone()
        for name, comp_dirs in components.items():
            comp_normed = F.normalize(comp_dirs[analysis_layer], dim=0)
            proj = (residual @ comp_normed) * comp_normed
            residual = residual - proj

        per_stimulus_residuals.append(residual)
        per_stimulus_labels.append(clinical[i].get("subcategory", "unknown"))

    # Pairwise cosine similarity of residuals
    residual_stack = torch.stack(per_stimulus_residuals)
    residual_cos_matrix = F.cosine_similarity(
        residual_stack.unsqueeze(1), residual_stack.unsqueeze(0), dim=2
    ).numpy()

    mean_pairwise_cos = np.mean(residual_cos_matrix[np.triu_indices(len(residual_cos_matrix), k=1)])
    print(f"  Mean pairwise cosine of residuals: {mean_pairwise_cos:.3f}")
    print(f"  (High = residuals point in similar direction = one mechanism)")
    print(f"  (Low = residuals are diverse = multiple mechanisms)")

    # Mean residual cosine by subcategory
    subcats = sorted(set(per_stimulus_labels))
    subcat_mean_cos = {}
    for sc in subcats:
        indices = [i for i, l in enumerate(per_stimulus_labels) if l == sc]
        if len(indices) >= 2:
            sub_cos = [residual_cos_matrix[i][j] for i in indices for j in indices if i < j]
            subcat_mean_cos[sc] = np.mean(sub_cos) if sub_cos else 0.0
            print(f"    {sc:25s}: within-group cos = {subcat_mean_cos[sc]:.3f} (n={len(indices)})")

    # =========================================================
    # ANALYSIS 5: Divergence mapping
    # =========================================================
    print("\n--- Analysis 5: Clinical-Factual Divergence Map ---")
    cos_by_layer = cosine_similarity_by_layer(clinical_dir, factual_dir)
    prev_cos = 1.0
    max_drop_layer = 0
    max_drop = 0
    for layer in all_layers:
        delta = cos_by_layer[layer] - prev_cos
        if delta < max_drop:
            max_drop = delta
            max_drop_layer = layer
        print(f"  Layer {layer:2d}: cos = {cos_by_layer[layer]:.3f}  Δ = {delta:+.3f}")
        prev_cos = cos_by_layer[layer]

    print(f"\n  Fastest divergence at layer {max_drop_layer} (Δcos = {max_drop:+.3f})")

    # =========================================================
    # SAVE RESULTS
    # =========================================================
    results = {
        "direction_tokens_by_layer": {
            str(l): v for l, v in direction_tokens_by_layer.items()
        },
        "residual_tokens_by_layer": {
            str(l): v for l, v in residual_tokens_by_layer.items()
        },
        "layer_norms": {str(l): v for l, v in layer_norms.items()},
        "layer_cos_with_final": {str(l): v for l, v in layer_cos_with_final.items()},
        "per_stimulus_residual_mean_cos": float(mean_pairwise_cos),
        "subcat_residual_coherence": {k: float(v) for k, v in subcat_mean_cos.items()},
        "clinical_factual_divergence": {str(l): v for l, v in cos_by_layer.items()},
        "fastest_divergence_layer": max_drop_layer,
        "analysis_layer": analysis_layer,
    }

    with open(os.path.join(output_dir, "h15_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # =========================================================
    # PLOTS
    # =========================================================
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(20, 10))

    # Top-left: Layer norms
    ax = axes[0, 0]
    ax.plot(all_layers, [layer_norms[l] for l in all_layers], 'b-o', markersize=4)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Direction Norm")
    ax.set_title("Signal Magnitude by Layer")
    ax.grid(True, alpha=0.3)

    # Top-middle: Cosine with final layer
    ax = axes[0, 1]
    ax.plot(all_layers, [layer_cos_with_final[l] for l in all_layers], 'g-o', markersize=4)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Cosine with Final Layer")
    ax.set_title("Direction Stability")
    ax.grid(True, alpha=0.3)

    # Top-right: Clinical-factual divergence
    ax = axes[0, 2]
    ax.plot(all_layers, [cos_by_layer[l] for l in all_layers], 'r-o', markersize=4)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Cosine (Clinical vs Factual)")
    ax.set_title("Divergence from Factual")
    ax.grid(True, alpha=0.3)

    # Bottom-left: Per-stimulus residual cosine matrix
    ax = axes[1, 0]
    im = ax.imshow(residual_cos_matrix, cmap='RdBu_r', vmin=-0.5, vmax=0.5)
    ax.set_title(f"Residual Pairwise Cos (L{analysis_layer})")
    ax.set_xlabel("Stimulus")
    ax.set_ylabel("Stimulus")
    plt.colorbar(im, ax=ax, shrink=0.8)

    # Bottom-middle: Sycophancy direction tokens at key layers
    ax = axes[1, 1]
    key_layers = [0, len(all_layers)//4, len(all_layers)//2, 3*len(all_layers)//4, len(all_layers)-1]
    key_layers = [all_layers[i] for i in key_layers]
    text_lines = []
    for l in key_layers:
        syc_tokens = direction_tokens_by_layer[l]["sycophantic_tokens"][:3]
        ther_tokens = direction_tokens_by_layer[l]["therapeutic_tokens"][:3]
        text_lines.append(f"L{l}: syc→{syc_tokens}  ther→{ther_tokens}")
    ax.text(0.05, 0.95, "\n\n".join(text_lines), transform=ax.transAxes,
            verticalalignment='top', fontsize=8, fontfamily='monospace')
    ax.set_title("Direction → Token Space")
    ax.axis('off')

    # Bottom-right: Residual direction tokens at key layers
    ax = axes[1, 2]
    text_lines = []
    for l in key_layers:
        pos_tokens = residual_tokens_by_layer[l]["positive_tokens"][:3]
        neg_tokens = residual_tokens_by_layer[l]["negative_tokens"][:3]
        text_lines.append(f"L{l}: +res→{pos_tokens}  -res→{neg_tokens}")
    ax.text(0.05, 0.95, "\n\n".join(text_lines), transform=ax.transAxes,
            verticalalignment='top', fontsize=8, fontfamily='monospace')
    ax.set_title("Residual → Token Space")
    ax.axis('off')

    fig.suptitle("H15: Full Clinical Sycophancy Circuit Decomposition", fontsize=14)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "h15_circuit_decomposition.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\nH15 results saved to {output_dir}")
    return results
