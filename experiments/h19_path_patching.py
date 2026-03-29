"""Hypothesis 19: Path Patching — Which Components Mediate Sycophancy?

Decomposes the causal effect from H18 into per-head and per-MLP
contributions. Identifies the minimal set of components that form
the clinical sycophancy circuit.
"""

import json
import os
import gc
import numpy as np
import torch

from pals.patching import (causal_trace, get_clean_hidden_states,
                            get_sublayer_caches, compute_logit_diff)
from pals.circuit import compute_component_effects, rank_components, build_circuit
from pals.models import get_device


def run(model, tokenizer, stimuli_dir, output_dir, layers=None, n_stimuli=10,
        critical_layers=None):
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(stimuli_dir, "cognitive_distortions.json")) as f:
        clinical = json.load(f)[:n_stimuli]

    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    device = get_device(model)

    # if critical layers not provided, use top third
    if critical_layers is None:
        if layers is not None:
            critical_layers = layers[len(layers)//2:]
        else:
            critical_layers = list(range(2 * n_layers // 3, n_layers))

    print(f"\n=== H19: Path Patching ===")
    print(f"Critical layers: {critical_layers}")
    print(f"Heads per layer: {n_heads}")
    print(f"Components to test: {len(critical_layers) * (n_heads + 1)}")
    print(f"Stimuli: {n_stimuli}")

    all_effects = []
    for si, s in enumerate(clinical):
        # construct therapeutic and sycophantic input ids
        ther_text = s["user_prompt"] + " " + s["therapeutic_completion"][:50]
        syc_text = s["user_prompt"] + " " + s["sycophantic_completion"][:50]
        ther_ids = tokenizer.encode(ther_text, return_tensors="pt").to(device)
        syc_ids = tokenizer.encode(syc_text, return_tensors="pt").to(device)

        ther_tok = tokenizer.encode(s["therapeutic_completion"], add_special_tokens=False)[0]
        syc_tok = tokenizer.encode(s["sycophantic_completion"], add_special_tokens=False)[0]

        # get sublayer caches from therapeutic run
        from pals.circuit import _get_clean_caches
        attn_cache, mlp_cache = _get_clean_caches(model, ther_ids, critical_layers)

        # baseline diffs
        with torch.no_grad():
            clean_logits = model(ther_ids).logits[0, -1, :]
            corrupt_logits = model(syc_ids).logits[0, -1, :]
        clean_diff = compute_logit_diff(clean_logits, ther_tok, syc_tok)
        baseline_diff = compute_logit_diff(corrupt_logits, ther_tok, syc_tok)

        effects = compute_component_effects(
            model, tokenizer, s, critical_layers, (attn_cache, mlp_cache),
            syc_ids, ther_tok, syc_tok, baseline_diff, clean_diff
        )
        all_effects.append(effects)
        print(f"  [{si+1}/{n_stimuli}] clean_diff={clean_diff:.3f}, baseline_diff={baseline_diff:.3f}")

        del attn_cache, mlp_cache
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # rank and build circuit
    ranked = rank_components(all_effects)
    circuit = build_circuit(ranked, threshold=0.01)

    print(f"\nTop 15 components by causal effect:")
    for key, mean_eff, std_eff in ranked[:15]:
        print(f"  {str(key):>30}: {mean_eff:+.4f} +/- {std_eff:.4f}")

    print(f"\nCircuit size: {len(circuit)} components (threshold=0.01)")

    results = {
        "ranked_components": [(str(k), float(m), float(s)) for k, m, s in ranked[:50]],
        "circuit_components": [str(c) for c in circuit],
        "circuit_size": len(circuit),
        "critical_layers": critical_layers,
        "n_heads": n_heads,
    }

    with open(os.path.join(output_dir, "h19_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # heatmap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matrix = np.zeros((len(critical_layers), n_heads + 1))
    ranked_dict = {k: m for k, m, s in ranked}
    for li, layer in enumerate(critical_layers):
        for h in range(n_heads):
            matrix[li, h] = ranked_dict.get((layer, 'head', h), 0.0)
        matrix[li, n_heads] = ranked_dict.get((layer, 'mlp', 0), 0.0)

    fig, ax = plt.subplots(figsize=(12, max(3, len(critical_layers) * 0.4)))
    im = ax.imshow(matrix, aspect='auto', cmap='RdBu_r',
                   vmin=-max(0.05, abs(matrix).max()), vmax=max(0.05, abs(matrix).max()))
    ax.set_yticks(range(len(critical_layers)))
    ax.set_yticklabels(critical_layers)
    ax.set_xlabel('Component (heads + MLP)')
    ax.set_ylabel('Layer')
    ax.set_title('H19: Per-Component Causal Effect (Path Patching)')
    plt.colorbar(im, ax=ax, label='Recovery fraction')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "h19_path_patching.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    print(f"\nH19 results saved to {output_dir}")
    return results
