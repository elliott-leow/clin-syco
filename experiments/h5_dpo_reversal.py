"""Hypothesis 5: DPO Reverses the Clinical Amplification Effect.

The base model amplifies clinical correctness in late layers (H2).
If DPO reverses this, the clinical amplification curve should flatten
or invert across training stages.

Analysis:
  Run the H2 logit lens analysis at each checkpoint (base → SFT → DPO)
  and compare how the correct-answer signal changes.

Note: (1) With OLMo-2 (base/instruct only), this experiment tests whether
alignment training in general reverses clinical amplification, not DPO
specifically. The DPO-specific claim requires OLMo-3 base/SFT/DPO checkpoints
(available on Colab). (2) The logit lens signal measures early-layer token
preference under the unembedding matrix, which is a necessary but not
sufficient condition for knowledge representation — the signal could reflect
token frequency biases or positional encoding artifacts rather than genuine
factual knowledge.
"""

import json
import os
import numpy as np
import torch

from pals.models import load_model, load_tokenizer, OLMO2_CHECKPOINTS, OLMO3_CHECKPOINTS
from pals.logit_lens import batch_correct_signal, aggregate_signals
from pals.viz import plot_logit_lens_signals


def run(model, tokenizer, stimuli_dir, output_dir,
        checkpoints=None, device="cpu", quantize_4bit=False):
    """Run H5: logit lens across checkpoints."""
    os.makedirs(output_dir, exist_ok=True)

    if checkpoints is None:
        checkpoints = OLMO2_CHECKPOINTS

    with open(os.path.join(stimuli_dir, "factual_control.json")) as f:
        factual = json.load(f)
    with open(os.path.join(stimuli_dir, "clinical_bridge.json")) as f:
        bridge = json.load(f)
    with open(os.path.join(stimuli_dir, "clinical_correct_answer.json")) as f:
        clinical_clear = json.load(f)

    print("\n=== H5: DPO Reversal of Clinical Amplification ===")

    all_results = {}

    for ckpt_name, model_id in checkpoints.items():
        print(f"\n--- Checkpoint: {ckpt_name} ({model_id}) ---")
        ckpt_model = load_model(model_id, device=device, quantize_4bit=quantize_4bit)
        ckpt_tok = load_tokenizer(model_id)

        fact_sig = batch_correct_signal(ckpt_model, ckpt_tok, factual, f"Factual ({ckpt_name})")
        bridge_sig = batch_correct_signal(ckpt_model, ckpt_tok, bridge, f"Bridge ({ckpt_name})")
        clear_sig = batch_correct_signal(ckpt_model, ckpt_tok, clinical_clear, f"Clear ({ckpt_name})")

        fact_agg = aggregate_signals(fact_sig)
        bridge_agg = aggregate_signals(bridge_sig)
        clear_agg = aggregate_signals(clear_sig)

        layers = fact_agg["layers"]
        early = layers[:len(layers) // 3]
        late = layers[2 * len(layers) // 3:]

        for label, agg in [("factual", fact_agg), ("bridge", bridge_agg), ("clear", clear_agg)]:
            e = np.mean([agg["mean"][l] for l in early])
            la = np.mean([agg["mean"][l] for l in late])
            print(f"  {label:15s}  early={e:+.3f}  late={la:+.3f}  delta={la - e:+.3f}")

        all_results[ckpt_name] = {
            "factual": {"mean": {str(k): float(v) for k, v in fact_agg["mean"].items()},
                        "std": {str(k): float(v) for k, v in fact_agg["std"].items()},
                        "layers": [str(l) for l in layers]},
            "bridge": {"mean": {str(k): float(v) for k, v in bridge_agg["mean"].items()},
                       "std": {str(k): float(v) for k, v in bridge_agg["std"].items()},
                       "layers": [str(l) for l in layers]},
            "clear": {"mean": {str(k): float(v) for k, v in clear_agg["mean"].items()},
                      "std": {str(k): float(v) for k, v in clear_agg["std"].items()},
                      "layers": [str(l) for l in layers]},
        }

        del ckpt_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        import gc; gc.collect()

    # --- Compare clinical clear amplification across checkpoints ---
    print(f"\n--- Clinical Clear-Practice: Amplification Delta by Checkpoint ---")
    for ckpt_name in checkpoints:
        data = all_results[ckpt_name]["clear"]
        layers = [int(l) for l in data["layers"]]
        early = layers[:len(layers) // 3]
        late = layers[2 * len(layers) // 3:]
        e_mean = np.mean([data["mean"][str(l)] for l in early])
        l_mean = np.mean([data["mean"][str(l)] for l in late])
        print(f"  {ckpt_name:12s}: early={e_mean:+.3f}, late={l_mean:+.3f}, "
              f"amplification={l_mean - e_mean:+.3f}")

    # Save
    with open(os.path.join(output_dir, "h5_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    # Plot: one subplot per checkpoint showing all 3 conditions
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_ckpts = len(checkpoints)
    fig, axes = plt.subplots(1, n_ckpts, figsize=(6 * n_ckpts, 5), sharey=True)
    if n_ckpts == 1:
        axes = [axes]

    for ax, ckpt_name in zip(axes, checkpoints):
        data = all_results[ckpt_name]
        for cond, label, color in [("factual", "Factual", "tab:blue"),
                                    ("bridge", "Clinical bridge", "tab:orange"),
                                    ("clear", "Clinical clear", "tab:green")]:
            layers = [int(l) for l in data[cond]["layers"]]
            means = [data[cond]["mean"][str(l)] for l in layers]
            stds = [data[cond]["std"][str(l)] for l in layers]
            ax.plot(layers, means, marker="o", markersize=3, label=label, color=color)
            ax.fill_between(layers,
                            np.array(means) - np.array(stds),
                            np.array(means) + np.array(stds),
                            alpha=0.15, color=color)
        ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel("Layer")
        ax.set_title(f"Checkpoint: {ckpt_name}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Signal (log P(therapeutic) - log P(sycophantic))")
    fig.suptitle("H5: Correct-Answer Signal Across Training Stages", fontsize=13)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "h5_dpo_reversal.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\nH5 results saved to {output_dir}")
    return all_results
