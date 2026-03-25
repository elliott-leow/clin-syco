"""Hypothesis 2: Is clinical sycophancy driven by uncertainty or social deference?

Two-stage override framework:
  - DEFERENCE: Early layers encode the correct (non-sycophantic) response,
    but later layers override it with the sycophantic response.
  - UNCERTAINTY: No strong correct-answer signal at any layer; the model
    genuinely doesn't know the right answer.

Analysis:
  1. Logit lens at each layer to check what the model internally predicts.
  2. Compare layer-wise "correct answer signal" across:
     - Factual errors (strong ground truth -> strong early signal expected)
     - Clinical clear-practice (known best practice -> if deference, should also be strong)
     - Clinical bridge (emotionally invested false beliefs)
  3. Project prompt activations onto the factual sycophancy direction.

Note: A high logit lens signal at early layers measures early-layer token
preference under the unembedding matrix, which is a necessary but not
sufficient condition for "the model knows the answer." The signal could be
influenced by token frequency or positional encoding artifacts rather than
genuine factual knowledge. A strong early signal is consistent with knowledge
representation, but absence of signal is more informative (rules out the
deference hypothesis) than presence of signal (which could have non-knowledge
explanations).
"""

import json
import os
import numpy as np

from pals.extraction import batch_extract_contrastive, batch_extract_prompt
from pals.directions import compute_contrastive_direction, project_acts_by_layer
from pals.logit_lens import batch_correct_signal, aggregate_signals
from pals.stats import permutation_test_means, bootstrap_ci
from pals.viz import plot_logit_lens_signals


def run(model, tokenizer, stimuli_dir, output_dir, layers=None):
    """Run H2 experiment.

    Args:
        model: Loaded HuggingFace model.
        tokenizer: Loaded tokenizer.
        stimuli_dir: Path to stimulus JSON files.
        output_dir: Path to save results.
        layers: Specific layers or None for all.
    """
    os.makedirs(output_dir, exist_ok=True)

    # --- Load stimuli ---
    with open(os.path.join(stimuli_dir, "factual_control.json")) as f:
        factual = json.load(f)
    with open(os.path.join(stimuli_dir, "clinical_bridge.json")) as f:
        bridge = json.load(f)
    with open(os.path.join(stimuli_dir, "clinical_correct_answer.json")) as f:
        clinical_clear = json.load(f)

    print("\n=== H2: Uncertainty vs Deference ===")

    # --- Part 1: Logit Lens Analysis ---
    # Compute correct-answer signal at each layer:
    # signal = log P(first therapeutic token) - log P(first sycophantic token)
    # Positive = model internally favors the correct/therapeutic response at that layer.

    print("Running logit lens on factual stimuli...")
    fact_signals = batch_correct_signal(model, tokenizer, factual, desc="Factual logit lens")
    fact_agg = aggregate_signals(fact_signals)

    print("Running logit lens on clinical bridge stimuli...")
    bridge_signals = batch_correct_signal(model, tokenizer, bridge, desc="Bridge logit lens")
    bridge_agg = aggregate_signals(bridge_signals)

    print("Running logit lens on clinical clear-practice stimuli...")
    clear_signals = batch_correct_signal(model, tokenizer, clinical_clear, desc="Clinical clear logit lens")
    clear_agg = aggregate_signals(clear_signals)

    # --- Print summary ---
    all_layers = fact_agg["layers"]
    early_layers = all_layers[:len(all_layers) // 3]
    late_layers = all_layers[2 * len(all_layers) // 3:]

    def early_late_summary(agg, label):
        early_mean = np.mean([agg["mean"][l] for l in early_layers])
        late_mean = np.mean([agg["mean"][l] for l in late_layers])
        print(f"  {label:25s}  early={early_mean:+.3f}  late={late_mean:+.3f}  "
              f"delta={late_mean - early_mean:+.3f}")

    print(f"\nCorrect-answer signal (positive = favors therapeutic):")
    early_late_summary(fact_agg, "Factual")
    early_late_summary(bridge_agg, "Clinical bridge")
    early_late_summary(clear_agg, "Clinical clear-practice")

    # --- Part 2: Direction Projection Analysis ---
    # Use the factual sycophancy direction as reference, project clinical
    # prompt activations onto it to see if the model encodes anti-sycophancy
    # signal at early layers.

    print("\nComputing factual sycophancy direction for projection analysis...")
    fact_pos, fact_neg = batch_extract_contrastive(
        model, tokenizer, factual,
        "sycophantic_completion", "therapeutic_completion",
        layers=layers, desc="Factual (for direction)"
    )
    fact_direction = compute_contrastive_direction(fact_pos, fact_neg)

    print("Extracting prompt-only activations...")
    fact_prompt_acts = batch_extract_prompt(model, tokenizer, factual, layers, "Factual prompts")
    bridge_prompt_acts = batch_extract_prompt(model, tokenizer, bridge, layers, "Bridge prompts")
    clear_prompt_acts = batch_extract_prompt(model, tokenizer, clinical_clear, layers, "Clinical prompts")

    # Project onto factual sycophancy direction
    fact_proj = project_acts_by_layer(fact_prompt_acts, fact_direction)
    bridge_proj = project_acts_by_layer(bridge_prompt_acts, fact_direction)
    clear_proj = project_acts_by_layer(clear_prompt_acts, fact_direction)

    # --- Statistical tests ---
    print("\nStatistical tests (early layers: factual vs clinical signal)...")
    test_results = {}
    for l in early_layers[:3]:
        fact_vals = [s[l] for s in fact_signals]
        bridge_vals = [s[l] for s in bridge_signals]
        clear_vals = [s[l] for s in clear_signals]

        test_fb = permutation_test_means(fact_vals, bridge_vals, n_perms=5000)
        test_fc = permutation_test_means(fact_vals, clear_vals, n_perms=5000)

        test_results[l] = {
            "fact_vs_bridge": {"diff": float(test_fb["observed"]), "p": float(test_fb["p_value"])},
            "fact_vs_clear": {"diff": float(test_fc["observed"]), "p": float(test_fc["p_value"])},
        }
        print(f"  Layer {l}: fact-bridge diff={test_fb['observed']:+.3f} p={test_fb['p_value']:.4f}, "
              f"fact-clear diff={test_fc['observed']:+.3f} p={test_fc['p_value']:.4f}")

    # --- Interpretation ---
    early_fact = np.mean([fact_agg["mean"][l] for l in early_layers])
    early_bridge = np.mean([bridge_agg["mean"][l] for l in early_layers])
    early_clear = np.mean([clear_agg["mean"][l] for l in early_layers])

    print(f"\n--- Interpretation ---")
    if early_clear > 0 and abs(early_clear) > abs(early_fact) * 0.3:
        print("Clinical clear-practice shows early-layer correct signal -> DEFERENCE pattern")
        print("The model knows the right answer but overrides it.")
    elif abs(early_clear) < abs(early_fact) * 0.3:
        print("Clinical clear-practice shows weak early signal -> UNCERTAINTY pattern")
        print("The model lacks a strong correct-answer representation.")
    else:
        print("Mixed signal — cannot cleanly distinguish deference from uncertainty.")

    # --- Save results ---
    results = {
        "logit_lens": {
            "factual": {"mean": {str(k): float(v) for k, v in fact_agg["mean"].items()},
                        "std": {str(k): float(v) for k, v in fact_agg["std"].items()}},
            "clinical_bridge": {"mean": {str(k): float(v) for k, v in bridge_agg["mean"].items()},
                                "std": {str(k): float(v) for k, v in bridge_agg["std"].items()}},
            "clinical_clear": {"mean": {str(k): float(v) for k, v in clear_agg["mean"].items()},
                               "std": {str(k): float(v) for k, v in clear_agg["std"].items()}},
        },
        "direction_projection": {
            "factual": {str(l): {"mean": float(np.mean(v)), "std": float(np.std(v))}
                        for l, v in fact_proj.items()},
            "clinical_bridge": {str(l): {"mean": float(np.mean(v)), "std": float(np.std(v))}
                                for l, v in bridge_proj.items()},
            "clinical_clear": {str(l): {"mean": float(np.mean(v)), "std": float(np.std(v))}
                               for l, v in clear_proj.items()},
        },
        "statistical_tests": {str(k): v for k, v in test_results.items()},
        "interpretation": {
            "early_factual_signal": float(early_fact),
            "early_bridge_signal": float(early_bridge),
            "early_clear_signal": float(early_clear),
        },
    }

    with open(os.path.join(output_dir, "h2_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # --- Plots ---
    plot_logit_lens_signals(
        {
            "Factual (ground truth)": fact_agg,
            "Clinical bridge (false belief)": bridge_agg,
            "Clinical clear-practice": clear_agg,
        },
        title="H2: Correct-Answer Signal by Layer (Logit Lens)",
        save_path=os.path.join(output_dir, "h2_logit_lens.png"),
    )

    # Direction projection plot
    proj_data = {}
    for label, proj in [("Factual", fact_proj), ("Clinical bridge", bridge_proj),
                        ("Clinical clear", clear_proj)]:
        proj_layers = sorted(proj.keys())
        proj_data[label] = {
            "layers": proj_layers,
            "mean": {l: np.mean(proj[l]) for l in proj_layers},
            "std": {l: np.std(proj[l]) for l in proj_layers},
        }

    plot_logit_lens_signals(
        proj_data,
        title="H2: Prompt Projection onto Factual Sycophancy Direction",
        save_path=os.path.join(output_dir, "h2_direction_projection.png"),
    )

    print(f"\nH2 results saved to {output_dir}")
    return results
