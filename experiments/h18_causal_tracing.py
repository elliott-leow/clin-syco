"""Hypothesis 18: Causal Tracing — Where Is the Sycophancy Decision Made?

Activation patching from "therapeutic run" into "sycophantic run" at each
layer. Measures how much patching restores the therapeutic logit advantage,
identifying which layers causally drive the sycophancy decision.

Complements H2 (logit lens, correlational) and H15 (direction norms,
correlational) with a causal metric.
"""

import json
import os
import gc
import numpy as np
import torch

from pals.patching import causal_trace, compute_logit_diff
from pals.models import get_device


def run(model, tokenizer, stimuli_dir, output_dir, layers=None, n_stimuli=15,
        n_completion_tokens=3):
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(stimuli_dir, "cognitive_distortions.json")) as f:
        clinical = json.load(f)[:n_stimuli]
    with open(os.path.join(stimuli_dir, "factual_control.json")) as f:
        factual = json.load(f)[:n_stimuli]

    n_layers = model.config.num_hidden_layers
    if layers is None:
        layers = list(range(n_layers))

    print(f"\n=== H18: Causal Tracing ===")
    print(f"Layers: {len(layers)}, Stimuli: {len(clinical)} clinical + {len(factual)} factual")
    print(f"Completion tokens: {n_completion_tokens}")

    # causal trace for clinical stimuli
    print("\nTracing clinical sycophancy circuit...")
    clinical_traces = []
    for i, s in enumerate(clinical):
        trace = causal_trace(model, tokenizer, s, layers=layers,
                             n_completion_tokens=n_completion_tokens)
        clinical_traces.append(trace)
        if (i + 1) % 5 == 0:
            print(f"  [{i+1}/{len(clinical)}]")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # causal trace for factual stimuli
    print("\nTracing factual sycophancy circuit...")
    factual_traces = []
    for i, s in enumerate(factual):
        trace = causal_trace(model, tokenizer, s, layers=layers,
                             n_completion_tokens=n_completion_tokens)
        factual_traces.append(trace)
        if (i + 1) % 5 == 0:
            print(f"  [{i+1}/{len(factual)}]")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # aggregate
    mean_clinical = {}
    mean_factual = {}
    for l in layers:
        cvals = [t.get(l, 0.0) for t in clinical_traces]
        fvals = [t.get(l, 0.0) for t in factual_traces]
        mean_clinical[l] = float(np.mean(cvals))
        mean_factual[l] = float(np.mean(fvals))

    # find peak layers
    clin_peak = max(mean_clinical.items(), key=lambda x: x[1])
    fact_peak = max(mean_factual.items(), key=lambda x: x[1])

    print(f"\nClinical peak: layer {clin_peak[0]} (recovery = {clin_peak[1]:.3f})")
    print(f"Factual peak:  layer {fact_peak[0]} (recovery = {fact_peak[1]:.3f})")

    # identify critical layers (top 25% by recovery)
    sorted_clin = sorted(mean_clinical.items(), key=lambda x: -x[1])
    threshold = sorted_clin[len(sorted_clin) // 4][1] if len(sorted_clin) > 4 else 0
    critical_layers = [l for l, v in mean_clinical.items() if v >= threshold]
    print(f"Critical layers (top 25%): {critical_layers}")

    results = {
        "clinical_recovery_by_layer": {str(l): v for l, v in mean_clinical.items()},
        "factual_recovery_by_layer": {str(l): v for l, v in mean_factual.items()},
        "clinical_peak_layer": clin_peak[0],
        "clinical_peak_recovery": clin_peak[1],
        "factual_peak_layer": fact_peak[0],
        "factual_peak_recovery": fact_peak[1],
        "critical_layers": critical_layers,
        "n_completion_tokens": n_completion_tokens,
    }

    with open(os.path.join(output_dir, "h18_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(layers, [mean_clinical[l] for l in layers], 'o-', color='#c0392b',
            label='Clinical', markersize=4, lw=1.5)
    ax.plot(layers, [mean_factual[l] for l in layers], 's-', color='#2980b9',
            label='Factual', markersize=4, lw=1.5)
    ax.axhline(0, color='gray', ls=':', alpha=0.4)
    ax.set(xlabel='Layer', ylabel='Recovery fraction',
           title='H18: Causal Tracing — Where Is the Sycophancy Decision?')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "h18_causal_tracing.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    print(f"\nH18 results saved to {output_dir}")
    return results
