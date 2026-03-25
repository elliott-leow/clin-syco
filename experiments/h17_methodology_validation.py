"""Hypothesis 17: Methodology Validation — Reviewer-Requested Controls.

Addresses major methodological concerns from peer review:

1. Order-invariant decomposition (M4) — Run H9/H10's decomposition using OLS
   regression instead of Gram-Schmidt. If conclusions change, the ordering matters.

2. Low-alpha steering specificity (M5) — Test whether the steering vector
   outperforms random vectors at alpha=1-2 (not just alpha=8 where any large
   perturbation works).

3. PCA of contrastive difference vectors (M3) — How much variance does the
   first PC of the per-stimulus difference vectors capture? If <50%, the
   single-direction approximation is losing signal.

4. Reversed Gram-Schmidt ordering — Run the 2-component decomposition with
   factual first (instead of empathy first) and report whether conclusions flip.
"""

import json
import os
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from itertools import permutations

from pals.extraction import batch_extract_contrastive
from pals.directions import compute_contrastive_direction, cosine_similarity_by_layer
from pals.decomposition import decompose_by_layer, decompose_ols_by_layer
from pals.models import get_device
from pals.stats import bootstrap_ci


def run(model, tokenizer, stimuli_dir, output_dir, layers=None, n_random=20):
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(stimuli_dir, "cognitive_distortions.json")) as f:
        clinical = json.load(f)
    with open(os.path.join(stimuli_dir, "factual_control.json")) as f:
        factual = json.load(f)
    with open(os.path.join(stimuli_dir, "high_emotion_general.json")) as f:
        emotion = json.load(f)

    device = get_device(model)
    n_layers = model.config.num_hidden_layers

    # Extract all directions
    print("\n=== H17: Methodology Validation ===")
    print("Extracting directions...")

    clin_pos, clin_neg = batch_extract_contrastive(
        model, tokenizer, clinical,
        "sycophantic_completion", "therapeutic_completion",
        layers=layers, desc="Clinical"
    )
    clinical_dir = compute_contrastive_direction(clin_pos, clin_neg)

    fact_pos, fact_neg = batch_extract_contrastive(
        model, tokenizer, factual,
        "sycophantic_completion", "therapeutic_completion",
        layers=layers, desc="Factual"
    )
    factual_dir = compute_contrastive_direction(fact_pos, fact_neg)

    emp_pos, emp_neg = batch_extract_contrastive(
        model, tokenizer, emotion,
        "therapeutic_completion", "cold_completion",
        layers=layers, desc="Empathy"
    )
    empathy_dir = compute_contrastive_direction(emp_pos, emp_neg)

    all_layers = sorted(clinical_dir.keys())

    # =========================================================
    # VALIDATION 1: OLS vs Gram-Schmidt (2-component)
    # =========================================================
    print("\n--- V1: OLS vs Gram-Schmidt Decomposition (2-component) ---")

    gs_result = decompose_by_layer(
        clinical_dir,
        {"empathy": empathy_dir, "factual_sycophancy": factual_dir}
    )
    ols_result = decompose_ols_by_layer(
        clinical_dir,
        {"empathy": empathy_dir, "factual_sycophancy": factual_dir}
    )

    # Compare
    gs_emp = np.mean([gs_result[l]["unique_variance_explained"]["empathy"] for l in all_layers])
    gs_fact = np.mean([gs_result[l]["unique_variance_explained"]["factual_sycophancy"] for l in all_layers])
    gs_res = np.mean([gs_result[l]["residual_variance_fraction"] for l in all_layers])

    ols_emp = np.mean([ols_result[l]["variance_explained"]["empathy"] for l in all_layers])
    ols_fact = np.mean([ols_result[l]["variance_explained"]["factual_sycophancy"] for l in all_layers])
    ols_res = np.mean([ols_result[l]["residual_variance_fraction"] for l in all_layers])

    print(f"  {'Method':>15}  {'Empathy':>10}  {'Factual':>10}  {'Residual':>10}")
    print(f"  {'Gram-Schmidt':>15}  {gs_emp:>9.1%}  {gs_fact:>9.1%}  {gs_res:>9.1%}")
    print(f"  {'OLS (Type III)':>15}  {ols_emp:>9.1%}  {ols_fact:>9.1%}  {ols_res:>9.1%}")

    conclusion_same = (gs_emp > gs_fact) == (ols_emp > ols_fact)
    print(f"  Conclusion consistent: {conclusion_same}")
    print(f"  (Both say empathy > factual: {gs_emp > gs_fact})")

    # =========================================================
    # VALIDATION 2: Reversed Gram-Schmidt ordering
    # =========================================================
    print("\n--- V2: Reversed Gram-Schmidt Ordering ---")

    gs_reversed = decompose_by_layer(
        clinical_dir,
        {"factual_sycophancy": factual_dir, "empathy": empathy_dir}  # reversed
    )

    gs_rev_emp = np.mean([gs_reversed[l]["unique_variance_explained"]["empathy"] for l in all_layers])
    gs_rev_fact = np.mean([gs_reversed[l]["unique_variance_explained"]["factual_sycophancy"] for l in all_layers])
    gs_rev_res = np.mean([gs_reversed[l]["residual_variance_fraction"] for l in all_layers])

    print(f"  {'Order':>20}  {'Empathy':>10}  {'Factual':>10}  {'Residual':>10}")
    print(f"  {'Empathy first':>20}  {gs_emp:>9.1%}  {gs_fact:>9.1%}  {gs_res:>9.1%}")
    print(f"  {'Factual first':>20}  {gs_rev_emp:>9.1%}  {gs_rev_fact:>9.1%}  {gs_rev_res:>9.1%}")

    order_robust = (gs_emp > gs_fact) == (gs_rev_emp > gs_rev_fact)
    print(f"  Order-robust: {order_robust}")

    # =========================================================
    # VALIDATION 3: PCA of per-stimulus difference vectors
    # =========================================================
    print("\n--- V3: PCA of Per-Stimulus Contrastive Vectors ---")

    # At a representative mid-late layer
    pca_layer = all_layers[2 * len(all_layers) // 3]
    print(f"  Analysis layer: {pca_layer}")

    diff_vectors = []
    for i in range(len(clin_pos)):
        diff = clin_pos[i][pca_layer] - clin_neg[i][pca_layer]
        diff_vectors.append(diff)

    diff_matrix = torch.stack(diff_vectors)  # (n_stimuli, hidden_dim)
    # Center
    diff_matrix = diff_matrix - diff_matrix.mean(0)

    # SVD for PCA
    U, S, Vh = torch.linalg.svd(diff_matrix, full_matrices=False)
    explained_variance = (S ** 2) / (S ** 2).sum()

    cumvar = torch.cumsum(explained_variance, dim=0)
    pc1_var = float(explained_variance[0].item())
    pc5_cumvar = float(cumvar[min(4, len(cumvar)-1)].item())
    pc10_cumvar = float(cumvar[min(9, len(cumvar)-1)].item())

    print(f"  PC1 variance explained: {pc1_var:.1%}")
    print(f"  Top 5 PCs cumulative:   {pc5_cumvar:.1%}")
    print(f"  Top 10 PCs cumulative:  {pc10_cumvar:.1%}")

    if pc1_var > 0.5:
        print(f"  -> Single-direction approximation captures majority of variance")
    elif pc1_var > 0.3:
        print(f"  -> Single-direction captures substantial but not dominant variance")
    else:
        print(f"  -> WARNING: Single-direction approximation may be inadequate")

    # Also check: cosine between mean-diff direction and PC1
    mean_diff = diff_matrix.mean(0)
    mean_diff_normed = F.normalize(mean_diff, dim=0)
    pc1 = Vh[0]
    cos_mean_pc1 = float(F.cosine_similarity(mean_diff_normed.unsqueeze(0), pc1.unsqueeze(0)).item())
    print(f"  Cosine(mean-diff, PC1): {cos_mean_pc1:.3f}")

    # =========================================================
    # VALIDATION 4: Low-alpha steering specificity
    # =========================================================
    print("\n--- V4: Low-Alpha Steering Specificity ---")

    steer_layer = 2 * n_layers // 3
    steer_dir = clinical_dir[steer_layer].to(device)
    steer_norm = steer_dir.norm().item()

    test_alphas = [0.5, 1.0, 2.0]
    specificity_results = {}

    for alpha in test_alphas:
        # Real vector shift
        real_shifts = []
        for s in clinical[:15]:
            input_ids = tokenizer.encode(s["user_prompt"], return_tensors="pt").to(device)
            ther_tok = tokenizer.encode(s["therapeutic_completion"], add_special_tokens=False)[0]
            syc_tok = tokenizer.encode(s["sycophantic_completion"], add_special_tokens=False)[0]

            with torch.no_grad():
                logits_base = model(input_ids).logits[0, -1, :]
            lp_base = F.log_softmax(logits_base.float(), dim=-1)
            baseline = (lp_base[ther_tok] - lp_base[syc_tok]).item()

            def real_hook(module, inp, out, _d=steer_dir, _a=alpha):
                h = out[0] if isinstance(out, tuple) else out
                h = h.clone()
                h[:, -1, :] -= _a * _d
                return (h,) + out[1:] if isinstance(out, tuple) else h

            hook = model.model.layers[steer_layer].register_forward_hook(real_hook)
            with torch.no_grad():
                logits_steer = model(input_ids).logits[0, -1, :]
            hook.remove()
            lp_steer = F.log_softmax(logits_steer.float(), dim=-1)
            steered = (lp_steer[ther_tok] - lp_steer[syc_tok]).item()
            real_shifts.append(steered - baseline)

        real_mean = float(np.mean(real_shifts))

        # Random vectors
        random_means = []
        for r in range(n_random):
            rand_vec = torch.randn_like(steer_dir)
            rand_vec = F.normalize(rand_vec, dim=0) * steer_norm
            rand_vec = rand_vec.to(device)

            rand_shifts = []
            for s in clinical[:15]:
                input_ids = tokenizer.encode(s["user_prompt"], return_tensors="pt").to(device)
                ther_tok = tokenizer.encode(s["therapeutic_completion"], add_special_tokens=False)[0]
                syc_tok = tokenizer.encode(s["sycophantic_completion"], add_special_tokens=False)[0]

                with torch.no_grad():
                    logits_base = model(input_ids).logits[0, -1, :]
                lp_base = F.log_softmax(logits_base.float(), dim=-1)
                baseline = (lp_base[ther_tok] - lp_base[syc_tok]).item()

                def rand_hook(module, inp, out, _v=rand_vec, _a=alpha):
                    h = out[0] if isinstance(out, tuple) else out
                    h = h.clone()
                    h[:, -1, :] -= _a * _v
                    return (h,) + out[1:] if isinstance(out, tuple) else h

                hook = model.model.layers[steer_layer].register_forward_hook(rand_hook)
                with torch.no_grad():
                    logits_rand = model(input_ids).logits[0, -1, :]
                hook.remove()
                lp_rand = F.log_softmax(logits_rand.float(), dim=-1)
                rand_result = (lp_rand[ther_tok] - lp_rand[syc_tok]).item()
                rand_shifts.append(rand_result - baseline)

            random_means.append(float(np.mean(rand_shifts)))

        rand_mean = float(np.mean(random_means))
        rand_std = float(np.std(random_means))
        z = (real_mean - rand_mean) / rand_std if rand_std > 0 else float('inf')
        specific = z > 2.0

        specificity_results[alpha] = {
            "real_shift": real_mean,
            "random_mean": rand_mean,
            "random_std": rand_std,
            "z_score": float(z),
            "specific": bool(specific),
        }
        print(f"  α={alpha}: real={real_mean:+.4f}, random={rand_mean:+.4f}±{rand_std:.4f}, z={z:.2f} {'SPECIFIC' if specific else 'not specific'}")

    # =========================================================
    # SAVE
    # =========================================================
    results = {
        "ols_vs_gs": {
            "gram_schmidt": {"empathy": float(gs_emp), "factual": float(gs_fact), "residual": float(gs_res)},
            "ols_type_iii": {"empathy": float(ols_emp), "factual": float(ols_fact), "residual": float(ols_res)},
            "conclusion_consistent": bool(conclusion_same),
        },
        "reversed_ordering": {
            "empathy_first": {"empathy": float(gs_emp), "factual": float(gs_fact), "residual": float(gs_res)},
            "factual_first": {"empathy": float(gs_rev_emp), "factual": float(gs_rev_fact), "residual": float(gs_rev_res)},
            "order_robust": bool(order_robust),
        },
        "pca": {
            "layer": pca_layer,
            "pc1_variance": pc1_var,
            "top5_cumulative": pc5_cumvar,
            "top10_cumulative": pc10_cumvar,
            "cos_mean_diff_pc1": cos_mean_pc1,
        },
        "steering_specificity": specificity_results,
    }

    with open(os.path.join(output_dir, "h17_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Left: GS vs OLS comparison
    ax = axes[0]
    methods = ["Gram-Schmidt\n(emp first)", "Gram-Schmidt\n(fact first)", "OLS\n(order-free)"]
    emp_vals = [gs_emp, gs_rev_emp, ols_emp]
    fact_vals = [gs_fact, gs_rev_fact, ols_fact]
    x = np.arange(len(methods))
    ax.bar(x - 0.15, emp_vals, 0.3, label="Empathy", color="#ff7f00")
    ax.bar(x + 0.15, fact_vals, 0.3, label="Factual", color="#377eb8")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=8)
    ax.set_ylabel("Unique Variance Explained")
    ax.set_title("V1/V2: Decomposition Robustness")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # Middle: PCA scree plot
    ax = axes[1]
    n_show = min(15, len(explained_variance))
    ax.bar(range(n_show), [float(explained_variance[i]) for i in range(n_show)], color="#4daf4a")
    ax.axhline(0.5, color="red", linestyle="--", alpha=0.5, label="50% threshold")
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Variance Explained")
    ax.set_title(f"V3: PCA (PC1={pc1_var:.1%}, cos(mean,PC1)={cos_mean_pc1:.2f})")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # Right: Steering specificity by alpha
    ax = axes[2]
    alphas = sorted(specificity_results.keys())
    real_vals = [specificity_results[a]["real_shift"] for a in alphas]
    rand_vals = [specificity_results[a]["random_mean"] for a in alphas]
    rand_errs = [specificity_results[a]["random_std"] for a in alphas]
    ax.errorbar([str(a) for a in alphas], rand_vals, yerr=rand_errs,
                fmt="o-", color="#999999", label="Random (mean±std)", capsize=5)
    ax.plot([str(a) for a in alphas], real_vals, "s-", color="#e41a1c", label="Sycophancy direction")
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Alpha")
    ax.set_ylabel("Mean Shift (therapeutic - sycophantic)")
    ax.set_title("V4: Steering Specificity by Alpha")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle("H17: Methodology Validation — Reviewer Controls", fontsize=13)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "h17_methodology_validation.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\nH17 results saved to {output_dir}")
    return results
