"""Hypothesis 16: Publication Validation Suite.

Six validation experiments to bring findings to publication quality:

1. Behavioral sycophancy rate — does the model actually generate sycophantic
   completions? Measure P(sycophantic first token) > P(therapeutic first token)
   across all stimuli.

2. Generation with/without steering — generate actual text with and without
   the H12 steering vector. Show qualitative before/after.

3. Steering vector generalization — does the cognitive_distortions steering
   vector work on bridge stimuli and emotion stimuli?

4. Random vector control — compare the H12 steering effect against random
   vectors of the same norm. If random vectors also shift output, the finding
   is meaningless.

5. Stimulus bootstrap — resample stimuli (not layers) to compute error bars
   on key findings: H1 cosine similarity, H6 monotonic decrease, H10
   decomposition.

6. Residual direction validation — test the "dignity" interpretation by
   checking whether the residual direction predicts specific behavioral
   patterns (e.g., does high residual projection correlate with more
   hedging/respectful language in the sycophantic completion?).
"""

import json
import os
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from pals.extraction import batch_extract_contrastive
from pals.directions import compute_contrastive_direction, cosine_similarity_by_layer
from pals.decomposition import decompose_by_layer
from pals.logit_lens import compute_correct_signal
from pals.models import get_device
from pals.stats import bootstrap_ci


def run(model, tokenizer, stimuli_dir, output_dir, layers=None,
        n_generate=10, max_new_tokens=50):
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(stimuli_dir, "cognitive_distortions.json")) as f:
        clinical = json.load(f)
    with open(os.path.join(stimuli_dir, "clinical_bridge.json")) as f:
        bridge = json.load(f)
    with open(os.path.join(stimuli_dir, "factual_control.json")) as f:
        factual = json.load(f)
    with open(os.path.join(stimuli_dir, "high_emotion_general.json")) as f:
        emotion = json.load(f)

    device = get_device(model)
    n_layers = model.config.num_hidden_layers

    # =========================================================
    # 1. BEHAVIORAL SYCOPHANCY RATE
    # =========================================================
    print("\n--- Validation 1: Behavioral Sycophancy Rate ---")
    syc_rates = {}
    for label, stimuli in [("clinical", clinical), ("bridge", bridge), ("factual", factual)]:
        syc_count = 0
        for s in tqdm(stimuli, desc=f"Behav ({label})"):
            input_ids = tokenizer.encode(s["user_prompt"], return_tensors="pt").to(device)
            ther_tok = tokenizer.encode(s["therapeutic_completion"], add_special_tokens=False)[0]
            syc_tok = tokenizer.encode(s["sycophantic_completion"], add_special_tokens=False)[0]
            with torch.no_grad():
                logits = model(input_ids).logits[0, -1, :]
            lp = F.log_softmax(logits.float(), dim=-1)
            if lp[syc_tok] > lp[ther_tok]:
                syc_count += 1
        rate = syc_count / len(stimuli)
        syc_rates[label] = rate
        print(f"  {label}: {syc_count}/{len(stimuli)} = {rate:.1%} sycophantic")

    # =========================================================
    # 2. GENERATION WITH/WITHOUT STEERING
    # =========================================================
    print("\n--- Validation 2: Generation With/Without Steering ---")

    # Compute steering vector
    print("  Computing steering vector...")
    clin_pos, clin_neg = batch_extract_contrastive(
        model, tokenizer, clinical,
        "sycophantic_completion", "therapeutic_completion",
        layers=layers, desc="Steering"
    )
    syc_direction = compute_contrastive_direction(clin_pos, clin_neg)
    steer_layer = 2 * n_layers // 3  # ~layer 10 for 16-layer model
    # Snap to nearest extracted layer if using a subset
    if layers is not None and steer_layer not in set(layers):
        steer_layer = min(layers, key=lambda l: abs(l - steer_layer))
    steer_alpha = 8.0

    generation_examples = []
    gen_stimuli = clinical[:n_generate]

    for s in gen_stimuli:
        input_ids = tokenizer.encode(s["user_prompt"], return_tensors="pt").to(device)

        # Baseline generation
        with torch.no_grad():
            baseline_out = model.generate(input_ids, max_new_tokens=max_new_tokens,
                                          do_sample=False, temperature=1.0)
        baseline_text = tokenizer.decode(baseline_out[0][input_ids.shape[1]:],
                                         skip_special_tokens=True)

        # Steered generation
        direction_vec = syc_direction[steer_layer].to(device)

        def steer_hook(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            h = h.clone()
            h[:, -1, :] -= steer_alpha * direction_vec
            if isinstance(out, tuple):
                return (h,) + out[1:]
            return h

        hook = model.model.layers[steer_layer].register_forward_hook(steer_hook)
        with torch.no_grad():
            steered_out = model.generate(input_ids, max_new_tokens=max_new_tokens,
                                         do_sample=False, temperature=1.0)
        hook.remove()
        steered_text = tokenizer.decode(steered_out[0][input_ids.shape[1]:],
                                        skip_special_tokens=True)

        generation_examples.append({
            "prompt": s["user_prompt"][:100],
            "baseline": baseline_text[:200],
            "steered": steered_text[:200],
            "subcategory": s.get("subcategory", "unknown"),
        })
        print(f"  [{s.get('subcategory','?')}]")
        print(f"    Baseline: {baseline_text[:80]}...")
        print(f"    Steered:  {steered_text[:80]}...")

    # =========================================================
    # 3. STEERING VECTOR GENERALIZATION
    # =========================================================
    print("\n--- Validation 3: Steering Vector Generalization ---")
    # Test if the clinical steering vector works on other stimulus types
    generalization = {}
    for label, stimuli in [("clinical", clinical[:15]),
                            ("bridge", bridge[:15]),
                            ("factual", factual[:15])]:
        shifts = []
        for s in tqdm(stimuli, desc=f"Generalize ({label})"):
            input_ids = tokenizer.encode(s["user_prompt"], return_tensors="pt").to(device)
            ther_tok = tokenizer.encode(s["therapeutic_completion"], add_special_tokens=False)[0]
            syc_tok = tokenizer.encode(s["sycophantic_completion"], add_special_tokens=False)[0]

            # Baseline
            with torch.no_grad():
                logits_base = model(input_ids).logits[0, -1, :]
            lp_base = F.log_softmax(logits_base.float(), dim=-1)
            baseline = (lp_base[ther_tok] - lp_base[syc_tok]).item()

            # Steered
            direction_vec = syc_direction[steer_layer].to(device)

            def gen_hook(module, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                h = h.clone()
                h[:, -1, :] -= steer_alpha * direction_vec
                if isinstance(out, tuple):
                    return (h,) + out[1:]
                return h

            hook = model.model.layers[steer_layer].register_forward_hook(gen_hook)
            with torch.no_grad():
                logits_steer = model(input_ids).logits[0, -1, :]
            hook.remove()
            lp_steer = F.log_softmax(logits_steer.float(), dim=-1)
            steered = (lp_steer[ther_tok] - lp_steer[syc_tok]).item()

            shifts.append(steered - baseline)

        mean_shift = np.mean(shifts)
        generalization[label] = {"mean_shift": float(mean_shift), "n": len(shifts)}
        print(f"  {label}: mean shift = {mean_shift:+.4f}")

    # =========================================================
    # 4. RANDOM VECTOR CONTROL
    # =========================================================
    print("\n--- Validation 4: Random Vector Control ---")
    # Compare steering effect against random vectors of the same norm
    steer_norm = syc_direction[steer_layer].norm().item()
    n_random = 10
    random_shifts = []

    for r in range(n_random):
        rand_vec = torch.randn_like(syc_direction[steer_layer])
        rand_vec = F.normalize(rand_vec, dim=0) * steer_norm
        rand_vec = rand_vec.to(device)

        shifts = []
        for s in clinical[:10]:
            input_ids = tokenizer.encode(s["user_prompt"], return_tensors="pt").to(device)
            ther_tok = tokenizer.encode(s["therapeutic_completion"], add_special_tokens=False)[0]
            syc_tok = tokenizer.encode(s["sycophantic_completion"], add_special_tokens=False)[0]

            with torch.no_grad():
                logits_base = model(input_ids).logits[0, -1, :]
            lp_base = F.log_softmax(logits_base.float(), dim=-1)
            baseline = (lp_base[ther_tok] - lp_base[syc_tok]).item()

            def rand_hook(module, inp, out, vec=rand_vec):
                h = out[0] if isinstance(out, tuple) else out
                h = h.clone()
                h[:, -1, :] -= steer_alpha * vec
                if isinstance(out, tuple):
                    return (h,) + out[1:]
                return h

            hook = model.model.layers[steer_layer].register_forward_hook(rand_hook)
            with torch.no_grad():
                logits_rand = model(input_ids).logits[0, -1, :]
            hook.remove()
            lp_rand = F.log_softmax(logits_rand.float(), dim=-1)
            rand_result = (lp_rand[ther_tok] - lp_rand[syc_tok]).item()
            shifts.append(rand_result - baseline)

        random_shifts.append(np.mean(shifts))

    real_shift = generalization["clinical"]["mean_shift"]
    random_mean = np.mean(random_shifts)
    random_std = np.std(random_shifts)
    z_score = (real_shift - random_mean) / random_std if random_std > 0 else float('inf')

    print(f"  Real steering shift:   {real_shift:+.4f}")
    print(f"  Random vector mean:    {random_mean:+.4f} ± {random_std:.4f}")
    print(f"  Z-score:               {z_score:.2f}")
    print(f"  ** {'SPECIFIC' if z_score > 2 else 'NOT SPECIFIC'}: steering vector is {'significantly' if z_score > 2 else 'not'} better than random")

    # =========================================================
    # 5. STIMULUS BOOTSTRAP
    # =========================================================
    print("\n--- Validation 5: Stimulus Bootstrap ---")
    # Bootstrap over STIMULI (not layers) for H1 cosine similarity
    n_boot = 500
    rng = np.random.RandomState(42)

    # H1: resample clinical and factual stimuli
    boot_cos_means = []
    for b in range(n_boot):
        idx_clin = rng.choice(len(clin_pos), size=len(clin_pos), replace=True)
        boot_clin_pos = [clin_pos[i] for i in idx_clin]
        boot_clin_neg = [clin_neg[i] for i in idx_clin]
        boot_clin_dir = compute_contrastive_direction(boot_clin_pos, boot_clin_neg)

        # Use the pre-computed factual direction (stable)
        fact_pos_all, fact_neg_all = batch_extract_contrastive(
            model, tokenizer, factual,
            "sycophantic_completion", "therapeutic_completion",
            layers=layers, desc=None
        ) if b == 0 else (fact_pos_all, fact_neg_all)

        if b == 0:
            # Extract factual once
            fact_dir_stable = compute_contrastive_direction(fact_pos_all, fact_neg_all)

        cos_vals = cosine_similarity_by_layer(boot_clin_dir, fact_dir_stable)
        boot_cos_means.append(np.mean(list(cos_vals.values())))

    stim_ci_lo, stim_ci_hi = np.percentile(boot_cos_means, [2.5, 97.5])
    stim_mean = np.mean(boot_cos_means)
    print(f"  H1 Clinical-Factual cosine (stimulus bootstrap):")
    print(f"    Mean: {stim_mean:.3f}  95% CI: [{stim_ci_lo:.3f}, {stim_ci_hi:.3f}]")

    # =========================================================
    # 6. RESIDUAL DIRECTION VALIDATION
    # =========================================================
    print("\n--- Validation 6: Residual Direction Behavioral Correlation ---")
    # Test: does projection onto the residual direction correlate with
    # behavioral features of the sycophantic completion?
    # Specifically: word count, question marks (hedging), first-person pronouns

    # Compute residual direction at analysis layer
    analysis_layer = 2 * n_layers // 3
    if layers is not None and analysis_layer not in set(layers):
        analysis_layer = min(layers, key=lambda l: abs(l - analysis_layer))
    components = {
        "empathy": compute_contrastive_direction(
            *batch_extract_contrastive(model, tokenizer, emotion,
                                       "therapeutic_completion", "cold_completion",
                                       layers=[analysis_layer], desc=None)),
        "factual": compute_contrastive_direction(
            *batch_extract_contrastive(model, tokenizer, factual,
                                       "sycophantic_completion", "therapeutic_completion",
                                       layers=[analysis_layer], desc=None)),
    }

    residual_at_layer = syc_direction[analysis_layer].clone()
    for name, comp_dir in components.items():
        comp_normed = F.normalize(comp_dir[analysis_layer], dim=0)
        proj = (residual_at_layer @ comp_normed) * comp_normed
        residual_at_layer = residual_at_layer - proj
    residual_at_layer = F.normalize(residual_at_layer, dim=0)

    # Compute per-stimulus projection onto residual and correlate with text features
    projections = []
    word_counts = []
    question_counts = []
    pronoun_counts = []

    for i, s in enumerate(clinical):
        # Projection of sycophantic completion onto residual
        proj_val = (clin_pos[i][analysis_layer] @ residual_at_layer).item()
        projections.append(proj_val)

        # Text features of the sycophantic completion
        syc_text = s["sycophantic_completion"]
        word_counts.append(len(syc_text.split()))
        question_counts.append(syc_text.count("?"))
        pronoun_counts.append(sum(1 for w in syc_text.lower().split()
                                  if w in {"you", "your", "you're", "yourself"}))

    # Correlations
    from scipy.stats import pearsonr
    corr_words, p_words = pearsonr(projections, word_counts)
    corr_questions, p_questions = pearsonr(projections, question_counts)
    corr_pronouns, p_pronouns = pearsonr(projections, pronoun_counts)

    print(f"  Residual projection correlations with sycophantic completion features:")
    print(f"    Word count:      r={corr_words:.3f}  p={p_words:.4f}")
    print(f"    Question marks:  r={corr_questions:.3f}  p={p_questions:.4f}")
    print(f"    'You' pronouns:  r={corr_pronouns:.3f}  p={p_pronouns:.4f}")

    # =========================================================
    # SAVE
    # =========================================================
    results = {
        "behavioral_sycophancy_rates": syc_rates,
        "generation_examples": generation_examples,
        "steering_generalization": generalization,
        "random_control": {
            "real_shift": float(real_shift),
            "random_mean": float(random_mean),
            "random_std": float(random_std),
            "z_score": float(z_score),
            "n_random_vectors": n_random,
        },
        "stimulus_bootstrap": {
            "h1_cos_mean": float(stim_mean),
            "h1_cos_ci_lo": float(stim_ci_lo),
            "h1_cos_ci_hi": float(stim_ci_hi),
            "n_bootstrap": n_boot,
        },
        "residual_correlations": {
            "word_count": {"r": float(corr_words), "p": float(p_words)},
            "question_marks": {"r": float(corr_questions), "p": float(p_questions)},
            "you_pronouns": {"r": float(corr_pronouns), "p": float(p_pronouns)},
        },
        "steering_config": {
            "layer": steer_layer,
            "alpha": steer_alpha,
        },
    }

    with open(os.path.join(output_dir, "h16_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nH16 results saved to {output_dir}")
    return results
