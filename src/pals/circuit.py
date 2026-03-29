"""Circuit discovery: activation patching, component ranking, and evaluation."""

import gc
import random
from collections import defaultdict

import torch
import torch.nn.functional as F
from tqdm import tqdm

from pals.models import get_device


# ---------------------------------------------------------------------------
# Patching primitives (inline; these would live in pals.patching if it existed)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _compute_logit_diff(model, input_ids, tok_a, tok_b):
    """Log-prob difference: log P(tok_a) - log P(tok_b) at the last position."""
    device = get_device(model)
    logits = model(input_ids.to(device)).logits[0, -1, :].float()
    lp = F.log_softmax(logits, dim=-1)
    return (lp[tok_a] - lp[tok_b]).item()


@torch.no_grad()
def _get_clean_caches(model, input_ids, layers):
    """Run a clean forward pass and cache per-head attention outputs and MLP outputs.

    Returns:
        attn_caches: dict[int, Tensor]  -- layer -> (1, seq_len, hidden_dim)
        mlp_caches:  dict[int, Tensor]  -- layer -> (1, seq_len, hidden_dim)
    """
    device = get_device(model)
    attn_caches = {}
    mlp_caches = {}
    hooks = []

    for layer_idx in layers:
        layer_module = model.model.layers[layer_idx]

        def _attn_hook(mod, inp, out, idx=layer_idx):
            h = out[0] if isinstance(out, tuple) else out
            attn_caches[idx] = h.detach().cpu()

        def _mlp_hook(mod, inp, out, idx=layer_idx):
            h = out[0] if isinstance(out, tuple) else out
            mlp_caches[idx] = h.detach().cpu()

        hooks.append(layer_module.self_attn.register_forward_hook(_attn_hook))
        hooks.append(layer_module.mlp.register_forward_hook(_mlp_hook))

    try:
        model(input_ids.to(device))
    finally:
        for h in hooks:
            h.remove()

    return attn_caches, mlp_caches


@torch.no_grad()
def _run_with_patch(model, input_ids, patch_spec, tok_a, tok_b):
    """Forward pass that patches specified components from cached activations.

    Args:
        patch_spec: list of (layer_idx, component_type, component_idx, cached_tensor)
            component_type: 'head' or 'mlp'
            component_idx: head index for 'head', ignored for 'mlp'
            cached_tensor: (1, seq_len, hidden_dim) clean activation to patch in.

    Returns:
        logit_diff: log P(tok_a) - log P(tok_b)
    """
    device = get_device(model)
    n_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // n_heads
    hooks = []

    # Group patches by (layer, component_type) for efficiency
    attn_patches = defaultdict(list)  # layer -> [(head_idx, cached_tensor)]
    mlp_patches = {}                  # layer -> cached_tensor

    for layer_idx, comp_type, comp_idx, cached in patch_spec:
        if comp_type == "head":
            attn_patches[layer_idx].append((comp_idx, cached))
        elif comp_type == "mlp":
            mlp_patches[layer_idx] = cached

    # Install attention hooks
    for layer_idx, head_list in attn_patches.items():
        attn_module = model.model.layers[layer_idx].self_attn

        def _make_attn_hook(heads, clean_attn, n_h=n_heads, h_d=head_dim):
            def hook_fn(mod, inp, out):
                attn_out = out[0] if isinstance(out, tuple) else out
                batch, seq_len, _ = attn_out.shape
                clean = clean_attn.to(device=device, dtype=attn_out.dtype)

                # Reshape to (batch, seq, n_heads, head_dim) — only patch last position
                modified = attn_out.view(batch, seq_len, n_h, h_d).clone()
                clean_r = clean.view(clean.shape[0], clean.shape[1], n_h, h_d)

                for head_idx, _ in heads:
                    modified[:, -1, head_idx, :] = clean_r[:, -1, head_idx, :]

                modified = modified.view(batch, seq_len, -1)
                if isinstance(out, tuple):
                    return (modified,) + out[1:]
                return modified
            return hook_fn

        # All heads at this layer share the same clean cache
        clean_tensor = head_list[0][1]
        hooks.append(attn_module.register_forward_hook(
            _make_attn_hook(head_list, clean_tensor)
        ))

    # Install MLP hooks
    for layer_idx, cached in mlp_patches.items():
        mlp_module = model.model.layers[layer_idx].mlp

        def _make_mlp_hook(clean_mlp):
            def hook_fn(mod, inp, out):
                mlp_out = out[0] if isinstance(out, tuple) else out
                clean = clean_mlp.to(device=device, dtype=mlp_out.dtype)
                modified = mlp_out.clone()
                modified[:, -1, :] = clean[:, -1, :]
                if isinstance(out, tuple):
                    return (modified,) + out[1:]
                return modified
            return hook_fn

        hooks.append(mlp_module.register_forward_hook(_make_mlp_hook(cached)))

    try:
        logits = model(input_ids.to(device)).logits[0, -1, :].float()
        lp = F.log_softmax(logits, dim=-1)
        diff = (lp[tok_a] - lp[tok_b]).item()
    finally:
        for h in hooks:
            h.remove()

    return diff


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_component_effects(
    model, tokenizer, stimulus, critical_layers,
    clean_cache, corrupt_ids, ther_tok, syc_tok,
    baseline_diff, clean_diff,
):
    """Patch each component (attention head, MLP) one at a time and measure recovery.

    For one stimulus, runs the model on *corrupt_ids* while patching each component
    at each critical layer from the *clean_cache*.  Recovery fraction measures how
    much of the clean-minus-corrupt logit difference each component restores.

    Args:
        model: HuggingFace causal LM.
        tokenizer: tokenizer (unused here but kept for API consistency).
        stimulus: dict with stimulus metadata (for reference).
        critical_layers: list of layer indices to test.
        clean_cache: tuple (attn_caches, mlp_caches) from _get_clean_caches.
        corrupt_ids: (1, seq_len) token ids for the corrupted run.
        ther_tok: token id for therapeutic first token.
        syc_tok: token id for sycophantic first token.
        baseline_diff: logit diff under the corrupt run (no patching).
        clean_diff: logit diff under the clean run.

    Returns:
        effects: dict[component_key -> recovery_fraction]
            component_key is (layer, 'head', head_idx) or (layer, 'mlp', 0).
    """
    attn_caches, mlp_caches = clean_cache
    n_heads = model.config.num_attention_heads
    device = get_device(model)
    denom = clean_diff - baseline_diff
    if abs(denom) < 1e-10:
        denom = 1e-10  # avoid division by zero

    effects = {}

    # Test each attention head
    for layer in critical_layers:
        if layer not in attn_caches:
            continue
        for head_idx in range(n_heads):
            patch_spec = [(layer, "head", head_idx, attn_caches[layer])]
            patched_diff = _run_with_patch(
                model, corrupt_ids, patch_spec, ther_tok, syc_tok,
            )
            recovery = (patched_diff - baseline_diff) / denom
            effects[(layer, "head", head_idx)] = recovery

    # Test each MLP
    for layer in critical_layers:
        if layer not in mlp_caches:
            continue
        patch_spec = [(layer, "mlp", 0, mlp_caches[layer])]
        patched_diff = _run_with_patch(
            model, corrupt_ids, patch_spec, ther_tok, syc_tok,
        )
        recovery = (patched_diff - baseline_diff) / denom
        effects[(layer, "mlp", 0)] = recovery

    return effects


def rank_components(effects_list):
    """Aggregate per-stimulus component effects and rank by mean recovery.

    Args:
        effects_list: list of dicts from compute_component_effects, one per stimulus.

    Returns:
        list of (component_key, mean_effect, std_effect) sorted descending by mean.
    """
    # Collect all keys
    all_keys = set()
    for eff in effects_list:
        all_keys.update(eff.keys())

    stats = []
    for key in all_keys:
        vals = [eff[key] for eff in effects_list if key in eff]
        if not vals:
            continue
        mean_val = sum(vals) / len(vals)
        if len(vals) > 1:
            variance = sum((v - mean_val) ** 2 for v in vals) / (len(vals) - 1)
            std_val = variance ** 0.5
        else:
            std_val = 0.0
        stats.append((key, mean_val, std_val))

    stats.sort(key=lambda x: x[1], reverse=True)
    return stats


def build_circuit(ranked, threshold=0.01):
    """Return the set of component keys whose mean effect exceeds the threshold.

    Args:
        ranked: output of rank_components.
        threshold: minimum mean recovery fraction to include.

    Returns:
        set of component keys.
    """
    return {key for key, mean_eff, _ in ranked if mean_eff > threshold}


@torch.no_grad()
def evaluate_circuit(
    model, tokenizer, stimuli, circuit_components, layers=None,
    n_completion_tokens=3,
):
    """Evaluate a circuit by patching all its components simultaneously.

    Patches all circuit components from clean into corrupt and measures total
    recovery.  Also computes recovery for random subsets of the same size and
    for the complement set.

    Args:
        model: HuggingFace causal LM.
        tokenizer: tokenizer for encoding stimuli.
        stimuli: list of stimulus dicts with 'user_prompt',
                 'therapeutic_completion', 'sycophantic_completion'.
        circuit_components: set of component keys (from build_circuit).
        layers: list of layer indices to cache.  If None, inferred from
                circuit_components.
        n_completion_tokens: number of random-subset evaluation rounds.

    Returns:
        dict with 'circuit_recovery', 'random_recovery_mean',
        'random_recovery_std', 'complement_recovery'.
    """
    device = get_device(model)
    n_heads = model.config.num_attention_heads

    # Infer layers from circuit if not specified
    if layers is None:
        layers = sorted({key[0] for key in circuit_components})

    circuit_recoveries = []
    complement_recoveries = []
    random_recoveries = []
    circuit_size = len(circuit_components)

    # Build the full set of all possible components at these layers
    all_components = set()
    for layer in layers:
        for h in range(n_heads):
            all_components.add((layer, "head", h))
        all_components.add((layer, "mlp", 0))

    complement_components = all_components - circuit_components

    for stim in tqdm(stimuli, desc="Evaluating circuit"):
        prompt = stim["user_prompt"]
        ther_comp = stim["therapeutic_completion"]
        syc_comp = stim["sycophantic_completion"]

        ther_tok = tokenizer.encode(ther_comp, add_special_tokens=False)[0]
        syc_tok = tokenizer.encode(syc_comp, add_special_tokens=False)[0]

        # Clean run
        clean_ids = tokenizer.encode(prompt, return_tensors="pt")
        clean_diff = _compute_logit_diff(model, clean_ids, ther_tok, syc_tok)
        attn_caches, mlp_caches = _get_clean_caches(model, clean_ids, layers)

        # Corrupt run: use a shuffled version of the prompt tokens
        # Simple corruption: repeat a neutral token over the prompt length
        token_ids = clean_ids.clone()
        seq_len = token_ids.shape[1]
        # Corrupt by randomly permuting non-special tokens
        perm = torch.randperm(seq_len)
        corrupt_ids = token_ids[:, perm]
        baseline_diff = _compute_logit_diff(model, corrupt_ids, ther_tok, syc_tok)

        denom = clean_diff - baseline_diff
        if abs(denom) < 1e-10:
            continue  # skip degenerate stimuli

        # --- Circuit patching ---
        patch_spec = []
        for key in circuit_components:
            layer_idx, comp_type, comp_idx = key
            if comp_type == "head":
                patch_spec.append((layer_idx, "head", comp_idx, attn_caches[layer_idx]))
            elif comp_type == "mlp":
                patch_spec.append((layer_idx, "mlp", comp_idx, mlp_caches[layer_idx]))
        patched_diff = _run_with_patch(model, corrupt_ids, patch_spec, ther_tok, syc_tok)
        circuit_recoveries.append((patched_diff - baseline_diff) / denom)

        # --- Complement patching ---
        comp_spec = []
        for key in complement_components:
            layer_idx, comp_type, comp_idx = key
            if comp_type == "head":
                comp_spec.append((layer_idx, "head", comp_idx, attn_caches[layer_idx]))
            elif comp_type == "mlp":
                comp_spec.append((layer_idx, "mlp", comp_idx, mlp_caches[layer_idx]))
        comp_diff = _run_with_patch(model, corrupt_ids, comp_spec, ther_tok, syc_tok)
        complement_recoveries.append((comp_diff - baseline_diff) / denom)

        # --- Random subset patching ---
        all_comp_list = list(all_components)
        for _ in range(n_completion_tokens):
            if circuit_size >= len(all_comp_list):
                subset = all_comp_list
            else:
                subset = random.sample(all_comp_list, circuit_size)
            rand_spec = []
            for key in subset:
                layer_idx, comp_type, comp_idx = key
                if comp_type == "head":
                    rand_spec.append((layer_idx, "head", comp_idx, attn_caches[layer_idx]))
                elif comp_type == "mlp":
                    rand_spec.append((layer_idx, "mlp", comp_idx, mlp_caches[layer_idx]))
            rand_diff = _run_with_patch(model, corrupt_ids, rand_spec, ther_tok, syc_tok)
            random_recoveries.append((rand_diff - baseline_diff) / denom)

        # Memory cleanup
        del attn_caches, mlp_caches
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Aggregate results
    def _safe_mean(vals):
        return sum(vals) / len(vals) if vals else float("nan")

    def _safe_std(vals):
        if len(vals) < 2:
            return 0.0
        m = _safe_mean(vals)
        return (sum((v - m) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5

    return {
        "circuit_recovery": _safe_mean(circuit_recoveries),
        "random_recovery_mean": _safe_mean(random_recoveries),
        "random_recovery_std": _safe_std(random_recoveries),
        "complement_recovery": _safe_mean(complement_recoveries),
    }


@torch.no_grad()
def extract_circuit_directions(
    model, tokenizer, stimuli, circuit_components, layers=None,
):
    """Extract contrastive activation directions for each circuit component.

    For each component in the circuit, computes the mean difference between
    activations on therapeutic vs sycophantic completions (the causally-validated
    direction for steering).

    Args:
        model: HuggingFace causal LM.
        tokenizer: tokenizer.
        stimuli: list of stimulus dicts.
        circuit_components: set of component keys.
        layers: list of layer indices to cache.  If None, inferred from
                circuit_components.

    Returns:
        dict[component_key -> direction_vector (1-D CPU tensor)].
    """
    device = get_device(model)
    n_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // n_heads

    if layers is None:
        layers = sorted({key[0] for key in circuit_components})

    # Accumulate per-component activations for therapeutic and sycophantic
    ther_accum = defaultdict(list)
    syc_accum = defaultdict(list)

    for stim in tqdm(stimuli, desc="Extracting circuit directions"):
        prompt = stim["user_prompt"]
        ther_comp = stim["therapeutic_completion"]
        syc_comp = stim["sycophantic_completion"]

        # Therapeutic completion: encode prompt + first part of therapeutic
        ther_text = prompt + " " + ther_comp
        ther_ids = tokenizer.encode(ther_text, return_tensors="pt")
        prompt_len = tokenizer.encode(prompt, return_tensors="pt").shape[1]

        attn_ther, mlp_ther = _get_clean_caches(model, ther_ids, layers)

        # Sycophantic completion
        syc_text = prompt + " " + syc_comp
        syc_ids = tokenizer.encode(syc_text, return_tensors="pt")

        attn_syc, mlp_syc = _get_clean_caches(model, syc_ids, layers)

        # Extract per-component activations at the last prompt token
        pos = prompt_len - 1  # last prompt token index

        for key in circuit_components:
            layer_idx, comp_type, comp_idx = key

            if comp_type == "head":
                # Extract single head's activation at position `pos`
                if layer_idx in attn_ther:
                    t_act = attn_ther[layer_idx][0, pos, :]  # (hidden_dim,)
                    t_act_r = t_act.view(n_heads, head_dim)
                    ther_accum[key].append(t_act_r[comp_idx].clone())

                if layer_idx in attn_syc:
                    s_act = attn_syc[layer_idx][0, pos, :]
                    s_act_r = s_act.view(n_heads, head_dim)
                    syc_accum[key].append(s_act_r[comp_idx].clone())

            elif comp_type == "mlp":
                if layer_idx in mlp_ther:
                    t_mlp = mlp_ther[layer_idx][0, pos, :]  # (hidden_dim,)
                    ther_accum[key].append(t_mlp.clone())

                if layer_idx in mlp_syc:
                    s_mlp = mlp_syc[layer_idx][0, pos, :]
                    syc_accum[key].append(s_mlp.clone())

        # Memory cleanup
        del attn_ther, mlp_ther, attn_syc, mlp_syc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Compute mean-difference directions
    directions = {}
    for key in circuit_components:
        if key in ther_accum and key in syc_accum and ther_accum[key] and syc_accum[key]:
            ther_mean = torch.stack(ther_accum[key]).mean(dim=0)
            syc_mean = torch.stack(syc_accum[key]).mean(dim=0)
            diff = ther_mean - syc_mean
            directions[key] = F.normalize(diff, dim=0)

    return directions
