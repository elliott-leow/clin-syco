"""Activation patching primitives for causal circuit discovery."""

import gc
import torch
import torch.nn.functional as F
from pals.models import get_device


def compute_logit_diff(logits, ther_tok_id, syc_tok_id):
    """Compute log_prob(therapeutic) - log_prob(sycophantic) from logits.

    Args:
        logits: (vocab_size,) or (1, vocab_size) tensor of logits.
        ther_tok_id: Token id for the therapeutic first token.
        syc_tok_id: Token id for the sycophantic first token.

    Returns:
        float: log_prob difference (positive = favors therapeutic).
    """
    if logits.dim() > 1:
        logits = logits.squeeze(0)
    log_probs = F.log_softmax(logits.float(), dim=-1)
    return (log_probs[ther_tok_id] - log_probs[syc_tok_id]).item()


@torch.no_grad()
def get_clean_hidden_states(model, input_ids, layers):
    """Forward pass caching residual stream hidden states at specified layers.

    Args:
        model: HuggingFace model with model.model.layers structure.
        input_ids: (1, seq_len) tensor on the model's device.
        layers: List of layer indices to cache.

    Returns:
        dict[int, Tensor]: layer_idx -> (1, seq_len, hidden_dim) tensor on CPU.
    """
    cache = {}
    hooks = []
    targets = set(layers)

    def make_hook(idx):
        def fn(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            cache[idx] = h.detach().cpu()
        return fn

    for i in targets:
        hooks.append(model.model.layers[i].register_forward_hook(make_hook(i)))

    try:
        model(input_ids)
    finally:
        for h in hooks:
            h.remove()

    return cache


@torch.no_grad()
def patch_layer(model, corrupt_ids, clean_cache, patch_layer_idx, position=-1):
    """Run model on corrupt input, patching one layer's hidden state from cache.

    Replaces the residual stream at `position` in layer `patch_layer_idx`
    with the corresponding cached clean activation.

    Args:
        model: HuggingFace model.
        corrupt_ids: (1, seq_len) tensor of corrupted input ids.
        clean_cache: dict from get_clean_hidden_states with the clean run.
        patch_layer_idx: Which layer to patch.
        position: Token position to patch (-1 for last token).

    Returns:
        Tensor: logits at the last token position, shape (vocab_size,).
    """
    device = get_device(model)
    model_dtype = next(model.parameters()).dtype
    clean_vec = clean_cache[patch_layer_idx][:, position, :]  # (1, hidden_dim)

    def hook_fn(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        modified = h.clone()
        modified[:, position, :] = clean_vec.to(device=device, dtype=model_dtype)
        if isinstance(out, tuple):
            return (modified,) + out[1:]
        return modified

    hook = model.model.layers[patch_layer_idx].register_forward_hook(hook_fn)
    try:
        logits = model(corrupt_ids.to(device)).logits[0, -1, :]
    finally:
        hook.remove()

    return logits.float().cpu()


@torch.no_grad()
def patch_attention_head(model, corrupt_ids, clean_attn_cache, patch_layer_idx,
                         head_idx, position=-1):
    """Run model on corrupt input, patching one attention head's output.

    Reshapes the attention output to (batch, seq, n_heads, head_dim),
    replaces a single head at `position`, then reshapes back.

    Args:
        model: HuggingFace model.
        corrupt_ids: (1, seq_len) tensor.
        clean_attn_cache: dict[int, Tensor] from get_sublayer_caches (attn part).
        patch_layer_idx: Which layer to patch.
        head_idx: Which attention head to replace.
        position: Token position to patch (-1 for last token).

    Returns:
        Tensor: logits at the last token position, shape (vocab_size,).
    """
    device = get_device(model)
    model_dtype = next(model.parameters()).dtype
    n_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // n_heads

    clean_attn = clean_attn_cache[patch_layer_idx]  # (1, seq_len, hidden_dim) on CPU

    def hook_fn(module, inp, out):
        attn_out = out[0] if isinstance(out, tuple) else out
        batch, seq_len, _ = attn_out.shape
        modified = attn_out.clone()

        # Reshape to isolate heads
        reshaped = modified.view(batch, seq_len, n_heads, head_dim)
        clean_val = clean_attn.to(device=device, dtype=model_dtype)
        clean_reshaped = clean_val.view(1, -1, n_heads, head_dim)

        # Replace just the target head at the target position
        reshaped[:, position, head_idx, :] = clean_reshaped[:, position, head_idx, :]

        modified = reshaped.view(batch, seq_len, -1)
        if isinstance(out, tuple):
            return (modified,) + out[1:]
        return modified

    attn_module = model.model.layers[patch_layer_idx].self_attn
    hook = attn_module.register_forward_hook(hook_fn)
    try:
        logits = model(corrupt_ids.to(device)).logits[0, -1, :]
    finally:
        hook.remove()

    return logits.float().cpu()


@torch.no_grad()
def patch_mlp(model, corrupt_ids, clean_mlp_cache, patch_layer_idx, position=-1):
    """Run model on corrupt input, patching the MLP sublayer output.

    Args:
        model: HuggingFace model.
        corrupt_ids: (1, seq_len) tensor.
        clean_mlp_cache: dict[int, Tensor] from get_sublayer_caches (mlp part).
        patch_layer_idx: Which layer to patch.
        position: Token position to patch (-1 for last token).

    Returns:
        Tensor: logits at the last token position, shape (vocab_size,).
    """
    device = get_device(model)
    model_dtype = next(model.parameters()).dtype
    clean_vec = clean_mlp_cache[patch_layer_idx][:, position, :]  # (1, hidden_dim)

    def hook_fn(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        modified = h.clone()
        modified[:, position, :] = clean_vec.to(device=device, dtype=model_dtype)
        if isinstance(out, tuple):
            return (modified,) + out[1:]
        return modified

    mlp_module = model.model.layers[patch_layer_idx].mlp
    hook = mlp_module.register_forward_hook(hook_fn)
    try:
        logits = model(corrupt_ids.to(device)).logits[0, -1, :]
    finally:
        hook.remove()

    return logits.float().cpu()


@torch.no_grad()
def get_sublayer_caches(model, input_ids, layers):
    """Cache attention and MLP sublayer outputs at specified layers.

    Args:
        model: HuggingFace model with model.model.layers structure.
        input_ids: (1, seq_len) tensor on the model's device.
        layers: List of layer indices to cache.

    Returns:
        (attn_cache, mlp_cache): Each is dict[int, Tensor] mapping
        layer_idx -> (1, seq_len, hidden_dim) tensor on CPU.
    """
    attn_cache = {}
    mlp_cache = {}
    hooks = []
    targets = set(layers)

    def make_attn_hook(idx):
        def fn(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            attn_cache[idx] = h.detach().cpu()
        return fn

    def make_mlp_hook(idx):
        def fn(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            mlp_cache[idx] = h.detach().cpu()
        return fn

    for i in targets:
        hooks.append(
            model.model.layers[i].self_attn.register_forward_hook(make_attn_hook(i))
        )
        hooks.append(
            model.model.layers[i].mlp.register_forward_hook(make_mlp_hook(i))
        )

    try:
        model(input_ids)
    finally:
        for h in hooks:
            h.remove()

    return attn_cache, mlp_cache


@torch.no_grad()
def causal_trace(model, tokenizer, stimulus, layers=None, n_completion_tokens=3):
    """Run causal tracing for one stimulus: patch from therapeutic into sycophantic.

    Constructs a "therapeutic run" (prompt + start of therapeutic completion) and
    a "sycophantic run" (prompt + start of sycophantic completion). Patches the
    therapeutic hidden states into the sycophantic run at each layer and measures
    how much patching restores the therapeutic logit advantage.

    Args:
        model: HuggingFace model.
        tokenizer: Matching tokenizer.
        stimulus: Dict with "user_prompt", "therapeutic_completion",
                  and "sycophantic_completion" keys.
        layers: List of layer indices to trace, or None for all.
        n_completion_tokens: Number of completion tokens to include in the run.

    Returns:
        dict[int, float]: layer_idx -> recovery_fraction.
            recovery_fraction = (patched_diff - corrupt_diff) / (clean_diff - corrupt_diff)
            where diff = log_prob(ther) - log_prob(syc).
    """
    device = get_device(model)
    n_layers = model.config.num_hidden_layers
    trace_layers = layers if layers is not None else list(range(n_layers))

    prompt = stimulus["user_prompt"]
    ther_comp = stimulus["therapeutic_completion"]
    syc_comp = stimulus["sycophantic_completion"]

    # Encode the first token of each completion for logit diff
    ther_toks = tokenizer.encode(ther_comp, add_special_tokens=False)
    syc_toks = tokenizer.encode(syc_comp, add_special_tokens=False)
    ther_tok_id = ther_toks[0]
    syc_tok_id = syc_toks[0]

    # Build "clean" (therapeutic) input: prompt + first n completion tokens
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    ther_prefix = ther_toks[:n_completion_tokens]
    clean_ids = torch.tensor([prompt_ids + ther_prefix], device=device)

    # Build "corrupt" (sycophantic) input: prompt + first n completion tokens
    syc_prefix = syc_toks[:n_completion_tokens]
    corrupt_ids = torch.tensor([prompt_ids + syc_prefix], device=device)

    # Clean baseline: model's logit diff on therapeutic run
    clean_logits = model(clean_ids).logits[0, -1, :]
    clean_diff = compute_logit_diff(clean_logits, ther_tok_id, syc_tok_id)

    # Corrupt baseline: model's logit diff on sycophantic run
    corrupt_logits = model(corrupt_ids).logits[0, -1, :]
    corrupt_diff = compute_logit_diff(corrupt_logits, ther_tok_id, syc_tok_id)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Cache clean hidden states
    clean_cache = get_clean_hidden_states(model, clean_ids, trace_layers)

    # Patch each layer and measure recovery
    recovery = {}
    denom = clean_diff - corrupt_diff
    for layer_idx in trace_layers:
        patched_logits = patch_layer(model, corrupt_ids, clean_cache, layer_idx)
        patched_diff = compute_logit_diff(patched_logits, ther_tok_id, syc_tok_id)

        if abs(denom) > 1e-8:
            recovery[layer_idx] = (patched_diff - corrupt_diff) / denom
        else:
            recovery[layer_idx] = 0.0

    # Cleanup
    del clean_cache
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return recovery
