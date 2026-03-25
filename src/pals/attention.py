"""Attention head analysis: extraction, ablation, and routing detection."""

import torch
import torch.nn.functional as F
from tqdm import tqdm
from pals.models import get_device


@torch.no_grad()
def extract_attention_weights(model, input_ids):
    """Extract attention weights from all layers and heads.

    Returns:
        dict[int, Tensor]: layer_idx -> (n_heads, seq_len, seq_len) attention weights.
    """
    device = get_device(model)
    attn_weights = {}
    hooks = []
    n_layers = model.config.num_hidden_layers

    def make_hook(idx):
        def fn(module, args, kwargs, output):
            # Most HF models return (hidden_states, attn_weights, ...) when
            # output_attentions=True. We hook the attention module directly.
            pass
        return fn

    # Use output_attentions=True to get attention weights directly
    outputs = model(input_ids.to(device), output_attentions=True)
    if hasattr(outputs, "attentions") and outputs.attentions is not None:
        for i, attn in enumerate(outputs.attentions):
            attn_weights[i] = attn.detach().float().cpu().squeeze(0)

    return attn_weights


@torch.no_grad()
def get_head_output_contributions(model, input_ids, layer_idx, position=-1):
    """Get per-head output contributions at a specific layer and position.

    Each attention head produces an output vector. We extract these individually
    to measure how much each head contributes to the final residual stream.

    Returns:
        Tensor of shape (n_heads, head_dim) — each head's contribution.
    """
    device = get_device(model)
    hidden_states = {}
    hooks = []

    # Capture the input to the target layer's attention
    def capture_attn_input(module, args, output):
        if isinstance(output, tuple):
            hidden_states["attn_output"] = output[0].detach()
        else:
            hidden_states["attn_output"] = output.detach()

    # Hook the self-attention module
    attn_module = model.model.layers[layer_idx].self_attn
    hooks.append(attn_module.register_forward_hook(capture_attn_input))

    outputs = model(input_ids.to(device), output_attentions=True)

    for h in hooks:
        h.remove()

    if hasattr(outputs, "attentions") and outputs.attentions is not None:
        # Attention weights: (1, n_heads, seq_len, seq_len)
        attn = outputs.attentions[layer_idx][0, :, position, :]  # (n_heads, seq_len)
        return attn.float().cpu()

    return None


@torch.no_grad()
def ablate_heads_and_measure(model, tokenizer, prompt, completion_a, completion_b,
                             layer_idx, heads_to_ablate):
    """Ablate specific attention heads and measure change in completion preference.

    Returns the log-prob difference (completion_a - completion_b) for the first
    token of each completion, with and without ablation.

    Args:
        heads_to_ablate: list of head indices to zero out.

    Returns:
        dict with "baseline_diff" and "ablated_diff".
    """
    device = get_device(model)

    tok_a = tokenizer.encode(completion_a, add_special_tokens=False)
    tok_b = tokenizer.encode(completion_b, add_special_tokens=False)
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    # Baseline (no ablation)
    logits_base = model(input_ids).logits[0, -1, :]
    lp_base = F.log_softmax(logits_base.float(), dim=-1)
    baseline_diff = (lp_base[tok_a[0]] - lp_base[tok_b[0]]).item()

    # Ablation: zero out specific heads in the attention output
    def ablation_hook(module, args, output):
        # output is (attn_output, attn_weights, ...) or just attn_output
        attn_out = output[0] if isinstance(output, tuple) else output
        # attn_out shape: (batch, seq_len, hidden_dim)
        n_heads = model.config.num_attention_heads
        head_dim = model.config.hidden_size // n_heads

        # Reshape to (batch, seq_len, n_heads, head_dim)
        batch, seq_len, _ = attn_out.shape
        reshaped = attn_out.view(batch, seq_len, n_heads, head_dim)

        # Zero out specified heads
        for h in heads_to_ablate:
            reshaped[:, :, h, :] = 0.0

        modified = reshaped.view(batch, seq_len, -1)
        if isinstance(output, tuple):
            return (modified,) + output[1:]
        return modified

    attn_module = model.model.layers[layer_idx].self_attn
    hook = attn_module.register_forward_hook(ablation_hook)

    logits_abl = model(input_ids).logits[0, -1, :]
    lp_abl = F.log_softmax(logits_abl.float(), dim=-1)
    ablated_diff = (lp_abl[tok_a[0]] - lp_abl[tok_b[0]]).item()

    hook.remove()

    return {"baseline_diff": baseline_diff, "ablated_diff": ablated_diff}


def find_sycophancy_routing_heads(model, tokenizer, stimuli, layers_to_test,
                                  n_heads, desc="Head scan"):
    """Scan attention heads to find those whose ablation shifts output
    from sycophantic toward therapeutic.

    For each head in each layer, measures the average shift in
    log_prob(sycophantic) - log_prob(therapeutic) when ablated.
    Negative shift = head was promoting sycophancy.

    Returns:
        dict[(layer, head)] -> mean_shift
    """
    head_effects = {}

    for layer in layers_to_test:
        for head in range(n_heads):
            shifts = []
            for s in tqdm(stimuli, desc=f"L{layer}H{head}", leave=False):
                result = ablate_heads_and_measure(
                    model, tokenizer, s["user_prompt"],
                    s["sycophantic_completion"], s["therapeutic_completion"],
                    layer, [head]
                )
                shift = result["ablated_diff"] - result["baseline_diff"]
                shifts.append(shift)

            mean_shift = sum(shifts) / len(shifts) if shifts else float("nan")
            head_effects[(layer, head)] = mean_shift

    return head_effects
