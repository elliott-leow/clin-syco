"""Logit lens: project intermediate hidden states through the unembedding matrix."""

import torch
import torch.nn.functional as F
from tqdm import tqdm
from pals.models import get_device


@torch.no_grad()
def logit_lens(model, input_ids, position=-1):
    """Apply logit lens at each layer for a specific token position.

    Projects the residual stream at each layer through the final layernorm
    and unembedding matrix to get vocabulary logits.

    Args:
        model: HuggingFace model.
        input_ids: (1, seq_len) tensor.
        position: Token position to analyze (-1 for last token).

    Returns:
        dict[int, Tensor]: layer_idx -> (vocab_size,) logits on CPU.
    """
    device = get_device(model)
    hidden_states = {}
    hooks = []
    n_layers = model.config.num_hidden_layers

    def make_hook(idx):
        def fn(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            hidden_states[idx] = h.detach().cpu()
        return fn

    for i in range(n_layers):
        hooks.append(model.model.layers[i].register_forward_hook(make_hook(i)))

    try:
        model(input_ids.to(device))
    finally:
        for h in hooks:
            h.remove()

    norm = model.model.norm
    lm_head = model.lm_head

    logits = {}
    for i in range(n_layers):
        model_dtype = next(model.parameters()).dtype
        h = hidden_states[i][0, position, :].unsqueeze(0).unsqueeze(0).to(device=device, dtype=model_dtype)
        normed = norm(h)
        logits[i] = lm_head(normed).squeeze().float().cpu()

    return logits


def compute_correct_signal(model, tokenizer, prompt,
                           therapeutic_completion, sycophantic_completion):
    """Compute the 'correct answer signal' at each layer.

    Signal = log_prob(first therapeutic token) - log_prob(first sycophantic token)
    at the last prompt token position. Positive means the layer internally
    favors the therapeutic (correct) response.

    Returns:
        dict[int, float]: layer_idx -> signal strength.
    """
    ther_ids = tokenizer.encode(therapeutic_completion, add_special_tokens=False)
    syc_ids = tokenizer.encode(sycophantic_completion, add_special_tokens=False)

    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    layer_logits = logit_lens(model, input_ids, position=-1)

    signal = {}
    for layer_idx, lgt in layer_logits.items():
        log_probs = F.log_softmax(lgt, dim=-1)
        ther_lp = log_probs[ther_ids[0]].item()
        syc_lp = log_probs[syc_ids[0]].item()
        signal[layer_idx] = ther_lp - syc_lp

    return signal


def batch_correct_signal(model, tokenizer, stimuli, desc="Logit lens"):
    """Compute correct signal for a batch of stimuli.

    Each stimulus must have "user_prompt", "therapeutic_completion",
    and "sycophantic_completion" keys.

    Returns:
        list[dict[int, float]]: Per-stimulus signal dicts.
    """
    signals = []
    for s in tqdm(stimuli, desc=desc):
        sig = compute_correct_signal(
            model, tokenizer,
            s["user_prompt"],
            s["therapeutic_completion"],
            s["sycophantic_completion"],
        )
        signals.append(sig)
    return signals


def aggregate_signals(signals):
    """Average signal across stimuli at each layer.

    Returns:
        dict with "mean" and "std" per layer, plus "layers" list.
    """
    layers = sorted(signals[0].keys())
    import numpy as np
    matrix = np.array([[s[l] for l in layers] for s in signals])

    return {
        "layers": layers,
        "mean": {l: matrix[:, i].mean() for i, l in enumerate(layers)},
        "std": {l: matrix[:, i].std() for i, l in enumerate(layers)},
        "raw": matrix,
    }
