"""Activation extraction using PyTorch forward hooks."""

import gc
import torch
from tqdm import tqdm
from pals.models import get_device


@torch.no_grad()
def extract_activations(model, input_ids, layers=None):
    """Extract residual stream activations at each transformer layer.

    Args:
        model: HuggingFace model with model.model.layers structure.
        input_ids: (1, seq_len) tensor on the model's device.
        layers: List of layer indices, or None for all layers.

    Returns:
        dict[int, Tensor]: layer_idx -> (seq_len, hidden_dim) tensor on CPU.
    """
    hidden = {}
    hooks = []
    n_layers = model.config.num_hidden_layers
    targets = set(layers) if layers is not None else set(range(n_layers))

    def make_hook(idx):
        def fn(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            hidden[idx] = h.detach().cpu().float().squeeze(0)
        return fn

    for i in targets:
        hooks.append(model.model.layers[i].register_forward_hook(make_hook(i)))

    try:
        model(input_ids)
    finally:
        for h in hooks:
            h.remove()

    return hidden


def extract_completion_acts(model, tokenizer, prompt, completion,
                            layers=None, pool="mean"):
    """Extract activations over completion tokens only.

    Concatenates prompt + completion, runs a forward pass, then pools
    activations from the completion token positions.

    Args:
        pool: "mean" averages over completion tokens, "last" takes last token.

    Returns:
        dict[int, Tensor]: layer_idx -> (hidden_dim,) vector.
    """
    prompt_ids = tokenizer.encode(prompt, return_tensors="pt")
    full_text = prompt + " " + completion
    full_ids = tokenizer.encode(full_text, return_tensors="pt")
    full_ids = full_ids.to(get_device(model))

    prompt_len = prompt_ids.shape[1]
    acts = extract_activations(model, full_ids, layers)

    pooled = {}
    for idx, h in acts.items():
        comp = h[prompt_len:]
        if len(comp) == 0:
            comp = h[-1:]
        pooled[idx] = comp.mean(0) if pool == "mean" else comp[-1]
    return pooled


def extract_prompt_acts(model, tokenizer, prompt, layers=None):
    """Extract activation at the last prompt token (before generation)."""
    ids = tokenizer.encode(prompt, return_tensors="pt")
    ids = ids.to(get_device(model))
    acts = extract_activations(model, ids, layers)
    return {idx: h[-1] for idx, h in acts.items()}


def batch_extract_contrastive(model, tokenizer, stimuli, pos_key, neg_key,
                              layers=None, pool="mean", desc="Extracting"):
    """Extract activations for a list of contrastive stimulus pairs.

    Args:
        stimuli: List of dicts with "user_prompt" and keys for pos/neg completions.
        pos_key: Key for the positive (e.g. sycophantic) completion.
        neg_key: Key for the negative (e.g. therapeutic) completion.

    Returns:
        (pos_acts_list, neg_acts_list): Each is a list of dicts mapping
        layer_idx -> (hidden_dim,) tensor.
    """
    pos_list, neg_list = [], []
    for i, s in enumerate(tqdm(stimuli, desc=desc)):
        pos = extract_completion_acts(
            model, tokenizer, s["user_prompt"], s[pos_key], layers, pool
        )
        neg = extract_completion_acts(
            model, tokenizer, s["user_prompt"], s[neg_key], layers, pool
        )
        pos_list.append(pos)
        neg_list.append(neg)
        if (i + 1) % 10 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return pos_list, neg_list


def batch_extract_prompt(model, tokenizer, stimuli, layers=None, desc="Extracting prompts"):
    """Extract last-token prompt activations for a list of stimuli."""
    acts_list = []
    for s in tqdm(stimuli, desc=desc):
        acts = extract_prompt_acts(model, tokenizer, s["user_prompt"], layers)
        acts_list.append(acts)
    return acts_list
