"""Advanced steering methods for activation intervention experiments.

Provides reusable hook-based steering utilities:
  - Single-layer and multi-layer activation addition/subtraction.
  - Subspace steering along multiple directions simultaneously.
  - Evaluation harness with random-vector baselines and z-scores.
  - Nonlinear (MLP-based) steering trained on cached activations.
"""

import gc
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from pals.models import get_device


# ---------------------------------------------------------------------------
# Core hook factory
# ---------------------------------------------------------------------------

def steer_hook(direction, alpha, position=-1):
    """Return a forward-hook function that subtracts alpha * direction.

    The hook modifies the residual stream at the given token position.
    Direction should already be on the correct device and dtype.

    Args:
        direction: (hidden_dim,) tensor — the steering direction.
        alpha: float — scaling factor for the intervention.
        position: int — token position to modify (-1 = last token).

    Returns:
        A hook function compatible with ``register_forward_hook``.
    """
    def hook_fn(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        h = h.clone()
        h[:, position, :] -= alpha * direction
        if isinstance(out, tuple):
            return (h,) + out[1:]
        return h
    return hook_fn


# ---------------------------------------------------------------------------
# Context managers
# ---------------------------------------------------------------------------

@contextmanager
def apply_steering(model, layer, direction, alpha, position=-1):
    """Context manager that installs a steering hook on a single layer.

    The direction is cast to the model's dtype before hooking.

    Usage::

        with apply_steering(model, 42, vec, 8.0):
            logits = model(ids).logits
    """
    dtype = next(model.parameters()).dtype
    device = get_device(model)
    d = direction.to(device=device, dtype=dtype)
    hook = model.model.layers[layer].register_forward_hook(
        steer_hook(d, alpha, position)
    )
    try:
        yield
    finally:
        hook.remove()


@contextmanager
def apply_multi_layer_steering(model, layer_direction_alpha_list, position=-1):
    """Context manager for coordinated multi-layer steering.

    Args:
        model: HuggingFace model with model.model.layers structure.
        layer_direction_alpha_list: List of (layer, direction, alpha) tuples.
        position: int — token position to modify.
    """
    dtype = next(model.parameters()).dtype
    device = get_device(model)
    hooks = []
    for layer, direction, alpha in layer_direction_alpha_list:
        d = direction.to(device=device, dtype=dtype)
        h = model.model.layers[layer].register_forward_hook(
            steer_hook(d, alpha, position)
        )
        hooks.append(h)
    try:
        yield
    finally:
        for h in hooks:
            h.remove()


@contextmanager
def apply_subspace_steering(model, layer, directions_matrix, alphas, position=-1):
    """Steer along multiple directions at once on a single layer.

    The net offset subtracted is  sum_k( alphas[k] * directions_matrix[k] ).

    Args:
        model: HuggingFace model.
        layer: int — target layer index.
        directions_matrix: (k, hidden_dim) tensor — k direction vectors.
        alphas: (k,) tensor or array — per-direction scaling factors.
        position: int — token position to modify.
    """
    dtype = next(model.parameters()).dtype
    device = get_device(model)
    dirs = directions_matrix.to(device=device, dtype=dtype)
    a = torch.as_tensor(alphas, device=device, dtype=dtype)
    # Pre-compute the combined offset: (k,) @ (k, hidden_dim) -> (hidden_dim,)
    combined = a @ dirs

    def hook_fn(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        h = h.clone()
        h[:, position, :] -= combined
        if isinstance(out, tuple):
            return (h,) + out[1:]
        return h

    hook = model.model.layers[layer].register_forward_hook(hook_fn)
    try:
        yield
    finally:
        hook.remove()


@contextmanager
def apply_nonlinear_steering(model, layer, mlp, position=-1):
    """Apply a learned MLP steering offset at a given layer."""
    device = get_device(model)
    model_dtype = next(model.parameters()).dtype

    def hook_fn(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        h = h.clone()
        act = h[:, position, :].float().cpu()
        offset = mlp(act).to(device=device, dtype=model_dtype)
        h[:, position, :] -= offset
        if isinstance(out, tuple):
            return (h,) + out[1:]
        return h

    hook = model.model.layers[layer].register_forward_hook(hook_fn)
    try:
        yield
    finally:
        hook.remove()


# ---------------------------------------------------------------------------
# Evaluation harness
# ---------------------------------------------------------------------------

def _logit_diff(model, tokenizer, stimulus):
    """Compute log P(therapeutic) - log P(sycophantic) for a single stimulus."""
    device = get_device(model)
    input_ids = tokenizer.encode(stimulus["user_prompt"], return_tensors="pt").to(device)
    ther_tok = tokenizer.encode(
        stimulus["therapeutic_completion"], add_special_tokens=False
    )[0]
    syc_tok = tokenizer.encode(
        stimulus["sycophantic_completion"], add_special_tokens=False
    )[0]

    with torch.no_grad():
        logits = model(input_ids).logits[0, -1, :]
    lp = F.log_softmax(logits.float(), dim=-1)
    return (lp[ther_tok] - lp[syc_tok]).item()


@torch.no_grad()
def evaluate_steering_effect(model, tokenizer, stimuli, steering_context_fn,
                             n_random=10, random_layer=None, random_alpha=8.0):
    """Evaluate steering vs baseline and random-vector controls.

    Args:
        model: HuggingFace causal LM.
        tokenizer: Matching tokenizer.
        stimuli: List of stimulus dicts.
        steering_context_fn: Callable that returns a context manager.
        n_random: Number of random direction trials for z-score.
        random_layer: Layer to use for random-vector baseline. Required if
            steering_context_fn doesn't support direction_override.
        random_alpha: Alpha for random-vector baseline.

    Returns:
        dict with keys: baseline_diffs, steered_diffs, mean_baseline,
        mean_steered, mean_shift, random_mean_shifts, z_score.
    """
    import inspect

    # Check whether steering_context_fn supports direction_override
    sig = inspect.signature(steering_context_fn)
    supports_override = "direction_override" in sig.parameters

    # Baselines (no intervention)
    baseline_diffs = [_logit_diff(model, tokenizer, s) for s in stimuli]
    mean_baseline = float(np.mean(baseline_diffs))

    # Steered
    steered_diffs = []
    for s in stimuli:
        with steering_context_fn():
            steered_diffs.append(_logit_diff(model, tokenizer, s))
    mean_steered = float(np.mean(steered_diffs))
    mean_shift = mean_steered - mean_baseline

    # Random-vector baselines for z-score
    hidden_dim = model.config.hidden_size
    device = get_device(model)
    dtype = next(model.parameters()).dtype
    random_shifts = []

    for _ in range(n_random):
        rand_dir = torch.randn(hidden_dim, device=device, dtype=dtype)
        rand_dir = F.normalize(rand_dir, dim=0)

        if supports_override:
            rand_diffs = []
            for s in stimuli:
                with steering_context_fn(direction_override=rand_dir):
                    rand_diffs.append(_logit_diff(model, tokenizer, s))
        elif random_layer is not None:
            rand_diffs = []
            for s in stimuli:
                with apply_steering(model, random_layer, rand_dir, random_alpha):
                    rand_diffs.append(_logit_diff(model, tokenizer, s))
        else:
            # No layer info available — measure baselines as noise floor
            rand_diffs = [_logit_diff(model, tokenizer, s) for s in stimuli]

        random_shifts.append(float(np.mean(rand_diffs)) - mean_baseline)

    # z-score: how many SDs is the real shift from the random-shift mean
    random_shifts = np.array(random_shifts)
    rand_std = float(np.std(random_shifts)) if len(random_shifts) > 1 else 1e-8
    z_score = float(mean_shift - np.mean(random_shifts)) / max(rand_std, 1e-8)

    return {
        "baseline_diffs": baseline_diffs,
        "steered_diffs": steered_diffs,
        "mean_baseline": mean_baseline,
        "mean_steered": mean_steered,
        "mean_shift": mean_shift,
        "random_mean_shifts": random_shifts.tolist(),
        "z_score": z_score,
    }


# ---------------------------------------------------------------------------
# Nonlinear steering (small MLP trained on cached activations)
# ---------------------------------------------------------------------------

class SteeringMLP(nn.Module):
    """Small MLP that maps a residual-stream vector to a steering offset.

    Architecture: Linear -> ReLU -> Linear, with a small bottleneck to
    prevent overfitting on ~30 stimuli.
    """

    def __init__(self, input_dim, hidden_size=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_dim),
        )

    def forward(self, x):
        return self.net(x)


def train_nonlinear_steering(model, tokenizer, stimuli, layer,
                             hidden_size=32, epochs=30, lr=1e-3):
    """Train a small MLP to produce per-stimulus steering offsets.

    Training procedure (does NOT backprop through the full model):
      1. Cache residual-stream activations and the unembedding weight.
      2. For each stimulus, compute the logit diff when adding the MLP's
         output to the cached activation, using the frozen unembedding.
      3. Optimize to maximise log_prob(therapeutic) - log_prob(sycophantic).

    Args:
        model: HuggingFace causal LM (frozen during training).
        tokenizer: Matching tokenizer.
        stimuli: List of stimulus dicts.
        layer: int — layer whose residual stream to steer.
        hidden_size: Bottleneck width of the MLP.
        epochs: Training epochs.
        lr: Learning rate.

    Returns:
        SteeringMLP — trained MLP (on CPU, float32).
    """
    device = get_device(model)
    dtype = next(model.parameters()).dtype

    # --- Step 1: cache activations and token indices ---
    cached_acts = []   # (hidden_dim,) per stimulus
    ther_toks = []
    syc_toks = []

    print("Caching activations...")
    for s in tqdm(stimuli, desc="Cache"):
        input_ids = tokenizer.encode(
            s["user_prompt"], return_tensors="pt"
        ).to(device)

        # Capture the residual stream at `layer`, last token
        act_store = {}

        def make_hook(store):
            def fn(module, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                store["act"] = h[:, -1, :].detach().cpu().float()
            return fn

        hook = model.model.layers[layer].register_forward_hook(
            make_hook(act_store)
        )
        with torch.no_grad():
            model(input_ids)
        hook.remove()

        cached_acts.append(act_store["act"].squeeze(0))  # (hidden_dim,)
        ther_toks.append(
            tokenizer.encode(s["therapeutic_completion"],
                             add_special_tokens=False)[0]
        )
        syc_toks.append(
            tokenizer.encode(s["sycophantic_completion"],
                             add_special_tokens=False)[0]
        )

    # Stack cached activations: (n_stimuli, hidden_dim)
    cached_acts = torch.stack(cached_acts)

    # Get the unembedding matrix (lm_head weight) and final layernorm
    # so we can compute logits without running the full model.
    lm_head_weight = model.lm_head.weight.detach().cpu().float()  # (vocab, hidden)
    if hasattr(model.model, "norm"):
        # Apply final RMSNorm / LayerNorm to cached acts
        # We approximate by copying the norm to CPU
        norm_layer = model.model.norm
        norm_weight = norm_layer.weight.detach().cpu().float()
    else:
        norm_weight = None

    def apply_norm(x):
        """Apply RMSNorm (OLMo / Llama style) on CPU."""
        if norm_weight is None:
            return x
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + 1e-5)
        return x * norm_weight

    ther_toks_t = torch.tensor(ther_toks, dtype=torch.long)
    syc_toks_t = torch.tensor(syc_toks, dtype=torch.long)

    # --- Step 2: train MLP on cached data ---
    hidden_dim = cached_acts.shape[1]
    mlp = SteeringMLP(hidden_dim, hidden_size)
    optimizer = torch.optim.Adam(mlp.parameters(), lr=lr, weight_decay=0.1)

    print(f"Training nonlinear steering MLP ({hidden_size} hidden)...")
    train_log = []
    for epoch in range(epochs):
        optimizer.zero_grad()

        offsets = mlp(cached_acts)              # (n, hidden_dim)
        steered = cached_acts + offsets          # (n, hidden_dim)
        normed = apply_norm(steered)             # (n, hidden_dim)
        logits = normed @ lm_head_weight.T       # (n, vocab)
        lp = F.log_softmax(logits, dim=-1)       # (n, vocab)

        # Gather log-probs for therapeutic and sycophantic tokens
        lp_ther = lp[torch.arange(len(stimuli)), ther_toks_t]
        lp_syc = lp[torch.arange(len(stimuli)), syc_toks_t]

        # Loss: negative therapeutic advantage + L2 penalty on offset magnitude
        offset_penalty = 0.01 * (offsets ** 2).mean()
        loss = -(lp_ther - lp_syc).mean() + offset_penalty
        loss.backward()
        torch.nn.utils.clip_grad_norm_(mlp.parameters(), 1.0)
        optimizer.step()
        train_log.append(loss.item())

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  epoch {epoch+1:3d}/{epochs}  "
                  f"loss={loss.item():+.4f}  "
                  f"mean_diff={(lp_ther - lp_syc).mean().item():+.4f}")

    # Clean up
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    mlp.eval()
    return mlp, train_log
