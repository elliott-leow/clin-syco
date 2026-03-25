"""Model loading and text formatting utilities."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

LOCAL_MODEL = "allenai/OLMo-2-0425-1B"
COLAB_MODEL = "allenai/Olmo-3-1025-7B"

OLMO3_CHECKPOINTS = {
    "base": "allenai/Olmo-3-1025-7B",
    "sft": "allenai/Olmo-3-7B-Instruct-SFT",
    "dpo": "allenai/Olmo-3-7B-Instruct-DPO",
}

OLMO3_32B_CHECKPOINTS = {
    "base": "allenai/Olmo-3-1125-32B",
    "sft": "allenai/Olmo-3.1-32B-Instruct-SFT",
    "dpo": "allenai/Olmo-3.1-32B-Instruct-DPO",
}

OLMO2_CHECKPOINTS = {
    "base": "allenai/OLMo-2-0425-1B",
    "instruct": "allenai/OLMo-2-0425-1B-Instruct",
}


def load_model(name=None, device="cpu", quantize_4bit=False):
    """Load a HuggingFace causal LM for activation extraction.

    Args:
        name: HuggingFace model ID. Defaults to LOCAL_MODEL.
        device: "cpu", "mps", or "cuda".
        quantize_4bit: Use 4-bit quantization (requires bitsandbytes + CUDA).
    """
    if name is None:
        name = LOCAL_MODEL

    kwargs = {"torch_dtype": torch.float32, "attn_implementation": "eager"}

    if quantize_4bit:
        from transformers import BitsAndBytesConfig
        kwargs = {
            "quantization_config": BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
            ),
            "device_map": "auto",
            "attn_implementation": "eager",
        }
    elif device == "cuda":
        kwargs = {"torch_dtype": torch.float16, "device_map": "auto", "attn_implementation": "eager"}

    model = AutoModelForCausalLM.from_pretrained(name, **kwargs)
    if device == "mps" and not quantize_4bit:
        model = model.to(device)
    model.eval()
    return model


def load_tokenizer(name=None):
    """Load tokenizer, setting pad_token if missing."""
    if name is None:
        name = LOCAL_MODEL
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def get_device(model):
    """Get the device of the model's first parameter."""
    return next(model.parameters()).device
