#!/usr/bin/env python3
"""Run experiments locally on OLMo-2 1B for validation.

Usage:
    python run_local.py                     # Run all experiments
    python run_local.py --experiment h1     # Run only H1
    python run_local.py --experiment h4-h9  # Run only new hypotheses
    python run_local.py --device mps        # Use Apple GPU
"""

import argparse
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from pals.models import load_model, load_tokenizer, LOCAL_MODEL, OLMO2_CHECKPOINTS


def inject_metadata(result_json_path, metadata):
    """Read a result JSON, add metadata, and re-write it."""
    with open(result_json_path) as f:
        data = json.load(f)
    data["metadata"] = metadata
    with open(result_json_path, "w") as f:
        json.dump(data, f, indent=2)

ALL_EXPERIMENTS = ["h1", "h2", "h3", "h4", "h5", "h6", "h7", "h8", "h9",
                   "h10", "h11", "h12", "h13", "h14", "h15", "h16", "h17"]


def main():
    parser = argparse.ArgumentParser(description="PALS: Local validation on OLMo-2 1B")
    parser.add_argument("--model", default=LOCAL_MODEL, help="Model name")
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    parser.add_argument("--stimuli-dir", default="stimuli", help="Path to stimulus JSONs")
    parser.add_argument("--output-dir", default="results", help="Output directory")
    parser.add_argument("--experiment", default="all",
                        choices=["all", "h1-h3", "h4-h9", "h10-h17"] + ALL_EXPERIMENTS)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.experiment == "all":
        to_run = ALL_EXPERIMENTS
    elif args.experiment == "h1-h3":
        to_run = ["h1", "h2", "h3"]
    elif args.experiment == "h4-h9":
        to_run = ["h4", "h5", "h6", "h7", "h8", "h9"]
    elif args.experiment == "h10-h17":
        to_run = ["h10", "h11", "h12", "h13", "h14", "h15", "h16", "h17"]
    else:
        to_run = [args.experiment]

    # Load model
    print(f"Loading model: {args.model} on {args.device}...")
    t0 = time.time()
    model = load_model(args.model, device=args.device)
    tokenizer = load_tokenizer(args.model)
    print(f"Model loaded in {time.time() - t0:.1f}s")
    print(f"  Layers: {model.config.num_hidden_layers}")
    print(f"  Hidden dim: {model.config.hidden_size}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M")

    # Build metadata dict
    metadata = {
        "model": args.model,
        "device": args.device,
        "num_layers": model.config.num_hidden_layers,
        "hidden_size": model.config.hidden_size,
        "num_params_M": round(sum(p.numel() for p in model.parameters()) / 1e6),
        "platform": platform.platform(),
        "torch_version": __import__("torch").__version__,
    }

    # Verify stimuli
    required = ["cognitive_distortions.json", "clinical_bridge.json",
                "factual_control.json", "high_emotion_general.json",
                "clinical_correct_answer.json"]
    if any(h in to_run for h in ["h6", "h11", "h13"]):
        required.append("emotional_intensity_gradient.json")

    for fname in required:
        path = os.path.join(args.stimuli_dir, fname)
        if not os.path.exists(path):
            print(f"ERROR: Missing stimulus file: {path}")
            sys.exit(1)

    # --- Run experiments ---
    if "h1" in to_run:
        from experiments.h1_direction_comparison import run as run_h1
        t0 = time.time()
        h1_dir = os.path.join(args.output_dir, "h1")
        run_h1(model, tokenizer, args.stimuli_dir, h1_dir)
        elapsed = time.time() - t0
        inject_metadata(os.path.join(h1_dir, "h1_results.json"),
                        {**metadata, "timestamp": datetime.now(timezone.utc).isoformat(), "runtime_s": round(elapsed, 1)})
        print(f"H1 completed in {elapsed:.1f}s\n")

    if "h2" in to_run:
        from experiments.h2_uncertainty_deference import run as run_h2
        t0 = time.time()
        h2_dir = os.path.join(args.output_dir, "h2")
        run_h2(model, tokenizer, args.stimuli_dir, h2_dir)
        elapsed = time.time() - t0
        inject_metadata(os.path.join(h2_dir, "h2_results.json"),
                        {**metadata, "timestamp": datetime.now(timezone.utc).isoformat(), "runtime_s": round(elapsed, 1)})
        print(f"H2 completed in {elapsed:.1f}s\n")

    if "h3" in to_run:
        from experiments.h3_checkpoint_evolution import run as run_h3
        t0 = time.time()
        h3_dir = os.path.join(args.output_dir, "h3")
        run_h3(model, tokenizer, args.stimuli_dir, h3_dir,
               checkpoints=OLMO2_CHECKPOINTS, device=args.device)
        elapsed = time.time() - t0
        inject_metadata(os.path.join(h3_dir, "h3_results.json"),
                        {**metadata, "checkpoints_used": OLMO2_CHECKPOINTS,
                         "timestamp": datetime.now(timezone.utc).isoformat(), "runtime_s": round(elapsed, 1)})
        print(f"H3 completed in {elapsed:.1f}s\n")

    if "h4" in to_run:
        from experiments.h4_attention_routing import run as run_h4
        t0 = time.time()
        h4_dir = os.path.join(args.output_dir, "h4")
        run_h4(model, tokenizer, args.stimuli_dir, h4_dir,
               n_stimuli=5)  # Reduced for local speed
        elapsed = time.time() - t0
        inject_metadata(os.path.join(h4_dir, "h4_results.json"),
                        {**metadata, "timestamp": datetime.now(timezone.utc).isoformat(), "runtime_s": round(elapsed, 1)})
        print(f"H4 completed in {elapsed:.1f}s\n")

    if "h5" in to_run:
        from experiments.h5_dpo_reversal import run as run_h5
        t0 = time.time()
        h5_dir = os.path.join(args.output_dir, "h5")
        run_h5(model, tokenizer, args.stimuli_dir, h5_dir,
               checkpoints=OLMO2_CHECKPOINTS, device=args.device)
        elapsed = time.time() - t0
        inject_metadata(os.path.join(h5_dir, "h5_results.json"),
                        {**metadata, "checkpoints_used": OLMO2_CHECKPOINTS,
                         "timestamp": datetime.now(timezone.utc).isoformat(), "runtime_s": round(elapsed, 1)})
        print(f"H5 completed in {elapsed:.1f}s\n")

    if "h6" in to_run:
        from experiments.h6_emotional_gradient import run as run_h6
        t0 = time.time()
        h6_dir = os.path.join(args.output_dir, "h6")
        run_h6(model, tokenizer, args.stimuli_dir, h6_dir)
        elapsed = time.time() - t0
        inject_metadata(os.path.join(h6_dir, "h6_results.json"),
                        {**metadata, "timestamp": datetime.now(timezone.utc).isoformat(), "runtime_s": round(elapsed, 1)})
        print(f"H6 completed in {elapsed:.1f}s\n")

    if "h7" in to_run:
        from experiments.h7_distortion_subspaces import run as run_h7
        t0 = time.time()
        h7_dir = os.path.join(args.output_dir, "h7")
        run_h7(model, tokenizer, args.stimuli_dir, h7_dir)
        elapsed = time.time() - t0
        inject_metadata(os.path.join(h7_dir, "h7_results.json"),
                        {**metadata, "timestamp": datetime.now(timezone.utc).isoformat(), "runtime_s": round(elapsed, 1)})
        print(f"H7 completed in {elapsed:.1f}s\n")

    if "h8" in to_run:
        from experiments.h8_pretraining_entanglement import run as run_h8
        t0 = time.time()
        h8_dir = os.path.join(args.output_dir, "h8")
        run_h8(model, tokenizer, args.stimuli_dir, h8_dir,
               checkpoints=OLMO2_CHECKPOINTS, device=args.device)
        elapsed = time.time() - t0
        inject_metadata(os.path.join(h8_dir, "h8_results.json"),
                        {**metadata, "checkpoints_used": OLMO2_CHECKPOINTS,
                         "timestamp": datetime.now(timezone.utc).isoformat(), "runtime_s": round(elapsed, 1)})
        print(f"H8 completed in {elapsed:.1f}s\n")

    if "h9" in to_run:
        from experiments.h9_warmth_tax import run as run_h9
        t0 = time.time()
        h9_dir = os.path.join(args.output_dir, "h9")
        run_h9(model, tokenizer, args.stimuli_dir, h9_dir)
        elapsed = time.time() - t0
        inject_metadata(os.path.join(h9_dir, "h9_results.json"),
                        {**metadata, "timestamp": datetime.now(timezone.utc).isoformat(), "runtime_s": round(elapsed, 1)})
        print(f"H9 completed in {elapsed:.1f}s\n")

    if "h10" in to_run:
        from experiments.h10_extended_decomposition import run as run_h10
        t0 = time.time()
        h10_dir = os.path.join(args.output_dir, "h10")
        run_h10(model, tokenizer, args.stimuli_dir, h10_dir)
        elapsed = time.time() - t0
        inject_metadata(os.path.join(h10_dir, "h10_results.json"),
                        {**metadata, "timestamp": datetime.now(timezone.utc).isoformat(), "runtime_s": round(elapsed, 1)})
        print(f"H10 completed in {elapsed:.1f}s\n")

    if "h11" in to_run:
        from experiments.h11_orthogonal_complement import run as run_h11
        t0 = time.time()
        h11_dir = os.path.join(args.output_dir, "h11")
        run_h11(model, tokenizer, args.stimuli_dir, h11_dir)
        elapsed = time.time() - t0
        inject_metadata(os.path.join(h11_dir, "h11_results.json"),
                        {**metadata, "timestamp": datetime.now(timezone.utc).isoformat(), "runtime_s": round(elapsed, 1)})
        print(f"H11 completed in {elapsed:.1f}s\n")

    if "h12" in to_run:
        from experiments.h12_activation_addition import run as run_h12
        t0 = time.time()
        h12_dir = os.path.join(args.output_dir, "h12")
        run_h12(model, tokenizer, args.stimuli_dir, h12_dir, n_stimuli=10)
        elapsed = time.time() - t0
        inject_metadata(os.path.join(h12_dir, "h12_results.json"),
                        {**metadata, "timestamp": datetime.now(timezone.utc).isoformat(), "runtime_s": round(elapsed, 1)})
        print(f"H12 completed in {elapsed:.1f}s\n")

    if "h13" in to_run:
        from experiments.h13_distortion_emotion_interaction import run as run_h13
        t0 = time.time()
        h13_dir = os.path.join(args.output_dir, "h13")
        run_h13(model, tokenizer, args.stimuli_dir, h13_dir)
        elapsed = time.time() - t0
        inject_metadata(os.path.join(h13_dir, "h13_results.json"),
                        {**metadata, "timestamp": datetime.now(timezone.utc).isoformat(), "runtime_s": round(elapsed, 1)})
        print(f"H13 completed in {elapsed:.1f}s\n")

    if "h14" in to_run:
        from experiments.h14_token_attribution import run as run_h14
        t0 = time.time()
        h14_dir = os.path.join(args.output_dir, "h14")
        run_h14(model, tokenizer, args.stimuli_dir, h14_dir, n_stimuli=15)
        elapsed = time.time() - t0
        inject_metadata(os.path.join(h14_dir, "h14_results.json"),
                        {**metadata, "timestamp": datetime.now(timezone.utc).isoformat(), "runtime_s": round(elapsed, 1)})
        print(f"H14 completed in {elapsed:.1f}s\n")

    if "h15" in to_run:
        from experiments.h15_circuit_decomposition import run as run_h15
        t0 = time.time()
        h15_dir = os.path.join(args.output_dir, "h15")
        run_h15(model, tokenizer, args.stimuli_dir, h15_dir)
        elapsed = time.time() - t0
        inject_metadata(os.path.join(h15_dir, "h15_results.json"),
                        {**metadata, "timestamp": datetime.now(timezone.utc).isoformat(), "runtime_s": round(elapsed, 1)})
        print(f"H15 completed in {elapsed:.1f}s\n")

    if "h16" in to_run:
        from experiments.h16_publication_validation import run as run_h16
        t0 = time.time()
        h16_dir = os.path.join(args.output_dir, "h16")
        run_h16(model, tokenizer, args.stimuli_dir, h16_dir)
        elapsed = time.time() - t0
        inject_metadata(os.path.join(h16_dir, "h16_results.json"),
                        {**metadata, "timestamp": datetime.now(timezone.utc).isoformat(), "runtime_s": round(elapsed, 1)})
        print(f"H16 completed in {elapsed:.1f}s\n")

    if "h17" in to_run:
        from experiments.h17_methodology_validation import run as run_h17
        t0 = time.time()
        h17_dir = os.path.join(args.output_dir, "h17")
        run_h17(model, tokenizer, args.stimuli_dir, h17_dir)
        elapsed = time.time() - t0
        inject_metadata(os.path.join(h17_dir, "h17_results.json"),
                        {**metadata, "timestamp": datetime.now(timezone.utc).isoformat(), "runtime_s": round(elapsed, 1)})
        print(f"H17 completed in {elapsed:.1f}s\n")

    print("=" * 60)
    print(f"Experiments {', '.join(to_run)} complete. Results in: {args.output_dir}")


if __name__ == "__main__":
    main()
