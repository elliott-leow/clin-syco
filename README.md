# PALS: Probing Alignment in Language-model Sycophancy

Mechanistic interpretability study of clinical sycophancy in language models. We investigate whether LLMs that validate cognitive distortions (e.g., catastrophizing, mind-reading) use the same internal mechanisms as factual sycophancy (agreeing with wrong facts), and how preference optimization reshapes these representations.

## Hypotheses

| # | Hypothesis | Experiment |
|---|-----------|------------|
| H1 | Clinical sycophancy is a different representational direction from factual sycophancy | Contrastive direction extraction + cosine similarity |
| H2 | Clinical sycophancy is driven by social deference, not genuine uncertainty | Logit lens + direction projection |
| H3 | Sycophancy directions evolve across training checkpoints | Checkpoint comparison (base vs instruct) |
| H4 | Attention heads route differently for clinical vs factual sycophancy | Attention pattern analysis + head ablation |
| H5 | DPO reverses factual sycophancy more than clinical sycophancy | Cross-checkpoint direction comparison |
| H6 | Emotional intensity amplifies sycophancy activation | Emotional gradient analysis |
| H7 | Different cognitive distortion types occupy distinct subspaces | Per-distortion direction extraction |
| H8 | Empathy-sycophancy entanglement exists in pretraining | Base model direction analysis |
| H9 | Warmth tokens carry a sycophancy "tax" | Token-level decomposition |
| H10 | Sycophancy decomposes into empathy + frame-acceptance components | Extended component decomposition |
| H11 | Orthogonal complement of empathy isolates frame-acceptance | Projection-based decomposition |
| H12 | Activation steering can reduce clinical sycophancy | Activation addition experiments |
| H13 | Distortion type and emotional intensity interact | Interaction analysis |
| H14 | Sycophancy-triggering tokens are identifiable | Token-level attribution |
| H15 | Circuit decomposition reveals sycophancy pathways | Layer-by-layer circuit analysis |
| H16 | Results replicate under publication-grade validation | Statistical validation suite |
| H17 | Methodology controls pass reviewer scrutiny | Order-invariant decomposition, PCA variance, low-alpha steering |

## Models

- **Local validation:** OLMo-2 1B (`allenai/OLMo-2-0425-1B`)
- **Full experiments:** OLMo-3 7B Base / SFT / DPO (`allenai/Olmo-3-1025-7B`, `-Instruct-SFT`, `-Instruct-DPO`)

## Project Structure

```
experiments/     # Experiment scripts (h1-h17), each with a run() entry point
src/pals/        # Core library: extraction, directions, probing, stats, viz
stimuli/         # Stimulus JSON files (clinical, factual, bridge, emotional)
results/         # Output JSON + plots per hypothesis
notebooks/       # Colab notebook for full-scale experiments
run_local.py     # CLI runner for local validation on OLMo-2 1B
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

## Usage

```bash
# Run all experiments locally
python run_local.py --device mps

# Run a specific experiment
python run_local.py --experiment h1

# Run a group
python run_local.py --experiment h1-h3
```

For full-scale experiments on OLMo-3 7B, use the Colab notebook in `notebooks/`.

## Library Modules

| Module | Purpose |
|--------|---------|
| `pals.extraction` | Batch activation extraction with contrastive pairs |
| `pals.directions` | Contrastive direction computation and cosine similarity |
| `pals.probing` | Linear probe training, cross-domain transfer |
| `pals.logit_lens` | Layer-wise logit lens analysis |
| `pals.attention` | Attention pattern extraction and head ablation |
| `pals.decomposition` | Gram-Schmidt and OLS component decomposition |
| `pals.stats` | Permutation tests, bootstrap CIs, effect sizes |
| `pals.viz` | Plotting utilities for all experiment types |
| `pals.models` | Model loading, checkpoint configs, tokenizer setup |
