# Hierarchical-Digital-Twins

## About

This codebase accompanies the paper **"Data-Driven Hierarchical Digital Twins of Social Interactions"**. We demonstrate how generative digital twins can be derived from sparse behavioral data in trust games, capturing the latent dynamics of social interaction without restrictive mechanistic assumptions. The approach enables prediction of future behavior, mechanistic analysis of trust-building dynamics, and in-silico experimentation with novel scenarios.

## Overview
- **Model:** Hierarchical (or non-hierarchical) AL-RNN that encodes inputs, runs a recurrent latent dynamics model, and decodes ordinal outputs.
- **Structure:** `encoder` -> `rnn` (hierarchical or flat) -> `decoder` (shared/individual/hierarchical).

## Quick Start
- Run a single training session (defaults shown):

```bash
python train_model.py --model_type hierarchical --M 4 --P 1 --N_feat 10 --n_epochs 5000 --batch_size 64
```
- Launch multiple runs or wrappers using [train_launcher.py](train_launcher.py#L1).

## Key Files
- Training entry: [train_model.py](train_model.py#L1)
- Launcher/multi-run: [train_launcher.py](train_launcher.py#L1)
- Model definition: [model.py](model.py#L1)
- Data loading: [dataset.py](dataset.py#L1)

## Main Hyperparameters
- **Latent size (`--M`)**: dimensionality of latent state.
- **Positive units (`--P`)**: number of positive-only units (model-specific).
- **Features (`--N_feat`)**: hierarchical feature dimension (only for `hierarchical`).
- **Model type (`--model_type`)**: `hierarchical` or `non_hierarchical`.
- **Decoder mode (`--decoder_mode`)**: `shared`, `individual`, or `hierarchical`.
- **Training epochs / batch (`--n_epochs`, `--batch_size`)**: run length and batch size.
- **Learning rates:** `feature_lr`, `projection_lr`, `model_lr`, `encoder_lr` (set inside `train_model.py`).
- **Regularization / objective weights:** `beta_pred`, `beta_enc`, `beta_cons`, `beta_ent` (tune trade-offs between prediction, encoder, consistency, entropy).
- **Teacher forcing (`alpha_start`, `alpha_end`, `use_alpha_scheduling`, `n_interleave`)**: control scheduled teacher forcing.

## Data & Outputs
- Expect preprocessed inputs at `data/stacked_inputs.npy` and `data/stacked_invests.npy` (see `data/`).
- Outputs and checkpoints are written under `results/.../run_<id>/checkpoints` by default; validation metrics saved as JSON in the run directory.

## Notes
- Validation runs occur periodically (configured in [train_model.py](train_model.py#L1)) and the best model (by test MAE) is checkpointed.

