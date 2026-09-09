from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from hierarchical_decoder import HierarchicalDecoderCumulativeLink  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import load_dataset  # noqa: E402
from fs_rl_model import loglik as fs_loglik  # noqa: E402

"""
decoder_only_ablation.py
==========================
Tests whether the AL-RNN's recurrent encoder adds explanatory power beyond
its decoder alone, by fitting the SAME decoder class the real model uses
(`HierarchicalDecoderCumulativeLink`, the ordinal/cumulative-link
observation model) directly on STATIC per-trial cue features -- no RNN, no
recurrence, no trial-history dependence at all -- via gradient descent
(there is no closed-form MLE for the cumulative-link model), and comparing
test MAE/NLL to the full AL-RNN's reported number (test MAE 0.71).

Feature vectors of increasing dimensionality are tried, chosen to match the
distinctions already shown to matter in the RL/Bayesian analyses (rather
than an arbitrary compression such as PCA):
    dz=1 ("r"):        the RL model's causally-learned r_t belief (identity
                        + expression, additive, online) -- the most
                        compressed, "value-only" summary.
    dz=2 ("fair+lin"):  [trustee-is-fair indicator, linear expression cue]
    dz=5 ("id+lin"):    [4-dim trustee-identity one-hot, linear expression]
    dz=9 ("raw_cue"):   the full raw cue one-hot (4 identity + 5 expression)
                        -- maximal static information, no compression.

Decoder parameters (beta_0, beta) are fit per subject (decoder_mode=
'individual', matching the reported AL-RNN's ordinal observation model) by
Adam on the ordinal log-likelihood, train timesteps only (first 120 of 160,
identical split to the rest of this repo), then evaluated causally on the
held-out test timesteps -- same MAE/NLL convention used throughout.
"""

DEFAULT_INPUTS = REPO_ROOT / "data" / "stacked_inputs.npy"
DEFAULT_INVESTS = REPO_ROOT / "data" / "stacked_invests.npy"
DEFAULT_OUT = Path(__file__).resolve().parent / "results" / "decoder_only_results.json"

_DUMMY_RL_PARS = np.array([0.1, 0.5] + [0.0] * 4 + [0.0] * 4 + [0.0])


def _trial_to_timestep(feat_trial):
    """(n_trials, dz) -> (2*n_trials, dz): decision (even) steps get feat_trial, feedback (odd)
    steps repeat it (masked out by the decoder's own even/valid mask, so the value is unused)."""
    n_trials, dz = feat_trial.shape
    out = np.zeros((2 * n_trials, dz), dtype=np.float32)
    out[0::2] = feat_trial
    out[1::2] = feat_trial
    return out


def build_features(subjects, mode):
    feats = []
    for data in subjects:
        stim = data[:, 0].astype(int)
        emo = data[:, 1].astype(int)
        if mode == "r":
            _, diag = fs_loglik(_DUMMY_RL_PARS, data, fsmap=(1, 2, 3, 4),
                                 return_diagnostics=True, count_based_lr=True)
            trial_feat = diag["predRR"].reshape(-1, 1).astype(np.float32)
        elif mode == "fair+lin":
            is_fair = np.isin(stim, [1, 3]).astype(np.float32)
            c = (3.0 - emo).astype(np.float32)
            trial_feat = np.stack([is_fair, c], axis=1)
        elif mode == "id+lin":
            onehot_id = np.eye(4)[stim - 1]
            c = (3.0 - emo).reshape(-1, 1)
            trial_feat = np.concatenate([onehot_id, c], axis=1).astype(np.float32)
        elif mode == "raw_cue":
            onehot_id = np.eye(4)[stim - 1]
            onehot_emo = np.eye(5)[emo - 1]
            trial_feat = np.concatenate([onehot_id, onehot_emo], axis=1).astype(np.float32)
        else:
            raise ValueError(mode)
        feats.append(_trial_to_timestep(trial_feat))
    return np.stack(feats)   # (n_subjects, 160, dz)


DZ_MODES = {"r": 1, "fair+lin": 2, "id+lin": 5, "raw_cue": 9}


def fit_and_evaluate(z_np, invests_np, n_train_timesteps, dz, n_epochs=1500, lr=0.03, seed=0,
                      verbose=True, tag=""):
    torch.manual_seed(seed)
    n_subj, T, _ = z_np.shape
    decoder = HierarchicalDecoderCumulativeLink(dz=dz, dq=1, num_categories=5,
                                                 n_subjects=n_subj, decoder_mode="individual")
    z = torch.tensor(z_np, dtype=torch.float32)
    x = torch.tensor(invests_np, dtype=torch.float32)
    subject_ids = torch.arange(n_subj)

    z_train = z[:, :n_train_timesteps]
    x_train = x[:, :n_train_timesteps]

    optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        loss = -decoder.log_likelihood(x_train, z_train, subject_ids=subject_ids)
        loss.backward()
        optimizer.step()
        if verbose and (epoch % max(1, n_epochs // 5) == 0 or epoch == n_epochs - 1):
            print(f"[{tag}] epoch {epoch:4d}  train_nll_sum={loss.item():.2f}")

    with torch.no_grad():
        log_probs, valid_mask = decoder.log_likelihood_per_timestep(x, z, subject_ids=subject_ids)
        probs = decoder.get_category_probabilities(z, subject_ids=subject_ids)
        mode_action = torch.argmax(probs, dim=-1).squeeze(-1) + 1

    x_np = x.squeeze(-1).numpy()
    mode_np = mode_action.numpy().astype(float)
    logp_np = log_probs.numpy()
    valid_np = valid_mask.numpy()

    train_valid = valid_np.copy(); train_valid[:, n_train_timesteps:] = False
    test_valid = valid_np.copy(); test_valid[:, :n_train_timesteps] = False

    def mae(mask):
        return float(np.mean(np.abs(mode_np[mask] - x_np[mask]))) if mask.any() else float("nan")

    def nll(mask):
        return float(-np.mean(logp_np[mask])) if mask.any() else float("nan")

    return dict(train_mae=mae(train_valid), test_mae=mae(test_valid),
                train_nll=nll(train_valid), test_nll=nll(test_valid))


def fit_and_evaluate_hierarchical(z_np, invests_np, n_train_timesteps, dz, feature_dim,
                                   n_epochs=600, lr=0.03, seed=0, verbose=True, tag=""):
    """Same fit as fit_and_evaluate, but with decoder_mode='hierarchical': decoder parameters
    (beta_0, beta) are generated from a learned per-subject feature vector (shape
    (n_subjects, feature_dim)) via a SHARED linear projection, rather than being fully free per
    subject. `feature_dim` controls how much decoder capacity is shared (projection weights, fit
    once) vs. individualised (the low-rank per-subject feature vector) -- matching the AL-RNN's
    own decoder_mode='hierarchical' parameterization (its "P" hyperparameter), so this is the
    apples-to-apples check against the reported AL-RNN number."""
    torch.manual_seed(seed)
    n_subj, T, _ = z_np.shape
    decoder = HierarchicalDecoderCumulativeLink(dz=dz, dq=1, num_categories=5, n_subjects=n_subj,
                                                 decoder_mode="hierarchical", feature_dim=feature_dim)
    subject_features = torch.nn.Parameter(torch.randn(n_subj, feature_dim) * 0.1)

    z = torch.tensor(z_np, dtype=torch.float32)
    x = torch.tensor(invests_np, dtype=torch.float32)

    z_train = z[:, :n_train_timesteps]
    x_train = x[:, :n_train_timesteps]

    optimizer = torch.optim.Adam(list(decoder.parameters()) + [subject_features], lr=lr)
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        loss = -decoder.log_likelihood(x_train, z_train, feature_vectors=subject_features)
        loss.backward()
        optimizer.step()
        if verbose and (epoch % max(1, n_epochs // 3) == 0 or epoch == n_epochs - 1):
            print(f"[{tag}] epoch {epoch:4d}  train_nll_sum={loss.item():.2f}")

    with torch.no_grad():
        log_probs, valid_mask = decoder.log_likelihood_per_timestep(x, z, feature_vectors=subject_features)
        probs = decoder.get_category_probabilities(z, feature_vectors=subject_features)
        mode_action = torch.argmax(probs, dim=-1).squeeze(-1) + 1

    x_np = x.squeeze(-1).numpy()
    mode_np = mode_action.numpy().astype(float)
    logp_np = log_probs.numpy()
    valid_np = valid_mask.numpy()

    train_valid = valid_np.copy(); train_valid[:, n_train_timesteps:] = False
    test_valid = valid_np.copy(); test_valid[:, :n_train_timesteps] = False

    def mae(mask):
        return float(np.mean(np.abs(mode_np[mask] - x_np[mask]))) if mask.any() else float("nan")

    def nll(mask):
        return float(-np.mean(logp_np[mask])) if mask.any() else float("nan")

    n_shared = sum(p.numel() for p in decoder.parameters())
    n_individual = subject_features.numel()
    return dict(train_mae=mae(train_valid), test_mae=mae(test_valid),
                train_nll=nll(train_valid), test_nll=nll(test_valid),
                n_shared_params=n_shared, n_individual_params=n_individual)


FEATURE_DIMS = (1, 2, 4, 8, 16)
DEFAULT_OUT_HIER = Path(__file__).resolve().parent / "results" / "decoder_hierarchical_results.json"


def main():
    subjects, n_train_trials, n_test_trials = load_dataset(str(DEFAULT_INPUTS), str(DEFAULT_INVESTS))
    invests_np = np.load(DEFAULT_INVESTS)   # (n_subj, 160, 1) -- ground truth, with NaNs
    total_T = invests_np.shape[1]
    n_train_timesteps = int(total_T * 0.75)

    results = {}
    for mode, dz in DZ_MODES.items():
        print(f"\n=== decoder-only, static feature = '{mode}' (dz={dz}) ===")
        z_np = build_features(subjects, mode)
        metrics = fit_and_evaluate(z_np, invests_np, n_train_timesteps, dz, tag=mode)
        results[mode] = dict(dz=dz, **metrics)
        print(f"  test_mae={metrics['test_mae']:.4f}  test_nll={metrics['test_nll']:.4f}  "
              f"(train_mae={metrics['train_mae']:.4f}, train_nll={metrics['train_nll']:.4f})")

    print("\n=== Summary (mean test MAE / test NLL) ===")
    for mode, r in results.items():
        print(f"{mode:10s} dz={r['dz']}  test_mae={r['test_mae']:.4f}  test_nll={r['test_nll']:.4f}")

    DEFAULT_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(DEFAULT_OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {DEFAULT_OUT}")
    return results


def main_hierarchical(feature_dims=FEATURE_DIMS, n_epochs=600):
    subjects, n_train_trials, n_test_trials = load_dataset(str(DEFAULT_INPUTS), str(DEFAULT_INVESTS))
    invests_np = np.load(DEFAULT_INVESTS)
    total_T = invests_np.shape[1]
    n_train_timesteps = int(total_T * 0.75)

    results = {}
    for mode, dz in DZ_MODES.items():
        z_np = build_features(subjects, mode)
        results[mode] = {}
        for fd in feature_dims:
            tag = f"{mode}-fd{fd}"
            print(f"\n=== decoder-only (hierarchical), feature='{mode}' (dz={dz}), feature_dim={fd} ===")
            metrics = fit_and_evaluate_hierarchical(z_np, invests_np, n_train_timesteps, dz, fd,
                                                      n_epochs=n_epochs, tag=tag)
            results[mode][fd] = dict(dz=dz, feature_dim=fd, **metrics)
            print(f"  test_mae={metrics['test_mae']:.4f}  test_nll={metrics['test_nll']:.4f}  "
                  f"(train_mae={metrics['train_mae']:.4f}, shared_params={metrics['n_shared_params']}, "
                  f"individual_params={metrics['n_individual_params']})")

    print("\n=== Summary (mean test MAE across dz x feature_dim) ===")
    for mode in DZ_MODES:
        row = "  ".join(f"fd={fd}:{results[mode][fd]['test_mae']:.3f}" for fd in feature_dims)
        print(f"{mode:10s} {row}")

    DEFAULT_OUT_HIER.parent.mkdir(parents=True, exist_ok=True)
    with open(DEFAULT_OUT_HIER, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {DEFAULT_OUT_HIER}")
    return results


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["individual", "hierarchical"], default="hierarchical")
    ap.add_argument("--n-epochs", type=int, default=600)
    args = ap.parse_args()
    if args.mode == "individual":
        main()
    else:
        main_hierarchical(n_epochs=args.n_epochs)
