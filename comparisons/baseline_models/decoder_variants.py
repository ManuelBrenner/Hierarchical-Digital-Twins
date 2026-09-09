from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

from fs_rl_model import loglik as fs_loglik
from data import load_dataset

"""
decoder_variants.py
=====================
Isolates whether the RL model's weak predictive performance comes from its
*value learning* (the online repayment-rate belief r_t) or from its
*decision rule* -- the Fehr-Schmidt utility + softmax mapping from that
belief to a chosen investment.

The count-based-learning-rate diagnostic (fit_evaluate.py --count-based-lr)
already showed that faster/near-optimal-for-the-trial-order convergence of
r_t doesn't close the gap to a plain cue -> investment regression. This
script holds r_t fixed (computed causally via the same count-based update)
and swaps only the observation model on top of it, mirroring the
ordinal-vs-softmax decoder split already used for the AL-RNN (decoders.py):

    A) fs_softmax  : Fehr-Schmidt utility + softmax over actions (baseline;
                     the mechanistic, decision-theoretic decoder -- see
                     fs_rl_model.py / fit_evaluate.py)
    B) softmax_r   : free per-category (weight, bias) softmax regression on
                     the scalar belief r_t alone -- same value information
                     as (A), no assumed utility form
    C) ordinal_r   : monotonic ordinal cumulative-link model on r_t -- same
                     value information as (A)/(B), respects the ordinal
                     structure of investment levels 1..5 without a
                     money-metric utility
    D) softmax_cue : free per-category (weight-vector, bias) softmax
                     regression directly on the 9-dim cue one-hot, bypassing
                     the learned belief r_t entirely -- closest causal
                     analogue to a plain cue -> investment regression, given
                     here as an upper-bound reference point

All decoders are fit per subject with the same iterated empirical-Bayes
hierarchical shrinkage used in fit_evaluate.py (a Gaussian population prior
whose mean/variance is re-estimated across subjects each outer iteration),
and scored with the identical causal train(60)/test(20)-trial split and
mode-prediction MAE / predictive NLL convention.
"""

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUTS = REPO_ROOT / "data" / "stacked_inputs.npy"
DEFAULT_INVESTS = REPO_ROOT / "data" / "stacked_invests.npy"
DEFAULT_OUT = Path(__file__).resolve().parent / "results" / "decoder_variants_results.json"

# dummy RL params: alpha/kappa/FS terms don't affect predRR under count_based_lr=True
_DUMMY_RL_PARS = np.array([0.1, 0.5] + [0.0] * 4 + [0.0] * 4 + [0.0])


def compute_features(data):
    """Causal r_t (via the count-based RL update) and the raw 9-dim cue one-hot, per trial."""
    _, diag = fs_loglik(_DUMMY_RL_PARS, data, fsmap=(1, 2, 3, 4),
                         return_diagnostics=True, count_based_lr=True)
    r_t = diag["predRR"].reshape(-1, 1)

    stim = data[:, 0].astype(int)
    emo = data[:, 1].astype(int)
    onehot = np.concatenate([np.eye(4)[stim - 1], np.eye(5)[emo - 1]], axis=1)  # (T, 9)
    return r_t, onehot


# --------------------------------------------------------------------------
# Decoders: each maps params + per-trial features z (T, D) -> (T, 5) log-probs
# --------------------------------------------------------------------------
def softmax_logprobs(params, z):
    """Free per-category softmax regression: category 1 is the reference (logit fixed at 0)."""
    T, D = z.shape
    P = params.reshape(4, D + 1)
    logits_rest = z @ P[:, :D].T + P[:, D]           # (T, 4)
    logits = np.concatenate([np.zeros((T, 1)), logits_rest], axis=1)  # (T, 5)
    return logits - logsumexp(logits, axis=1, keepdims=True)


def n_params_softmax(D):
    return 4 * (D + 1)


def ordinal_logprobs(params, z):
    """Monotonic ordinal cumulative-link model, single linear index eta = z @ w."""
    T, D = z.shape
    w = params[:D]
    tau = np.empty(4)
    tau[0] = params[D]
    tau[1:] = tau[0] + np.cumsum(np.exp(params[D + 1:D + 4]))
    eta = z @ w                                       # (T,)
    cum = 1.0 / (1.0 + np.exp(-(tau[None, :] - eta[:, None])))   # (T,4) P(Y<=k)
    cum_full = np.concatenate([cum, np.ones((T, 1))], axis=1)     # (T,5)
    prev = np.concatenate([np.zeros((T, 1)), cum_full[:, :-1]], axis=1)
    probs = np.clip(cum_full - prev, 1e-10, 1.0)
    return np.log(probs)


def n_params_ordinal(D):
    return D + 4


def _score_nll(logp_all, action, n_train):
    T = logp_all.shape[0]
    valid = action != 0
    scored = valid.copy()
    if n_train is not None:
        scored[n_train:] = False
    if not scored.any():
        return 1e6
    idx = np.clip(action - 1, 0, 4)
    ll = logp_all[np.arange(T), idx][scored]
    nll = -np.sum(ll)
    return nll if np.isfinite(nll) else 1e6


def neg_loglik(params, logprobs_fn, z, action, n_train=None):
    return _score_nll(logprobs_fn(params, z), action, n_train)


# --------------------------------------------------------------------------
# Generic iterated empirical-Bayes hierarchical fit (unconstrained params)
# --------------------------------------------------------------------------
def fit_hierarchical_generic(z_list, action_list, n_train, n_params, logprobs_fn,
                              n_outer=6, seed=0, prior_scale=3.0, verbose=True, tag=""):
    pop_mean = np.zeros(n_params)
    init_var = np.full(n_params, prior_scale ** 2)
    var_floor = 0.05 * init_var
    pop_var = init_var.copy()

    n_subj = len(z_list)
    thetas = np.zeros((n_subj, n_params))
    rng = np.random.default_rng(seed)

    for outer in range(n_outer):
        for i in range(n_subj):
            if outer == 0:
                x0_list = [np.zeros(n_params)] + [rng.normal(scale=0.5, size=n_params) for _ in range(3)]
            else:
                jitter = rng.normal(scale=0.1, size=(2, n_params))
                x0_list = [thetas[i]] + [thetas[i] + j for j in jitter]

            best = None
            for x0 in x0_list:
                def obj(p, _z=z_list[i], _a=action_list[i]):
                    nll = neg_loglik(p, logprobs_fn, _z, _a, n_train)
                    return nll + 0.5 * np.sum((p - pop_mean) ** 2 / pop_var)
                res = minimize(obj, x0, method="L-BFGS-B", options=dict(maxiter=500))
                if (best is None) or (res.fun < best.fun):
                    best = res
            thetas[i] = best.x

        pop_mean = thetas.mean(axis=0)
        pop_var = np.maximum(thetas.var(axis=0), var_floor)
        if verbose:
            print(f"[{tag}] outer {outer + 1}/{n_outer}: |pop_mean|={np.linalg.norm(pop_mean):.3f}  "
                  f"mean(pop_var)={pop_var.mean():.3f}")

    return thetas


def _pearson_corr(a, b):
    """Pearson r, matching test_model.py's evaluate_model_trajectories convention (NaN if either
    side is constant)."""
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def evaluate_decoder(params, z, action, n_train, logprobs_fn):
    logp_all = logprobs_fn(params, z)
    mode_action = np.argmax(logp_all, axis=1) + 1
    T = logp_all.shape[0]
    idx = np.clip(action - 1, 0, 4)
    chosen_logp = logp_all[np.arange(T), idx]

    valid = action != 0
    train_mask = valid.copy(); train_mask[n_train:] = False
    test_mask = valid.copy(); test_mask[:n_train] = False

    def mae(mask):
        return float(np.mean(np.abs(mode_action[mask] - action[mask]))) if mask.any() else float("nan")

    def nll(mask):
        return float(np.mean(-chosen_logp[mask])) if mask.any() else float("nan")

    def corr(mask):
        return _pearson_corr(mode_action[mask], action[mask]) if mask.any() else float("nan")

    return dict(train_mae=mae(train_mask), test_mae=mae(test_mask),
                train_nll=nll(train_mask), test_nll=nll(test_mask),
                train_corr=corr(train_mask), test_corr=corr(test_mask))


def summarize(metrics_list):
    return dict(
        mean_train_mae=float(np.nanmean([m["train_mae"] for m in metrics_list])),
        mean_test_mae=float(np.nanmean([m["test_mae"] for m in metrics_list])),
        mean_train_nll=float(np.nanmean([m["train_nll"] for m in metrics_list])),
        mean_test_nll=float(np.nanmean([m["test_nll"] for m in metrics_list])),
        mean_train_corr=float(np.nanmean([m["train_corr"] for m in metrics_list])),
        mean_test_corr=float(np.nanmean([m["test_corr"] for m in metrics_list])),
        per_subject=metrics_list,
    )


def main():
    subjects, n_train, n_test = load_dataset(str(DEFAULT_INPUTS), str(DEFAULT_INVESTS))
    z_r, z_cue, actions = [], [], []
    for data in subjects:
        r_t, onehot = compute_features(data)
        z_r.append(r_t)
        z_cue.append(onehot)
        actions.append(data[:, 3].astype(int))

    results = {}

    print("=== B) softmax_r: free softmax regression on scalar belief r_t ===")
    nB = n_params_softmax(1)
    thetasB = fit_hierarchical_generic(z_r, actions, n_train, nB, softmax_logprobs, tag="softmax_r")
    metricsB = [dict(subject_id=i, **evaluate_decoder(thetasB[i], z_r[i], actions[i], n_train, softmax_logprobs))
                for i in range(len(subjects))]
    results["softmax_r"] = summarize(metricsB)

    print("\n=== C) ordinal_r: ordinal cumulative-link model on scalar belief r_t ===")
    nC = n_params_ordinal(1)
    thetasC = fit_hierarchical_generic(z_r, actions, n_train, nC, ordinal_logprobs, tag="ordinal_r")
    metricsC = [dict(subject_id=i, **evaluate_decoder(thetasC[i], z_r[i], actions[i], n_train, ordinal_logprobs))
                for i in range(len(subjects))]
    results["ordinal_r"] = summarize(metricsC)

    print("\n=== D) softmax_cue: free softmax regression directly on the 9-dim cue one-hot ===")
    nD = n_params_softmax(9)
    thetasD = fit_hierarchical_generic(z_cue, actions, n_train, nD, softmax_logprobs, tag="softmax_cue")
    metricsD = [dict(subject_id=i, **evaluate_decoder(thetasD[i], z_cue[i], actions[i], n_train, softmax_logprobs))
                for i in range(len(subjects))]
    results["softmax_cue"] = summarize(metricsD)

    print("\n=== Summary (mean test MAE / test NLL / test corr across 32 subjects) ===")
    for name, r in results.items():
        print(f"{name:14s}  test_mae={r['mean_test_mae']:.4f}  test_nll={r['mean_test_nll']:.4f}  "
              f"test_corr={r['mean_test_corr']:.4f}  "
              f"(train_mae={r['mean_train_mae']:.4f}, train_nll={r['mean_train_nll']:.4f})")

    DEFAULT_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(DEFAULT_OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {DEFAULT_OUT}")
    return results


if __name__ == "__main__":
    main()
