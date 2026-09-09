from __future__ import annotations

import json
from functools import partial
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

from data import load_dataset
from decoder_variants import (fit_hierarchical_generic, evaluate_decoder, softmax_logprobs,
                               n_params_softmax, _score_nll, _pearson_corr)

"""
bayes_model.py
================
A discrete Bayesian trustee-tracking model in the style of Ray, King-Casas,
Montague & Dayan (2008), as an independent robustness check on the RL +
Fehr-Schmidt result.

Model (see `type_model_note.pdf`)
-----------------------------------
K discrete generosity types with fixed return rates rho_1 < ... < rho_K.
Each of the 4 trustees has its own belief b^(j)_t over types, started
uniform. Predicted return combines the belief mean with an expression-cue
offset c(e_t):

    RRhat_t = sum_k b_t^(j)(k) rho_k + c(e_t)

Choice value and softmax (the model's own "native" decision rule -- pure
risk-neutral expected-value maximisation, no inequity aversion):

    Q_t(a) = a * (3*RRhat_t - 1),      P(a) ~ exp(beta * Q_t(a))

After observing the true repayment ratio RR_t, the shown trustee's belief is
updated by a Bayesian multiply-and-renormalise (log-space additive, no
learning rate):

    b_{t+1}^(j)(k) ~ b_t^(j)(k) * N(RR_t; rho_k + c(e_t), sigma^2)

Expression-cue flexibility (`mode`)
--------------------------------------
c(e_t) is computed as `basis[e_t] @ we`, where `we` is fit and `basis` sets
how much freedom the expression effect has:
  "linear": basis is a single column (3 - expression_index), so c(e_t) is a
    fixed slope `we[0]` times a signed cue in {-2,...,+2} (1 free param;
    matches the PDF's literal w*c(e_t) term, sign-corrected so index 1
    ("happy") = +2, matching the PDF's own "expected w>0" assumption).
  "free": basis is a one-hot-style matrix over the 4 non-neutral expression
    levels {1,2,4,5} (level 3 = neutral, offset always 0), so each level
    gets its own free additive offset (4 free params) -- exactly as
    flexible as the RL model's per-expression-level weight vector `ws2`.
Running both modes through all three structures below gives a capacity
ladder *within* the Bayesian family, directly comparable to the RL family's
own r_t (free-per-level) vs. raw-cue (fully free) comparison.

Structures implemented (skipping the PDF's structure (A); see prior
analysis -- weak rationale, mainly parity with the digital-twin paper):
  (B)  dynamic belief-tracker + native Q-softmax           -- params {we, beta}
  (B') dynamic belief-tracker + flexible softmax decoder    -- params {we, decoder}
  (C)  static, no belief update: RRhat = g_j + c(e_t)       -- params {g_1..g_4, we, beta}

Data-informed hyperparameters (fixed from TRAIN data only, pooled across
subjects; see `compute_grid_and_sigma`): the type grid rho spans the
train-pooled empirical RR range (not an arbitrary [0,1]), and sigma is the
train-pooled residual std of RR after removing trustee-identity and
expression structure (matched to whichever `mode`'s basis is in use),
rather than being fit via the (weak, indirect) choice likelihood.

b0 note: the PDF lists the prior b0 as a fittable per-subject parameter in
its structure (A). Fitting it freely would add K-1 parameters per subject
on top of an already-sparse 60 train trials, so b0 is fixed uniform here.
"""

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = Path(__file__).resolve().parent / "results" / "bayes_results.json"

K_TYPES = 15

# basis[expression_index-1, :] @ we  ->  c(e_t)
EXPR_BASIS = {
    # levels 1..5 -> (3-level): a single signed slope, {-2,...,+2}
    "linear": np.array([[2.0], [1.0], [0.0], [-1.0], [-2.0]]),
    # levels 1..5 -> one-hot over the 4 non-neutral levels {1,2,4,5}; level 3 (neutral) is always 0
    "free": np.array([[1, 0, 0, 0],
                       [0, 1, 0, 0],
                       [0, 0, 0, 0],
                       [0, 0, 1, 0],
                       [0, 0, 0, 1]], dtype=float),
}

WE_BOUNDS = {
    "linear": [(0.0, 5.0)],
    "free": [(-5.0, 5.0)] * 4,
}


def compute_grid_and_sigma(subjects, n_train, basis, K=K_TYPES):
    """Train-pooled (across subjects), causal (train-only) empirical hyperparameters: the type
    grid rho spans the observed RR range, and sigma is the residual std of RR after removing
    trustee-identity + this basis's expression structure (OLS on train data)."""
    stim, emo, rr = [], [], []
    for d in subjects:
        dtr = d[:n_train]
        m = dtr[:, 3] != 0
        stim.append(dtr[m, 0]); emo.append(dtr[m, 1]); rr.append(dtr[m, 2])
    stim = np.concatenate(stim).astype(int)
    emo = np.concatenate(emo).astype(int)
    rr = np.concatenate(rr)

    rho = np.linspace(rr.min(), rr.max(), K)

    Xc = basis[emo - 1]
    X = np.column_stack([np.eye(4)[stim - 1], Xc])
    coef, *_ = np.linalg.lstsq(X, rr, rcond=None)
    resid = rr - X @ coef
    sigma = float(resid.std())
    return rho, sigma


def _softmax_1d(x):
    z = x - x.max()
    ez = np.exp(z)
    return ez / ez.sum()


def causal_RRhat(we, sigma, data, rho, basis):
    """The causal per-trial belief-tracking loop alone, factored out so both the native
    Q-softmax decision rule and any alternative decoder can share the identical belief
    trajectory RRhat_t (analogous to fs_rl_model's predRR)."""
    K = rho.size
    T = data.shape[0]
    log_b = np.zeros((4, K))
    RRhat = np.zeros(T)
    for t in range(T):
        j = int(data[t, 0]) - 1
        c = float(basis[int(data[t, 1]) - 1] @ we)
        a_t = int(data[t, 3])

        b = _softmax_1d(log_b[j])
        RRhat[t] = float(np.sum(b * rho) + c)

        if a_t == 0:
            continue
        RR_obs = data[t, 2]
        ll_k = -0.5 * ((RR_obs - (rho + c)) ** 2) / (sigma ** 2)
        log_b[j] = log_b[j] + ll_k
    return RRhat


# --------------------------------------------------------------------------
# (B) Full dynamic Bayesian belief-tracker + native Q-softmax decision rule
# --------------------------------------------------------------------------
def loglik_dynamic(params, data, rho, sigma, basis, n_train=None, return_diagnostics=False):
    """params = [*we, beta]. `data`: (T,4) [stimtyp, shown_emo, RR_tilde, action]."""
    n_we = basis.shape[1]
    we = params[:n_we]
    beta = params[n_we]
    K = rho.size
    data = np.asarray(data, dtype=float)
    T = data.shape[0]
    at = np.arange(1, 6)

    log_b = np.zeros((4, K))
    RRhat = np.zeros(T)
    prt = np.full(T, np.nan)
    P = np.zeros((T, 5))

    for t in range(T):
        j = int(data[t, 0]) - 1
        c = float(basis[int(data[t, 1]) - 1] @ we)
        a_t = int(data[t, 3])

        b = _softmax_1d(log_b[j])
        rr_hat = float(np.sum(b * rho) + c)
        RRhat[t] = rr_hat

        if a_t == 0:
            prt[t] = 0.2
            continue

        Q = at * (3.0 * rr_hat - 1.0)
        z = beta * (Q - Q.max())
        ez = np.exp(z)
        pr = ez / ez.sum()
        P[t] = pr
        prt[t] = pr[a_t - 1]

        RR_obs = data[t, 2]
        ll_k = -0.5 * ((RR_obs - (rho + c)) ** 2) / (sigma ** 2)
        log_b[j] = log_b[j] + ll_k

    scored = prt[:n_train] if n_train is not None else prt
    if np.any(scored <= 0) or not np.all(np.isfinite(scored)):
        logL = -1e6
    else:
        logL = float(np.sum(np.log(scored)))

    if return_diagnostics:
        return logL, dict(pr_t=prt, predRR=RRhat, P=P)
    return logL


def _neg_loglik_dynamic(params, data, rho, sigma, basis, n_train):
    return -loglik_dynamic(params, data, rho, sigma, basis, n_train=n_train)


def fit_hierarchical_dynamic(subjects, n_train, rho, sigma, basis, we_bounds,
                              n_outer=6, seed=0, verbose=True, tag="bayes-dynamic"):
    """Iterated empirical-Bayes hierarchical fit of the (B) dynamic model."""
    bounds = list(we_bounds) + [(1e-3, 20.0)]
    n_params = len(bounds)
    lb = np.array([b[0] for b in bounds])
    ub = np.array([b[1] for b in bounds])
    pop_mean = (lb + ub) / 2.0
    init_var = ((ub - lb) / 2.0) ** 2
    var_floor = 0.05 * init_var
    pop_var = init_var.copy()

    n_subj = len(subjects)
    thetas = np.tile(pop_mean, (n_subj, 1))
    rng = np.random.default_rng(seed)

    for outer in range(n_outer):
        for i, data in enumerate(subjects):
            if outer == 0:
                x0_list = [pop_mean.copy()] + [rng.uniform(lb, ub) for _ in range(4)]
            else:
                jitter = rng.normal(scale=0.05 * (ub - lb), size=(2, n_params))
                x0_list = [thetas[i]] + [thetas[i] + j for j in jitter]

            best = None
            for x0 in x0_list:
                x0c = np.clip(x0, lb, ub)
                def obj(p, _d=data):
                    nll = _neg_loglik_dynamic(p, _d, rho, sigma, basis, n_train)
                    return nll + 0.5 * np.sum((p - pop_mean) ** 2 / pop_var)
                res = minimize(obj, x0c, method="L-BFGS-B", bounds=bounds, options=dict(maxiter=500))
                if (best is None) or (res.fun < best.fun):
                    best = res
            thetas[i] = best.x

        pop_mean = thetas.mean(axis=0)
        pop_var = np.maximum(thetas.var(axis=0), var_floor)
        if verbose:
            print(f"[{tag}] outer {outer + 1}/{n_outer}: we={np.round(pop_mean[:-1], 3)} "
                  f"beta={pop_mean[-1]:.3f}")

    return thetas


def loglik_dynamic_decoder(params, data, n_decoder_params, rho, sigma, basis, n_train=None):
    """(B) variant with the native linear-Q softmax replaced by a flexible decoder on the same
    causal Bayesian belief RRhat_t -- analogue of the RL model's `softmax_r` decoder-swap test."""
    n_we = basis.shape[1]
    we = params[:n_we]
    decoder_params = params[n_we:n_we + n_decoder_params]
    RRhat = causal_RRhat(we, sigma, data, rho, basis)
    logp_all = softmax_logprobs(decoder_params, RRhat.reshape(-1, 1))
    action = data[:, 3].astype(int)
    return _score_nll(logp_all, action, n_train)


def fit_hierarchical_dynamic_decoder(subjects, n_train, rho, sigma, basis, we_bounds,
                                      n_outer=6, seed=0, verbose=True, tag="bayes-softmax_r"):
    """Same iterated empirical-Bayes scheme, for loglik_dynamic_decoder's
    [*we, *softmax_r params] parameter vector."""
    n_we = basis.shape[1]
    n_dec = n_params_softmax(1)
    n_params = n_we + n_dec
    bounds = list(we_bounds) + [(None, None)] * n_dec
    lb = np.array([b[0] for b in we_bounds] + [-np.inf] * n_dec)
    ub = np.array([b[1] for b in we_bounds] + [np.inf] * n_dec)

    pop_mean_we = np.array([(b[0] + b[1]) / 2.0 for b in we_bounds])
    pop_var_we = np.array([((b[1] - b[0]) / 2.0) ** 2 for b in we_bounds])
    pop_mean = np.concatenate([pop_mean_we, np.zeros(n_dec)])
    pop_var = np.concatenate([pop_var_we, np.full(n_dec, 9.0)])
    var_floor = 0.05 * pop_var

    n_subj = len(subjects)
    thetas = np.tile(pop_mean, (n_subj, 1))
    rng = np.random.default_rng(seed)

    for outer in range(n_outer):
        for i, data in enumerate(subjects):
            if outer == 0:
                x0_list = [pop_mean.copy()] + [pop_mean + rng.normal(scale=0.5, size=n_params) for _ in range(3)]
            else:
                jitter = rng.normal(scale=0.1, size=(2, n_params))
                x0_list = [thetas[i]] + [thetas[i] + j for j in jitter]

            best = None
            for x0 in x0_list:
                x0c = np.clip(x0, lb, ub)
                def obj(p, _d=data):
                    nll = loglik_dynamic_decoder(p, _d, n_dec, rho, sigma, basis, n_train=n_train)
                    return nll + 0.5 * np.sum((p - pop_mean) ** 2 / pop_var)
                res = minimize(obj, x0c, method="L-BFGS-B", bounds=bounds, options=dict(maxiter=500))
                if (best is None) or (res.fun < best.fun):
                    best = res
            thetas[i] = best.x

        pop_mean = thetas.mean(axis=0)
        pop_var = np.maximum(thetas.var(axis=0), var_floor)
        if verbose:
            print(f"[{tag}] outer {outer + 1}/{n_outer}: we={np.round(pop_mean[:n_we], 3)}")

    return thetas, n_dec


def evaluate_dynamic_decoder(params, data, n_train, n_decoder_params, rho, sigma, basis):
    n_we = basis.shape[1]
    we = params[:n_we]
    decoder_params = params[n_we:n_we + n_decoder_params]
    RRhat = causal_RRhat(we, sigma, data, rho, basis)
    z = RRhat.reshape(-1, 1)
    return evaluate_decoder(decoder_params, z, data[:, 3].astype(int), n_train, softmax_logprobs)


def evaluate_dynamic(params, data, n_train, rho, sigma, basis):
    _, diag = loglik_dynamic(params, data, rho, sigma, basis, n_train=None, return_diagnostics=True)
    action_true = data[:, 3].astype(int)
    mode_action = np.argmax(diag["P"], axis=1) + 1
    pr_t = diag["pr_t"]

    valid = action_true != 0
    train_mask = valid.copy(); train_mask[n_train:] = False
    test_mask = valid.copy(); test_mask[:n_train] = False

    def mae(mask):
        return float(np.mean(np.abs(mode_action[mask] - action_true[mask]))) if mask.any() else float("nan")

    def nll(mask):
        p = np.clip(pr_t[mask], 1e-10, 1.0)
        return float(np.mean(-np.log(p))) if mask.any() else float("nan")

    def corr(mask):
        return _pearson_corr(mode_action[mask], action_true[mask]) if mask.any() else float("nan")

    return dict(train_mae=mae(train_mask), test_mae=mae(test_mask),
                train_nll=nll(train_mask), test_nll=nll(test_mask),
                train_corr=corr(train_mask), test_corr=corr(test_mask))


# --------------------------------------------------------------------------
# (C) Static, one generosity estimate per trustee, no belief update
# RRhat = g_j + c(e_t); params = [g1,g2,g3,g4, *we, beta]; reuses the generic
# decoder-fitting infra from decoder_variants.py (no sequential state).
# --------------------------------------------------------------------------
def ray_static_logprobs(params, z, n_we):
    g = params[:4]
    we = params[4:4 + n_we]
    beta = params[4 + n_we]
    rr_hat = z[:, :4] @ g + z[:, 4:4 + n_we] @ we           # (T,)
    at = np.arange(1, 6)
    Q = (3.0 * rr_hat[:, None] - 1.0) * at[None, :]  # (T,5)
    logits = beta * Q
    return logits - logsumexp(logits, axis=1, keepdims=True)


def _static_features(data, basis):
    stim = data[:, 0].astype(int)
    emo = data[:, 1].astype(int)
    onehot = np.eye(4)[stim - 1]
    expr = basis[emo - 1]
    return np.concatenate([onehot, expr], axis=1)


def main():
    subjects, n_train, n_test = load_dataset()
    actions = [d[:, 3].astype(int) for d in subjects]

    results = {}
    for mode in ["linear", "free"]:
        basis = EXPR_BASIS[mode]
        n_we = basis.shape[1]
        we_bounds = WE_BOUNDS[mode]
        rho, sigma = compute_grid_and_sigma(subjects, n_train, basis)

        print(f"\n########## mode={mode}  (n_we={n_we}) ##########")
        print(f"rho in [{rho.min():.4f}, {rho.max():.4f}]  K={rho.size}  sigma={sigma:.4f}")

        print(f"=== (B) dynamic + native Q-softmax, mode={mode} ===")
        thetasB = fit_hierarchical_dynamic(subjects, n_train, rho, sigma, basis, we_bounds, tag=f"B-{mode}")
        metricsB = [dict(subject_id=i, **evaluate_dynamic(thetasB[i], subjects[i], n_train, rho, sigma, basis))
                    for i in range(len(subjects))]

        print(f"\n=== (B') dynamic + flexible softmax decoder, mode={mode} ===")
        thetasBp, n_dec = fit_hierarchical_dynamic_decoder(subjects, n_train, rho, sigma, basis, we_bounds,
                                                             tag=f"Bp-{mode}")
        metricsBp = [dict(subject_id=i,
                           **evaluate_dynamic_decoder(thetasBp[i], subjects[i], n_train, n_dec, rho, sigma, basis))
                     for i in range(len(subjects))]

        print(f"\n=== (C) static per-trustee generosity + expression term, mode={mode} ===")
        z_static = [_static_features(d, basis) for d in subjects]
        logprobs_fn_C = partial(ray_static_logprobs, n_we=n_we)
        n_params_C = 4 + n_we + 1
        thetasC = fit_hierarchical_generic(z_static, actions, n_train, n_params_C, logprobs_fn_C, tag=f"C-{mode}")
        metricsC = [dict(subject_id=i,
                          **evaluate_decoder(thetasC[i], z_static[i], actions[i], n_train, logprobs_fn_C))
                    for i in range(len(subjects))]

        def summarize(metrics):
            return dict(
                mean_train_mae=float(np.nanmean([m["train_mae"] for m in metrics])),
                mean_test_mae=float(np.nanmean([m["test_mae"] for m in metrics])),
                mean_train_nll=float(np.nanmean([m["train_nll"] for m in metrics])),
                mean_test_nll=float(np.nanmean([m["test_nll"] for m in metrics])),
                mean_train_corr=float(np.nanmean([m["train_corr"] for m in metrics])),
                mean_test_corr=float(np.nanmean([m["test_corr"] for m in metrics])),
                per_subject=metrics,
            )

        results[mode] = {
            "hyperparameters": dict(rho_min=float(rho.min()), rho_max=float(rho.max()),
                                     K=int(rho.size), sigma=sigma, n_we=n_we),
            "dynamic_bayes": summarize(metricsB),
            "dynamic_bayes_softmax_r": summarize(metricsBp),
            "static_ray": summarize(metricsC),
        }

        print(f"\n--- mode={mode} summary (test MAE / test NLL / test corr) ---")
        for name in ["dynamic_bayes", "dynamic_bayes_softmax_r", "static_ray"]:
            r = results[mode][name]
            print(f"{name:25s} test_mae={r['mean_test_mae']:.4f}  test_nll={r['mean_test_nll']:.4f}  "
                  f"test_corr={r['mean_test_corr']:.4f}  "
                  f"(train_mae={r['mean_train_mae']:.4f}, train_nll={r['mean_train_nll']:.4f})")

    DEFAULT_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(DEFAULT_OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {DEFAULT_OUT}")
    return results


if __name__ == "__main__":
    main()
