from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from fs_rl_model import loglik, neg_loglik
from data import load_dataset

"""
fit_evaluate.py
================
Fit the RL + Fehr-Schmidt trust-game model (see `fs_rl_model.py` /
`fehr description.pdf`) per subject on the train-period trials of
stacked_invests.npy / stacked_inputs.npy, then evaluate held-out
performance (MAE and predictive NLL) on the test-period trials, using
exactly the same temporal train/test split as `dataset.SimplifiedDataset`
(train_split=0.75: first 60 of 80 trials train, last 20 test).

The model is fit by maximum likelihood *only* on train-trial choices, but
its RL weights are updated causally across the *entire* 80-trial sequence
(train followed by test) -- the repayment feedback that drives learning is
observed input at every trial, train or test, not something being predicted,
so letting it continue updating the weights into the test period is not a
leakage of the test *choices*, and mirrors how a recurrent model would just
keep running forward through the sequence.

The model additionally includes a sticky-choice term (kappa, see
`fs_rl_model.py`) capturing bias toward repeating the previous investment,
independent of learned value.

By default subjects are fit hierarchically: an iterated empirical-Bayes
scheme alternates (1) per-subject MAP fits under a Gaussian population prior
N(pop_mean, pop_var) over the parameters and (2) re-estimating pop_mean/
pop_var as the across-subject mean/variance of the current per-subject
estimates. This pools statistical strength across the 32 subjects -- with
only ~60 train trials and, for the emotional trustees, ~5 trials per
identity x expression cell, independent per-subject MLE is noisy; shrinkage
toward the population regularises it, loosely analogous to how the AL-RNN
pools information across subjects. Pass --flat to fall back to independent
per-subject MLE (no pooling).

Usage
-----
    python fit_evaluate.py                      # hierarchical (default)
    python fit_evaluate.py --flat               # independent per-subject MLE
    python fit_evaluate.py --fsmap 1,2,1,2      # pool inequity weights by fairness
    python fit_evaluate.py --fsmap 1,1,1,1      # canonical two-parameter Fehr-Schmidt
    python fit_evaluate.py --subjects 0,1,2     # fit only a few subjects (debugging)
"""

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUTS = REPO_ROOT / "data" / "stacked_inputs.npy"
DEFAULT_INVESTS = REPO_ROOT / "data" / "stacked_invests.npy"
DEFAULT_OUT = Path(__file__).resolve().parent / "results" / "fs_rl_results.json"


# --------------------------------------------------------------------------
# Optimisation (multi-start constrained MLE, per subject)
# --------------------------------------------------------------------------
def make_bounds_and_constraint(K):
    """Box bounds + Fehr-Schmidt coupling betaFS_k <= alphaFS_k for all k."""
    #        alpha beta        alphaFS(K)         betaFS(K)          kappa
    lb = [0.0, 1e-3] + [0.0] * K + [0.0] * K + [-5.0]  # beta > 0: avoids 1/beta singularity in the softmax
    ub = [1.0, 1.0] + [10.0] * K + [1.0] * K + [5.0]
    bounds = list(zip(lb, ub))

    def coupling(x):  # SLSQP inequality: fun(x) >= 0  ->  alphaFS_k - betaFS_k >= 0
        return np.asarray(x[2:2 + K]) - np.asarray(x[2 + K:2 + 2 * K])
    constraint = {"type": "ineq", "fun": coupling}
    return bounds, constraint, np.array(lb), np.array(ub)


def initial_conditions(K, seed=0):
    """A handful of Fehr-Schmidt-consistent starting points (0 <= betaFS <= alphaFS)."""
    fixed = [
        [0.30, 0.50] + [0.40] * K + [0.15] * K + [0.0],
        [0.50, 0.50] + [0.20] * K + [0.05] * K + [0.0],
        [0.10, 0.30] + [0.80] * K + [0.40] * K + [0.0],
        [0.50, 0.50] + [0.00] * K + [0.00] * K + [0.0],
    ]
    rng = np.random.default_rng(seed)
    rand = []
    for _ in range(4):
        a = rng.uniform(0, 1)
        b = rng.uniform(0, 1)
        aFS = rng.uniform(0, 2, K)
        bFS = np.array([rng.uniform(0, min(1.0, aFS[k])) for k in range(K)])
        kappa0 = rng.uniform(-1.0, 1.0)
        rand.append([a, b] + list(aFS) + list(bFS) + [kappa0])
    return [np.array(x) for x in fixed + rand]


def param_names(K):
    names = ["alpha", "beta"]
    names += [f"alphaFS{k+1}" for k in range(K)]
    names += [f"betaFS{k+1}" for k in range(K)]
    names += ["kappa"]
    return names


def fit_subject(data, n_train, fsmap=(1, 2, 3, 4), seed=0, count_based_lr=False):
    """Multi-start constrained MLE for one subject, scored on train trials only (no pooling)."""
    K = int(np.max(fsmap))
    bounds, constraint, lb, ub = make_bounds_and_constraint(K)
    x0s = initial_conditions(K, seed=seed)

    best = None
    for x0 in x0s:
        x0 = np.clip(x0, lb, ub)
        res = minimize(neg_loglik, x0, args=(data, tuple(fsmap), n_train, count_based_lr),
                        method="SLSQP", bounds=bounds, constraints=[constraint],
                        options=dict(maxiter=1000, ftol=1e-8))
        if (best is None) or (res.fun < best.fun):
            best = res
    return best.x, bool(best.success)


def _map_objective(pars, data, fsmap, n_train, pop_mean, pop_var, count_based_lr):
    """Negative log train-likelihood + Gaussian population-prior penalty (MAP objective)."""
    nll = neg_loglik(pars, data, fsmap=fsmap, n_train=n_train, count_based_lr=count_based_lr)
    prior_nll = 0.5 * np.sum((pars - pop_mean) ** 2 / pop_var)
    return nll + prior_nll


def fit_hierarchical(subjects, n_train, fsmap=(1, 2, 3, 4), n_outer=6, seed=0, verbose=True,
                      count_based_lr=False):
    """
    Iterated empirical-Bayes hierarchical fit across subjects.

    Alternates:
      (1) E-like step: for each subject, MAP-fit params maximising
          train log-likelihood + log N(params; pop_mean, pop_var);
      (2) M-like step: re-estimate pop_mean/pop_var as the across-subject
          mean/variance of the current per-subject MAP estimates.

    This pools statistical strength across subjects (shrinking noisy
    individual fits toward the population) without a full MCMC/Bayesian
    implementation -- the standard "empirical priors" approach used for
    hierarchical MLE fitting of RL models (e.g. Huys et al., 2011).

    Returns
    -------
    thetas : (n_subjects, n_params) fitted per-subject parameters
    successes : (n_subjects,) bool, optimiser convergence on the final iteration
    pop_mean, pop_var : (n_params,) final population hyperparameters
    """
    K = int(np.max(fsmap))
    n_params = 3 + 2 * K
    bounds, constraint, lb, ub = make_bounds_and_constraint(K)

    pop_mean = (lb + ub) / 2.0
    init_var = ((ub - lb) / 2.0) ** 2
    var_floor = 0.05 * init_var
    pop_var = init_var.copy()

    n_subj = len(subjects)
    thetas = np.tile(pop_mean, (n_subj, 1))
    successes = np.zeros(n_subj, dtype=bool)

    rng = np.random.default_rng(seed)
    for outer in range(n_outer):
        for i, data in enumerate(subjects):
            if outer == 0:
                x0_list = initial_conditions(K, seed=seed + i)
            else:
                jitter = rng.normal(scale=0.05 * (ub - lb), size=(2, n_params))
                x0_list = [thetas[i]] + [thetas[i] + j for j in jitter]

            best = None
            for x0 in x0_list:
                x0c = np.clip(x0, lb, ub)
                res = minimize(_map_objective, x0c,
                                args=(data, tuple(fsmap), n_train, pop_mean, pop_var, count_based_lr),
                                method="SLSQP", bounds=bounds, constraints=[constraint],
                                options=dict(maxiter=500, ftol=1e-8))
                if (best is None) or (res.fun < best.fun):
                    best = res
            thetas[i] = best.x
            successes[i] = bool(best.success)

        pop_mean = thetas.mean(axis=0)
        pop_var = np.maximum(thetas.var(axis=0), var_floor)
        if verbose:
            names = param_names(K)
            means_str = ", ".join(f"{n}={m:.3f}" for n, m in zip(names, pop_mean))
            print(f"[hierarchical] outer iter {outer + 1}/{n_outer}: pop mean -> {means_str}")

    return thetas, successes, pop_mean, pop_var


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------
def _pearson_corr(a, b):
    """Pearson r, matching test_model.py's evaluate_model_trajectories convention (NaN if either
    side is constant, e.g. too few valid points or a degenerate all-same-category prediction)."""
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def evaluate_subject(params, data, n_train, fsmap=(1, 2, 3, 4), count_based_lr=False):
    """
    Run the fitted model causally across the full (train+test) trial
    sequence and score mode-prediction MAE, predictive NLL, and mode-vs-true
    Pearson correlation separately on the train and test trial ranges.
    Omission trials (action == 0) are excluded, matching how the AL-RNN
    comparison masks out NaN targets.
    """
    _, diag = loglik(params, data, fsmap=fsmap, return_diagnostics=True, count_based_lr=count_based_lr)
    action_true = data[:, 3].astype(int)
    n_trials = data.shape[0]

    mode_action = np.argmax(diag["P"], axis=1) + 1  # (T,) categories 1..5
    pr_t = diag["pr_t"]

    valid = action_true != 0
    train_mask = valid.copy(); train_mask[n_train:] = False
    test_mask = valid.copy(); test_mask[:n_train] = False

    def _mae(mask):
        if not mask.any():
            return float("nan")
        return float(np.mean(np.abs(mode_action[mask] - action_true[mask])))

    def _nll(mask):
        if not mask.any():
            return float("nan")
        p = np.clip(pr_t[mask], 1e-10, 1.0)
        return float(np.mean(-np.log(p)))

    def _corr(mask):
        if not mask.any():
            return float("nan")
        return _pearson_corr(mode_action[mask], action_true[mask])

    return dict(
        n_trials=n_trials,
        n_train_trials=n_train,
        n_test_trials=n_trials - n_train,
        train_mae=_mae(train_mask),
        test_mae=_mae(test_mask),
        train_nll=_nll(train_mask),
        test_nll=_nll(test_mask),
        train_corr=_corr(train_mask),
        test_corr=_corr(test_mask),
    )


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def _parse_fsmap(s):
    vals = tuple(int(x) for x in s.split(","))
    if len(vals) != 4:
        raise argparse.ArgumentTypeError("fsmap must be 4 comma-separated integers, e.g. 1,2,3,4")
    return vals


def _parse_subjects(s):
    return [int(x) for x in s.split(",")]


def main():
    ap = argparse.ArgumentParser(
        description="Fit & evaluate the RL + Fehr-Schmidt trust-game model on stacked_invests/inputs.npy.")
    ap.add_argument("--inputs", default=str(DEFAULT_INPUTS), help="path to stacked_inputs.npy")
    ap.add_argument("--invests", default=str(DEFAULT_INVESTS), help="path to stacked_invests.npy")
    ap.add_argument("--fsmap", type=_parse_fsmap, default=(1, 2, 3, 4),
                     help="trustee->FS-slot map: 1,2,3,4 (per trustee) | 1,2,1,2 (fair/unfair) | 1,1,1,1 (global)")
    ap.add_argument("--train-split", type=float, default=0.75,
                     help="fraction of timesteps used for train (must match dataset.SimplifiedDataset)")
    ap.add_argument("--subjects", type=_parse_subjects, default=None,
                     help="comma-separated subject indices to run (default: all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--flat", action="store_true",
                     help="fit each subject independently by plain MLE, no population pooling")
    ap.add_argument("--n-outer", type=int, default=6,
                     help="hierarchical fit only: number of empirical-Bayes outer iterations")
    ap.add_argument("--count-based-lr", action="store_true",
                     help="replace the fixed-alpha delta rule with a per-feature 1/visit-count "
                          "learning rate (causal running mean); alpha is then unused")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="output JSON path")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    subjects, n_train, n_test = load_dataset(args.inputs, args.invests, train_split=args.train_split)
    subj_indices = args.subjects if args.subjects is not None else list(range(len(subjects)))
    names = param_names(int(max(args.fsmap)))
    fit_subjects = [subjects[s] for s in subj_indices]

    pop_info = None
    if args.flat:
        thetas, successes = [], []
        for data in fit_subjects:
            params, success = fit_subject(data, n_train, fsmap=args.fsmap, seed=args.seed,
                                           count_based_lr=args.count_based_lr)
            thetas.append(params)
            successes.append(success)
    else:
        thetas, successes, pop_mean, pop_var = fit_hierarchical(
            fit_subjects, n_train, fsmap=args.fsmap, n_outer=args.n_outer, seed=args.seed,
            verbose=not args.quiet, count_based_lr=args.count_based_lr)
        pop_info = dict(pop_mean=dict(zip(names, pop_mean.tolist())),
                         pop_std=dict(zip(names, np.sqrt(pop_var).tolist())))
        successes = list(successes)

    per_subject = []
    for s, params, success in zip(subj_indices, thetas, successes):
        data = subjects[s]
        params = np.asarray(params)
        train_logL = loglik(params, data, fsmap=args.fsmap, n_train=n_train, count_based_lr=args.count_based_lr)
        metrics = evaluate_subject(params, data, n_train, fsmap=args.fsmap, count_based_lr=args.count_based_lr)
        row = dict(subject_id=s, params=dict(zip(names, params.tolist())),
                   train_logL=train_logL, success=bool(success), **metrics)
        per_subject.append(row)
        if not args.quiet:
            print(f"subject {s:2d}  train_mae={row['train_mae']:.3f}  test_mae={row['test_mae']:.3f}  "
                  f"train_nll={row['train_nll']:.3f}  test_nll={row['test_nll']:.3f}  "
                  f"({'ok' if success else 'FAILED'})")

    summary = dict(
        fit_mode="flat" if args.flat else "hierarchical",
        count_based_lr=args.count_based_lr,
        n_subjects=len(per_subject),
        n_train_trials=n_train,
        n_test_trials=n_test,
        fsmap=list(args.fsmap),
        mean_train_mae=float(np.nanmean([r["train_mae"] for r in per_subject])),
        mean_test_mae=float(np.nanmean([r["test_mae"] for r in per_subject])),
        mean_train_nll=float(np.nanmean([r["train_nll"] for r in per_subject])),
        mean_test_nll=float(np.nanmean([r["test_nll"] for r in per_subject])),
        mean_train_corr=float(np.nanmean([r["train_corr"] for r in per_subject])),
        mean_test_corr=float(np.nanmean([r["test_corr"] for r in per_subject])),
        population=pop_info,
        per_subject=per_subject,
    )

    print(f"\n=== RL + Fehr-Schmidt baseline ({summary['fit_mode']}) ===")
    print(f"subjects: {summary['n_subjects']}   train trials/subj: {n_train}   test trials/subj: {n_test}")
    print(f"mean train MAE: {summary['mean_train_mae']:.4f}   mean test MAE: {summary['mean_test_mae']:.4f}")
    print(f"mean test corr: {summary['mean_test_corr']:.4f}")
    print(f"mean train NLL: {summary['mean_train_nll']:.4f}   mean test NLL: {summary['mean_test_nll']:.4f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved results to {out_path}")
    return summary


if __name__ == "__main__":
    main()
