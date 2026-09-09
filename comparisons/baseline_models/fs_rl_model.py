from __future__ import annotations
import numpy as np

"""
fs_rl_model.py
================
Reinforcement-learning trust-game model with a Fehr-Schmidt inequity-averse
utility, as described in `fehr description.pdf`:

    * plain RL updating of the repayment-ratio weights (single learning rate),
      weights initialised at ZERO  -> no cue priors (no omega, no d-omega);
    * a pure Fehr-Schmidt utility  -> no middle-option (L2) bias term.

Model
-----
On each trial the predicted repayment ratio for the presented identity/cue is
a linear read-out of a learned weight vector:

    r_t = w_t . phi_t                                              (eqn 1)

The investor's own and the trustee's predicted gains for an investment a
(a in {1,...,5}, i.e. 10-50 units in tens) are

    G_i(a) = (5 - a) + 3*a*r_t                                      (eqn 2)
    G_j(a) = 3*a - 3*a*r_t                                          (eqn 3)

and the (Fehr & Schmidt, 1999) inequity-averse utility of action a is

    U_t(a) = G_i(a)
             - alphaFS_j * max(G_j(a) - G_i(a), 0)     # disadvantageous (envy)
             - betaFS_j  * max(G_i(a) - G_j(a), 0)     # advantageous   (guilt)
                                                                      (eqn 4)

Actions are coupled to utility by a softmax (eqn 5), and the weights are
updated by a reward-prediction error after observing the actual repayment
ratio (eqn 6):

    w_{t+1} = w_t + alpha * (r_tilde_t - r_t) * phi_t

Sticky choice (extension beyond the PDF)
-----------------------------------------
To capture choice-history effects that are orthogonal to learned value (e.g.
anchoring/perseveration on the previous investment), a stickiness parameter
kappa adds a flat bonus to the utility of whichever action was actually
chosen on the previous non-omitted trial, before the softmax:

    U_t(a) <- U_t(a) + kappa * 1[a == a_{t-1}]

kappa > 0 encodes perseveration (repeat bias), kappa < 0 alternation bias.

Count-based learning rate (diagnostic variant)
------------------------------------------------
`loglik(..., count_based_lr=True)` replaces the fixed-`alpha` delta rule with
a per-feature visit-count step size: the k-th time an identity/expression
feature is seen, its weight is updated with step size 1/k instead of the
constant `alpha`. This turns each cue weight into a (causal, trial-by-trial)
running mean of its prediction errors rather than a fixed-rate exponential
trace, so it should converge close to what a batch regression of repayment
on cue dummies would give, while still never using future information. Under
this mode `alpha` (pars[0]) is present in `pars` for interface compatibility
but unused.

Parameter vector `pars` (length 3 + 2*K, with K = number of FS slots)
-----------------------------------------------------------------------
    pars[0]            alpha    : RL learning rate
    pars[1]            beta     : softmax inverse-temperature (exploration)
    pars[2 : 2+K]      alphaFS  : Fehr-Schmidt ENVY  weights (disadvantageous ineq.)
    pars[2+K : 2+2K]   betaFS   : Fehr-Schmidt GUILT weights (advantageous  ineq.)
    pars[2+2K]         kappa    : sticky-choice weight (previous-action bonus)

`fsmap` maps the four trustees (1=fair-emo, 2=unfair-emo, 3=fair-neut,
4=unfair-neut) to the K FS parameter slots:
    (1,2,3,4) -> one (alphaFS,betaFS) pair per trustee     (K=4, default)
    (1,2,1,2) -> pooled by fairness (fair vs. unfair)      (K=2)
    (1,1,1,1) -> single global pair (canonical Fehr-Schmidt)(K=1)

Input trial array
------------------
`data` is a (T, 4) array with columns [stimtyp, shown_emo, repayment_ratio,
action]: stimtyp in {1,2,3,4}, shown_emo in {1..5}, repayment_ratio in
[0,1], action (investment level) in {1..5}, or action == 0 for an omitted
(non-)decision, which is treated as chance (p = 1/5) and skipped for
learning.
"""


def unpack(pars, fsmap=(1, 2, 3, 4)):
    fsmap = np.asarray(fsmap, dtype=int)
    K = int(fsmap.max())
    pars = np.asarray(pars, dtype=float)
    if pars.size != 3 + 2 * K:
        raise ValueError(f"Expected {3 + 2*K} parameters for K={K} FS slots, got {pars.size}.")
    alpha = pars[0]
    beta = pars[1]
    alphaFS = pars[2:2 + K]
    betaFS = pars[2 + K:2 + 2 * K]
    kappa = pars[2 + 2 * K]
    return alpha, beta, alphaFS, betaFS, kappa, fsmap, K


def loglik(pars, data, fsmap=(1, 2, 3, 4), return_diagnostics=False,
           n_train=None, count_based_lr=False):
    """
    Log-likelihood of the simplified RL + Fehr-Schmidt model.

    Parameters
    ----------
    pars : array-like, length 2 + 2*K
    data : (T, 4) array, columns [stimtyp, shown_emo, repayment_ratio, action]
    fsmap : tuple mapping trustee -> FS slot.
    return_diagnostics : if True, also return per-trial model quantities.
    n_train : if given, only trials [0, n_train) contribute to the returned
        log-likelihood (used to fit on train trials only), but the RL
        weights are still updated causally across *all* T trials so that
        `return_diagnostics` output covers the full (train+test) trajectory.

    Returns
    -------
    logL (float)  [and a dict of diagnostics if requested]
    """
    alpha, beta, alphaFS, betaFS, kappa, fsmap, K = unpack(pars, fsmap)
    data = np.asarray(data, dtype=float)
    T = data.shape[0]

    # weights initialised at zero -> no priors
    ws1 = np.zeros(4)   # identity weights
    ws2 = np.zeros(5)   # cue (expression) weights
    at = np.arange(1, 6)          # possible actions 1..5 (== 10..50 units, in tens)
    nA = at.size
    prev_action = None             # last non-omitted action, for the sticky-choice bonus
    n1 = np.zeros(4)   # per-identity visit counts (count_based_lr only)
    n2 = np.zeros(5)   # per-expression visit counts (count_based_lr only)

    prt = np.full(T, np.nan)
    predRR = np.zeros(T)
    U = np.zeros((T, nA))
    P = np.zeros((T, nA))
    RPE = np.zeros(T)
    ownval = np.zeros(T)
    otherval = np.zeros(T)
    envy = np.zeros(T)
    guilt = np.zeros(T)

    for t in range(T):
        t_stim = int(data[t, 0])
        t_emo = int(data[t, 1])
        t_rzq = data[t, 2]
        t_a = int(data[t, 3])

        f1 = np.zeros(4); f1[t_stim - 1] = 1.0
        f2 = np.zeros(5); f2[t_emo - 1] = 1.0
        qt = ws1 @ f1 + ws2 @ f2
        predRR[t] = qt

        if t_a == 0:              # omission: no decision, no learning signal
            prt[t] = 1.0 / nA
            if t > 0:
                U[t] = U[t - 1]
            continue

        Vai = (5 - at) + 3 * at * qt      # own gain G_i(a) (in tens)
        Vaj = 3 * at - 3 * at * qt        # other gain G_j(a) (in tens)
        j = fsmap[t_stim - 1] - 1
        disadv = np.maximum(Vaj - Vai, 0.0)   # trustee ahead -> envy
        advan = np.maximum(Vai - Vaj, 0.0)    # investor ahead -> guilt
        Ut = Vai - alphaFS[j] * disadv - betaFS[j] * advan   # Fehr-Schmidt utility
        if prev_action is not None:
            Ut[prev_action - 1] += kappa      # sticky-choice bonus for repeating the last action
        U[t] = Ut

        ownval[t] = Vai[t_a - 1]
        otherval[t] = Vaj[t_a - 1]
        envy[t] = disadv[t_a - 1]
        guilt[t] = advan[t_a - 1]

        # softmax choice probability (numerically stabilised)
        z = (1.0 / beta) * (Ut - Ut.max())
        ez = np.exp(z)
        pr = ez / ez.sum()
        P[t] = pr
        prt[t] = pr[t_a - 1]

        # reward prediction-error weight update (single learning rate)
        delta = t_rzq - qt
        if count_based_lr:
            n1[t_stim - 1] += 1
            n2[t_emo - 1] += 1
            ws1 = ws1 + (1.0 / n1[t_stim - 1]) * delta * f1
            ws2 = ws2 + (1.0 / n2[t_emo - 1]) * delta * f2
        else:
            ws1 = ws1 + alpha * delta * f1
            ws2 = ws2 + alpha * delta * f2
        RPE[t] = delta
        prev_action = t_a

    scored = prt[:n_train] if n_train is not None else prt
    if np.any(scored <= 0) or not np.all(np.isfinite(scored)):
        logL = -1e6
    else:
        logL = float(np.sum(np.log(scored)))

    if return_diagnostics:
        diag = dict(pr_t=prt, predRR=predRR, U=U, P=P, RPE=RPE,
                    ownval=ownval, otherval=otherval, envy=envy, guilt=guilt)
        return logL, diag
    return logL


def neg_loglik(pars, data, fsmap=(1, 2, 3, 4), n_train=None, count_based_lr=False):
    """Objective for minimisers."""
    return -loglik(pars, data, fsmap=fsmap, n_train=n_train, count_based_lr=count_based_lr)
