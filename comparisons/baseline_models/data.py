from __future__ import annotations
import numpy as np

"""
data.py
=======
Adapter from this repo's (stacked_invests.npy, stacked_inputs.npy) format to
the (T, 4) per-subject trial array [stimtyp, shown_emo, repayment_ratio,
action] expected by `fs_rl_model.py`.

Timestep layout (see project context): each of the 80 real trials per
subject occupies 2 consecutive timesteps out of 160, alternating:

    decision step (even index t=2k):   cue one-hot in inputs, action in invests
    feedback step (odd index t=2k+1):  repayment info in inputs, NaN in invests

`stacked_inputs.npy` columns:
    [0:4]  identity one-hot (4 trustees)
    [4:9]  expression one-hot (5 levels; already forced to "neutral" by the
           task for the two neutral-trustee identities)
    [9]    feedback: repayment amount received, 3*a*r_tilde (in tens)
    [10]   feedback: own gain this trial, (5-a) + 3*a*r_tilde (in tens)

Both feedback columns are algebraically redundant given the action `a` and
therefore pin down the actually-observed repayment ratio r_tilde exactly:

    r_tilde = feedback[9] / (3*a)

(cross-checked against the second column: max abs discrepancy ~1e-16 over
the full dataset).

Trials with no decision (investment NaN, i.e. an omission) are encoded with
action = 0, matching `fs_rl_model.py`'s omission handling.
"""


def build_subject_trials(inputs_subj: np.ndarray, invests_subj: np.ndarray) -> np.ndarray:
    """
    Build the (n_trials, 4) [stimtyp, shown_emo, repayment_ratio, action]
    array for one subject from that subject's (T, 11) inputs and (T, 1) (or
    (T,)) invests arrays.
    """
    invests_subj = np.asarray(invests_subj).reshape(-1)
    T = invests_subj.shape[0]
    assert T % 2 == 0, f"expected an even number of timesteps, got {T}"
    n_trials = T // 2

    dec_inputs = inputs_subj[0::2]     # (n_trials, 11)
    dec_actions = invests_subj[0::2]   # (n_trials,)
    fb_inputs = inputs_subj[1::2]      # (n_trials, 11)

    stimtyp = np.argmax(dec_inputs[:, 0:4], axis=1) + 1
    shown_emo = np.argmax(dec_inputs[:, 4:9], axis=1) + 1

    action = np.where(np.isnan(dec_actions), 0, dec_actions).astype(int)

    repayment_amount = fb_inputs[:, 9]
    with np.errstate(invalid="ignore", divide="ignore"):
        repayment_ratio = repayment_amount / (3.0 * np.where(action == 0, 1, action))
    repayment_ratio = np.where(action == 0, 0.0, repayment_ratio)

    data = np.stack([stimtyp, shown_emo, repayment_ratio, action], axis=1).astype(float)
    return data


def build_all_subjects(inputs: np.ndarray, invests: np.ndarray) -> list[np.ndarray]:
    """Build per-subject trial arrays for every subject in the dataset."""
    n_subjects = inputs.shape[0]
    return [build_subject_trials(inputs[s], invests[s]) for s in range(n_subjects)]


def load_dataset(inputs_path="../../data/stacked_inputs.npy",
                  invests_path="../../data/stacked_invests.npy",
                  train_split: float = 0.75):
    """
    Load the .npy files and return per-subject trial arrays plus the trial
    count that corresponds to `train_split` of the *timesteps*, matching
    `dataset.SimplifiedDataset` exactly (train_split=0.75 -> first 120 of 160
    timesteps -> first 60 of 80 trials train, last 20 trials test).
    """
    inputs = np.load(inputs_path)
    invests = np.load(invests_path)
    total_timesteps = inputs.shape[1]
    train_timesteps = int(total_timesteps * train_split)
    assert train_timesteps % 2 == 0
    n_train_trials = train_timesteps // 2

    subjects = build_all_subjects(inputs, invests)
    n_trials = subjects[0].shape[0]
    n_test_trials = n_trials - n_train_trials
    return subjects, n_train_trials, n_test_trials
