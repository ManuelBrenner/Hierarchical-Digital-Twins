from __future__ import annotations
import json
from pathlib import Path

"""
summarize_results.py
=====================
Consolidates the scattered result files produced across the RL-baseline
experiments (fs_rl_results*.json, decoder_variants_results.json) into one
summary table, so the numbers used in any write-up are pulled directly from
the saved fit results rather than retyped by hand.
"""

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def _load(name):
    with open(RESULTS_DIR / name) as f:
        return json.load(f)


def main():
    baseline = _load("fs_rl_results.json")                # fixed-alpha, FS+softmax, flat, no kappa
    plus_kappa = _load("fs_rl_results_flat.json")          # + sticky-choice kappa, flat
    plus_hier = _load("fs_rl_results_hierarchical.json")   # + kappa, hierarchical shrinkage
    plus_countlr = _load("fs_rl_results_countlr_flat.json")  # + kappa, count-based LR, flat
    decoders = _load("decoder_variants_results.json")      # softmax_r / ordinal_r / softmax_cue

    rows = [
        ("1. RL baseline (fixed-α, FS utility + softmax)", baseline["mean_test_mae"], baseline["mean_test_nll"]),
        ("2. + sticky-choice κ", plus_kappa["mean_test_mae"], plus_kappa["mean_test_nll"]),
        ("3. + hierarchical shrinkage", plus_hier["mean_test_mae"], plus_hier["mean_test_nll"]),
        ("4. + count-based learning rate", plus_countlr["mean_test_mae"], plus_countlr["mean_test_nll"]),
        ("5. softmax decoder on r_t (softmax_r)", decoders["softmax_r"]["mean_test_mae"], decoders["softmax_r"]["mean_test_nll"]),
        ("6. ordinal decoder on r_t (ordinal_r)", decoders["ordinal_r"]["mean_test_mae"], decoders["ordinal_r"]["mean_test_nll"]),
        ("7. softmax decoder on raw cue (softmax_cue)", decoders["softmax_cue"]["mean_test_mae"], decoders["softmax_cue"]["mean_test_nll"]),
    ]

    summary = {
        "n_subjects": baseline["n_subjects"],
        "n_train_trials": baseline["n_train_trials"],
        "n_test_trials": baseline["n_test_trials"],
        "models": [dict(name=n, test_mae=mae, test_nll=nll) for n, mae, nll in rows],
    }

    print(f"{'model':<45s} {'test MAE':>10s} {'test NLL':>10s}")
    for name, mae, nll in rows:
        print(f"{name:<45s} {mae:>10.3f} {nll:>10.3f}")

    out_path = RESULTS_DIR / "summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved consolidated summary to {out_path}")


if __name__ == "__main__":
    main()
