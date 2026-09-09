import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # comparisons/ -> repo root
from non_hierarchical_al_rnn import xavier_uniform_, xavier_uniform_3d_  # noqa: E402

class NonHierarchicalGRU(nn.Module):
    """
    Non-hierarchical GRU backbone: each subject gets their own independent GRU cell parameters (no
    weight sharing, no hypernetwork) -- a drop-in replacement for NonHierarchicalAL_RNN. Implements the
    same interface (get_initial_state, forward_step, predict_label, use_inputs) expected by
    predict_sequence_using_gtf, so the rest of the pipeline (teacher forcing, decoder, encoder, losses)
    is unchanged.

    torch.nn.GRUCell can't be used directly: its weights are single nn.Parameters shared across the
    whole batch, but every subject needs its own distinct weight matrices selected by subject_ids -- so
    the cell update is implemented manually here, matching nn.GRUCell's equations.
    """
    def __init__(self, M, P=None, n_subjects=1, input_dim=0, use_inputs=True, scaling=0.05, learn_initial_states=True):
        """
        Args:
            M: Latent/hidden dimension
            P: Unused (kept for interface parity with NonHierarchicalAL_RNN, which uses it for the
               piecewise-linear ReLU split -- the GRU has no such split)
            n_subjects: Number of subjects (each gets their own parameters)
            input_dim: Dimension of external inputs
            use_inputs: Whether to use external inputs
            scaling: Scaling factor for parameter initialization
            learn_initial_states: Whether to learn initial states or fix them to zeros
        """
        super().__init__()
        self.M = M
        self.n_subjects = n_subjects
        self.input_dim = input_dim if use_inputs else 0
        self.use_inputs = use_inputs
        self.scaling = scaling
        self.learn_initial_states = learn_initial_states

        # Packed gate order matches torch.nn.GRUCell: reset, update, new
        self.weight_ih = nn.Parameter(xavier_uniform_3d_(torch.empty(n_subjects, 3 * M, self.input_dim)) * scaling)
        self.weight_hh = nn.Parameter(xavier_uniform_3d_(torch.empty(n_subjects, 3 * M, M)) * scaling)
        self.bias_ih = nn.Parameter(torch.zeros(n_subjects, 3 * M))
        self.bias_hh = nn.Parameter(torch.zeros(n_subjects, 3 * M))

        # Initial state parameters: (n_subjects, M)
        self.z0_params = nn.Parameter(xavier_uniform_(torch.empty(n_subjects, M)) * scaling)

        # Unused downstream currently, kept only for interface parity with NonHierarchicalAL_RNN
        self.D = nn.Parameter(torch.randn(2, M) * 0.1)

        print(f"Initialized NonHierarchicalGRU with:")
        print(f"- Hidden/latent dimension (M): {self.M}")
        print(f"- Number of subjects: {self.n_subjects}")
        print(f"- Using external inputs: {self.use_inputs}")
        if use_inputs:
            print(f"- Input dimension: {self.input_dim}")

    def get_initial_state(self, subject_ids):
        """Return per-subject learned initial states"""
        return self.z0_params[subject_ids]

    def forward_step(self, z, input, subject_ids):
        """
        Single GRU cell step for a batch of subjects, using each subject's own weights.

        Args:
            z: Current hidden state (batch_size, M)
            input: External input (batch_size, input_dim) or None if not using inputs
            subject_ids: Subject IDs for parameter selection (batch_size,)

        Returns:
            Updated hidden state (batch_size, M)
        """
        W_ih = self.weight_ih[subject_ids]  # (batch, 3M, input_dim)
        W_hh = self.weight_hh[subject_ids]  # (batch, 3M, M)
        b_ih = self.bias_ih[subject_ids]    # (batch, 3M)
        b_hh = self.bias_hh[subject_ids]    # (batch, 3M)

        if self.use_inputs and input is not None:
            x = torch.nan_to_num(input, nan=0.0)
        else:
            x = torch.zeros(z.shape[0], self.input_dim, device=z.device)

        gi = torch.bmm(W_ih, x.unsqueeze(-1)).squeeze(-1) + b_ih  # (batch, 3M)
        gh = torch.bmm(W_hh, z.unsqueeze(-1)).squeeze(-1) + b_hh  # (batch, 3M)

        i_r, i_z, i_n = gi.chunk(3, dim=1)
        h_r, h_z, h_n = gh.chunk(3, dim=1)

        r = torch.sigmoid(i_r + h_r)
        upd = torch.sigmoid(i_z + h_z)
        n = torch.tanh(i_n + r * h_n)
        z_new = (1 - upd) * n + upd * z
        return z_new

    def predict_label(self, z):
        """Predict class logits using linear projection"""
        return z @ self.D.t()  # (batch_size, 2)
