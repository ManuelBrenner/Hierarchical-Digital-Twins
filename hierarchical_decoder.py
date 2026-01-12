import torch
import torch.nn as nn
import torch.nn.functional as F

class HierarchicalDecoderCumulativeLink(nn.Module):
    """
    Cumulative link decoder supporting shared, individual, and hierarchical (feature-vector-based) parameterizations.
    """
    def __init__(self, dz, dq, num_categories=5, n_subjects=1, decoder_mode='shared', feature_dim=None):
        super().__init__()
        self.dz = dz
        self.dq = dq
        self.num_categories = num_categories
        self.n_subjects = n_subjects
        self.decoder_mode = decoder_mode
        self.feature_dim = feature_dim

        if decoder_mode == 'shared':
            # One set of parameters for all subjects
            self.beta_0 = nn.Parameter(torch.randn(dq, num_categories - 1) * 0.1)
            self.beta = nn.Parameter(torch.randn(dq, dz) * 0.1)
        elif decoder_mode == 'individual':
            # One set of parameters per subject
            self.beta_0 = nn.Parameter(torch.randn(n_subjects, dq, num_categories - 1) * 0.1)
            self.beta = nn.Parameter(torch.randn(n_subjects, dq, dz) * 0.1)
        elif decoder_mode == 'hierarchical':
            assert feature_dim is not None, "feature_dim must be provided for hierarchical decoder."
            # Project feature vector to decoder parameters
            self.beta_0_proj = nn.Linear(feature_dim, dq * (num_categories - 1))
            self.beta_proj = nn.Linear(feature_dim, dq * dz)
        else:
            raise ValueError(f"Unknown decoder_mode: {decoder_mode}")

    def get_decoder_params(self, subject_ids=None, feature_vectors=None):
        if self.decoder_mode == 'shared':
            # Shared parameters
            beta_0 = self.beta_0.unsqueeze(0)  # (1, dq, num_categories-1)
            beta = self.beta.unsqueeze(0)      # (1, dq, dz)
        elif self.decoder_mode == 'individual':
            # Select parameters for each subject in the batch
            beta_0 = self.beta_0[subject_ids]  # (batch, dq, num_categories-1)
            beta = self.beta[subject_ids]      # (batch, dq, dz)
        elif self.decoder_mode == 'hierarchical':
            # Generate parameters from feature vectors
            # feature_vectors: (batch, feature_dim)
            if feature_vectors is None:
                raise ValueError("feature_vectors must be provided for hierarchical decoder mode")
            batch_size = feature_vectors.shape[0]
            beta_0 = self.beta_0_proj(feature_vectors)  # (batch, dq*(num_categories-1))
            beta_0 = beta_0.view(batch_size, self.dq, self.num_categories - 1)
            beta = self.beta_proj(feature_vectors)      # (batch, dq*dz)
            beta = beta.view(batch_size, self.dq, self.dz)
        else:
            raise ValueError(f"Unknown decoder_mode: {self.decoder_mode}")
        return beta_0, beta

    def forward(self, z, subject_ids=None, feature_vectors=None):
        """
        z: (batch, T, dz)
        subject_ids: (batch,)
        feature_vectors: (batch, feature_dim) if hierarchical
        """
        probabilities = self.get_category_probabilities(z, subject_ids, feature_vectors)
        x = torch.argmax(probabilities, dim=-1) + 1
        x = x.float()
        x[torch.any(z.isnan(), dim=-1), :] = float('nan')
        return x

    def get_category_probabilities(self, z, subject_ids=None, feature_vectors=None):
        beta_0, beta = self.get_decoder_params(subject_ids, feature_vectors)
        linear_predictor = self.calculate_linear_predictor(z, beta_0, beta)
        cumulative_probabilities = self.inverse_logit_link_function(linear_predictor)
        probabilities = self.calculate_probabilities_from_cumulative_probabilities(cumulative_probabilities)
        # Handle numerical issues
        probabilities[(probabilities < 1e-5) & (probabilities > -1e-5)] = 0.
        prob_sum = probabilities.sum(dim=-1, keepdim=True)
        probabilities = probabilities / (prob_sum + 1e-10)
        return probabilities

    def calculate_linear_predictor(self, z, beta_0, beta):
        batch_size, T, dz = z.shape
        dq = self.dq
        num_cats_minus_1 = self.num_categories - 1

        if beta.shape[0] == 1:
            # Shared: beta (1, dq, dz) -> (dq, dz)
            beta = beta[0]
            # --- FIX: reparameterize beta_0 as in non-hierarchical decoder ---
            beta_0 = beta_0[0]
            beta_0_tilde = torch.cumsum(torch.exp(beta_0), dim=1)
            beta_0_tilde = beta_0_tilde - torch.exp(beta_0[:, 0]).unsqueeze(1) + beta_0[:, 0].unsqueeze(1)
            z_flat = z.reshape(-1, dz)  # (batch * T, dz)
            lin_pred = z_flat @ beta.T  # (batch * T, dq)
            lin_pred = lin_pred.reshape(batch_size, T, dq)
            lin_pred = beta_0_tilde.unsqueeze(0).unsqueeze(0) - lin_pred.unsqueeze(-1)
        else:
            # Individual/hierarchical: beta (batch, dq, dz)
            lin_pred = []
            for b in range(batch_size):
                z_b = z[b]  # (T, dz)
                beta_b = beta[b]  # (dq, dz)
                beta0_b = beta_0[b]  # (dq, num_cats_minus_1)
                # --- FIX: reparameterize beta0_b as in shared branch ---
                beta0_b_tilde = torch.cumsum(torch.exp(beta0_b), dim=1)
                beta0_b_tilde = beta0_b_tilde - torch.exp(beta0_b[:, 0]).unsqueeze(1) + beta0_b[:, 0].unsqueeze(1)
                z_flat = z_b  # (T, dz)
                lin_pred_b = z_flat @ beta_b.T  # (T, dq)
                lin_pred_b = beta0_b_tilde.unsqueeze(0) - lin_pred_b.unsqueeze(-1)  # (T, dq, num_cats_minus_1)
                lin_pred.append(lin_pred_b)
            lin_pred = torch.stack(lin_pred, dim=0)  # (batch, T, dq, num_cats_minus_1)
        return lin_pred

    def inverse_logit_link_function(self, linear_predictor):
        cumul_prob = torch.sigmoid(linear_predictor)
        # Ensure cumul_prob is 4D
        if cumul_prob.dim() == 3:
            cumul_prob = cumul_prob.unsqueeze(-1)
        batch_size, T, dq, num_cats_minus_1 = cumul_prob.shape
        ones = torch.ones(batch_size, T, dq, 1, device=cumul_prob.device)
        cumul_prob = torch.cat([cumul_prob, ones], dim=-1)
        return cumul_prob

    def calculate_probabilities_from_cumulative_probabilities(self, cumul_prob):
        zeros = torch.zeros(cumul_prob.shape[:-1] + (1,), device=cumul_prob.device)
        prepended = torch.cat([zeros, cumul_prob], dim=-1)
        prob = torch.diff(prepended, dim=-1)
        return prob

    def log_likelihood(self, x, z, subject_ids=None, feature_vectors=None):
        probabilities = self.get_category_probabilities(z, subject_ids, feature_vectors)
        if probabilities.shape[2] == 1:
            probabilities = probabilities.squeeze(2)
        else:
            probabilities = probabilities[:, :, 0, :]
        batch_size, T, num_categories = probabilities.shape
        even_mask = torch.zeros(T, dtype=torch.bool, device=probabilities.device)
        even_mask[::2] = True
        nan_mask = ~torch.isnan(x).squeeze(-1)
        valid_mask = even_mask.unsqueeze(0) & nan_mask
        if not valid_mask.any():
            return torch.tensor(0.0, device=probabilities.device)
        valid_probabilities = probabilities[valid_mask]
        valid_x = x.squeeze(-1)[valid_mask]
        categorical_indices = valid_x.long() - 1
        if -1 in categorical_indices:
            categorical_indices[categorical_indices == -1] = 1
        batch_indices = torch.arange(valid_probabilities.shape[0], device=valid_probabilities.device)
        selected_probs = valid_probabilities[batch_indices, categorical_indices]
        selected_probs[selected_probs < 1e-10] = 1e-10
        ll_cumulative_link = torch.sum(torch.log(selected_probs))
        assert torch.isfinite(ll_cumulative_link)
        return ll_cumulative_link

    def predict_spikes(self, z, subject_ids=None, feature_vectors=None, n_samples=1):
        predictions = self.forward(z, subject_ids, feature_vectors)
        return predictions, predictions 