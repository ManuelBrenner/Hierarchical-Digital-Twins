import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

def xavier_uniform_(tensor, gain=1.0):
    """Xavier uniform initialization"""
    fan_in, fan_out = tensor.size()
    std = gain * math.sqrt(2.0 / (fan_in + fan_out))
    bound = math.sqrt(3.0) * std
    with torch.no_grad():
        return tensor.uniform_(-bound, bound)

class HierarchicalAL_RNN(nn.Module):
    def __init__(self, M, P, N_feat=20, n_trials=1, input_dim=0, use_inputs=True, scaling=0.05, learn_initial_states=True):
        """
        Hierarchical version of AL-RNN where model parameters are generated from trial-specific feature vectors.
        
        Args:
            M: Latent dimension
            P: Number of positive units
            N_feat: Dimension of feature vectors
            n_trials: Number of trials (for initial states)
            input_dim: Dimension of external inputs
            use_inputs: Whether to use external inputs
            scaling: Scaling factor for parameter initialization
            learn_initial_states: Whether to learn initial states or fix them to zeros
        """
        super(HierarchicalAL_RNN, self).__init__()
        
        # Initialize model dimensions
        self.M = M
        self.P = P
        self.N_feat = N_feat
        self.n_trials = n_trials
        self.input_dim = input_dim if use_inputs else 0
        self.use_inputs = use_inputs
        self.scaling = scaling
        self.learn_initial_states = learn_initial_states
        
        # Initialize feature vectors for trials
        initial_feature_vector = torch.randn(1, self.N_feat)
        self.feature_vectors = nn.Parameter(initial_feature_vector.repeat(self.n_trials, 1))
        
        # Initialize projection matrices for generating model parameters
        self.proj_A = nn.Parameter(xavier_uniform_(torch.empty(self.N_feat, self.M)) * self.scaling)
        self.proj_W = nn.Parameter(xavier_uniform_(torch.empty(self.N_feat, self.M * self.M)) * self.scaling)
        self.proj_h = nn.Parameter(xavier_uniform_(torch.empty(self.N_feat, self.M)) * self.scaling)
        
        # Initialize projection for initial states (z0)
        self.proj_z0 = nn.Parameter(xavier_uniform_(torch.empty(self.N_feat, self.M)) * self.scaling)
        
        # Initialize input projection if using external inputs
        if use_inputs:
            self.proj_C = nn.Parameter(xavier_uniform_(torch.empty(self.N_feat, self.M * self.input_dim)) * self.scaling)
        
        # Simple linear projection for classification (2 classes)
        self.D = nn.Parameter(torch.randn(2, self.M) * 0.1)
        
        print(f"Initialized HierarchicalAL_RNN with:")
        print(f"- Latent dimension (M): {self.M}")
        print(f"- Number of positive units (P): {self.P}")
        print(f"- Feature vector dimension (N_feat): {self.N_feat}")
        print(f"- Number of trials: {self.n_trials}")
        print(f"- Using external inputs: {self.use_inputs}")
        if use_inputs:
            print(f"- Input dimension: {self.input_dim}")
    
    def get_model_params(self, trial_ids):
        # Get feature vectors for the batch of trials
        feature_vectors = self.feature_vectors[trial_ids]  # (batch_size, N_feat)
        
        # Generate parameters through projections for each trial in batch
        A = feature_vectors @ self.proj_A  # (batch_size, M)
        W = (feature_vectors @ self.proj_W).view(-1, self.M, self.M)  # (batch_size, M, M)
        h = feature_vectors @ self.proj_h  # (batch_size, M)
        
        if self.use_inputs:
            C = (feature_vectors @ self.proj_C).view(-1, self.M, self.input_dim)  # (batch_size, M, input_dim)
        else:
            C = None
            
        return A, W, h, C
    
    def get_initial_state(self, trial_ids):
        """Generate initial states from trial-specific feature vectors"""
        feature_vectors = self.feature_vectors[trial_ids]  # (batch_size, N_feat)
        z0 = feature_vectors @ self.proj_z0  # (batch_size, M)
        return z0
    
    def forward_step(self, z, input, trial_ids):
        """
        Single step forward pass of the RNN for a batch of trials.
        
        Args:
            z: Current latent state (batch_size, M)
            input: External input (batch_size, input_dim) or None if not using inputs
            trial_ids: Trial IDs for parameter generation (batch_size,)
            
        Returns:
            Updated latent state (batch_size, M)
        """
        # Get model parameters for this batch of trials
        A, W, h, C = self.get_model_params(trial_ids)  # All parameters are now batch-specific
        
        # Make a copy of the input tensor to retain unactivated values
        z_unactivated = torch.clone(z)
        
        # Apply ReLU activation on the last P units (without inplace operation)
        z_activated = torch.cat([z[:, :-self.P], F.relu(z[:, -self.P:])], dim=1)
        
        # Compute the forward pass for each trial in batch
        # A * z_unactivated: (batch_size, M) * (batch_size, M) -> (batch_size, M)
        # z_activated @ W.transpose(1, 2): (batch_size, M) @ (batch_size, M, M) -> (batch_size, M)
        out = A * z_unactivated + torch.bmm(z_activated.unsqueeze(1), W.transpose(1, 2)).squeeze(1) + h
        
        # Add external input if using inputs
        if self.use_inputs and input is not None:
            # Convert NaN values in input to zero
            input_clean = torch.nan_to_num(input, nan=0.0)
            
            # input @ C.t(): (batch_size, input_dim) @ (batch_size, M, input_dim) -> (batch_size, M)
            input_term = torch.bmm(input_clean.unsqueeze(1), C.transpose(1, 2)).squeeze(1)
            out = out + input_term
            
        return out
    
    def predict_label(self, z):
        """Predict class logits using linear projection"""
        # Simple linear projection to 2D (one output per class)
        return z @ self.D.t()  # (batch_size, 2) - raw logits for both classes

    def get_feature_vectors(self, subject_ids):
        """Return feature vectors for a batch of subject_ids (for hierarchical decoder)."""
        return self.feature_vectors[subject_ids]

def predict_sequence_using_gtf(model, initial_state, inputs, encoded_x, alpha, n_interleave, trial_ids, lengths=None):
    """
    Predict sequence using teacher forcing.
    
    Args:
        model: HierarchicalAL_RNN model
        initial_state: Initial latent state (batch_size, M)
        inputs: External inputs (batch_size, T, input_dim) or None if not using inputs
        encoded_x: Encoded states for teacher forcing (batch_size, T, M)
        alpha: Teacher forcing parameter
        n_interleave: Teacher forcing frequency
        trial_ids: Trial IDs for parameter generation (batch_size,)
        lengths: Tensor of sequence lengths (batch_size,) or None if all sequences are same length
    """
    T = encoded_x.size(1)  # number of time steps
    b = initial_state.size(0)  # batch size
    M = initial_state.size(1)  # latent dimension
    Z = torch.empty(size=(T, b, M), device=initial_state.device)
    z = initial_state
    
    # Generate sequence with teacher forcing
    for t in range(T):
        # Apply teacher forcing at regular intervals
        if (t % n_interleave == 0) and (t > 0):
            z = alpha * encoded_x[:, t] + (1 - alpha) * z
        
        # Update state with current input using forward_step
        if model.use_inputs:
            z = model.forward_step(z, inputs[:, t], trial_ids)
        else:
            z = model.forward_step(z, None, trial_ids)
        
        Z[t] = z
    
    # Get final states based on actual sequence lengths if provided
    if lengths is not None:
        # Convert lengths to 0-based indices for indexing
        indices = lengths - 1  # (batch_size,)
        # Get the final state for each sequence based on its length
        final_states = Z[indices, torch.arange(b, device=Z.device)]  # (batch_size, M)
    else:
        # If no lengths provided, use the last state
        final_states = Z[-1]  # (batch_size, M)
    
    predicted_labels = model.predict_label(final_states)
    
    return Z.permute(1, 0, 2), predicted_labels  # (batch_size, T, M), (batch_size, 2)

def latent_likelihood(z_lat, z_enc, R_z):
    """Compute latent likelihood"""
    def mahalonobis_distance(residual, matrix):
        distance = 0
        for i in range(residual.shape[0]):
            distance += -0.5 * (residual[i].t() @ residual[i] * torch.inverse(torch.diag(matrix ** 2))).sum()
        return distance

    def log_det(diagonal_matrix):
        return -torch.log(diagonal_matrix).sum()

    LL_z = mahalonobis_distance(z_lat - z_enc, R_z)
    T = z_lat.shape[1]

    return LL_z + log_det(R_z) * (T - 1) / 2 