import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Poisson, NegativeBinomial
import math

class BaseDecoder(nn.Module):
    """Base class for all decoders with unified interface"""
    def __init__(self, dz, dq):
        super().__init__()
        self.dz = dz  # latent dimension
        self.dq = dq  # output dimension (number of neurons)
        
    def forward(self, z):
        """Convert latent states to distribution parameters"""
        raise NotImplementedError
        
    def log_likelihood(self, x, z):
        """Compute log likelihood of observations given latent states"""
        raise NotImplementedError
        
    def predict_spikes(self, z, n_samples=1):
        """Generate spike predictions from latent states"""
        raise NotImplementedError

class Decoder_Poisson(BaseDecoder):
    """Poisson decoder for spike count data"""
    def __init__(self, dz, dq):
        super().__init__(dz, dq)
        self.linear = nn.Linear(dz, dq)
        
    def forward(self, z):
        """Convert latent states to Poisson rate parameters"""
        return torch.exp(self.linear(z))
    
    def log_likelihood(self, x, z):
        """Compute Poisson log likelihood"""
        rates = self.forward(z)
        dist = Poisson(rates)
        return dist.log_prob(x).sum(dim=-1)
    
    def predict_spikes(self, z, n_samples=1):
        """Generate spike predictions"""
        rates = self.forward(z)
        dist = Poisson(rates)
        spikes = dist.sample((n_samples,))
        return spikes, rates

class Decoder_GeneralizedPoisson(BaseDecoder):
    """Generalized Poisson decoder with dispersion parameter"""
    def __init__(self, dz, dq):
        super().__init__(dz, dq)
        self.rate_linear = nn.Linear(dz, dq)
        self.dispersion_linear = nn.Linear(dz, dq)
        
    def forward(self, z):
        """Convert latent states to rate and dispersion parameters"""
        rates = torch.exp(self.rate_linear(z))
        dispersion = torch.exp(self.dispersion_linear(z))
        return rates, dispersion
    
    def log_likelihood(self, x, z):
        """Compute generalized Poisson log likelihood"""
        rates, dispersion = self.forward(z)
        # Simplified implementation - you may want to use a proper generalized Poisson distribution
        dist = Poisson(rates)
        return dist.log_prob(x).sum(dim=-1)
    
    def predict_spikes(self, z, n_samples=1):
        """Generate spike predictions"""
        rates, dispersion = self.forward(z)
        dist = Poisson(rates)
        spikes = dist.sample((n_samples,))
        return spikes, rates

class Decoder_NegativeBinomial(BaseDecoder):
    """Negative Binomial decoder"""
    def __init__(self, dz, dq):
        super().__init__(dz, dq)
        self.rate_linear = nn.Linear(dz, dq)
        self.dispersion_linear = nn.Linear(dz, dq)
        
    def forward(self, z):
        """Convert latent states to rate and dispersion parameters"""
        rates = torch.exp(self.rate_linear(z))
        dispersion = torch.exp(self.dispersion_linear(z))
        return rates, dispersion
    
    def log_likelihood(self, x, z):
        """Compute negative binomial log likelihood"""
        rates, dispersion = self.forward(z)
        dist = NegativeBinomial(rates, dispersion)
        return dist.log_prob(x).sum(dim=-1)
    
    def predict_spikes(self, z, n_samples=1):
        """Generate spike predictions"""
        rates, dispersion = self.forward(z)
        dist = NegativeBinomial(rates, dispersion)
        spikes = dist.sample((n_samples,))
        return spikes, rates

class Decoder_cumulative_link(BaseDecoder):
    """Cumulative link decoder for ordinal data"""
    def __init__(self, dz, dq, num_categories=5):
        super().__init__(dz, dq)
        self.num_categories = num_categories
        
        # Parameters for cumulative link model
        self.beta_0 = nn.Parameter(torch.randn(dq, num_categories - 1) * 0.1, requires_grad=True)
        self.beta = nn.Parameter(torch.randn(dq, dz) * 0.1, requires_grad=True)
        
    def forward(self, z):
        """Convert latent states to category probabilities"""
        probabilities = self.get_category_probabilities(z)
        # probabilities shape: (batch_size, T, dq, num_categories)
        # Take argmax along the last dimension (categories)
        x = torch.argmax(probabilities, dim=-1) + 1
        x = x.float()
        # Handle NaN values
        x[torch.any(z.isnan(), dim=-1), :] = float('nan')
        return x
    
    def get_category_probabilities(self, z):
        """Get probabilities for each category"""
        linear_predictor = self.calculate_linear_predictor(z)
        cumulative_probabilities = self.inverse_logit_link_function(linear_predictor)
        probabilities = self.calculate_probabilities_from_cumulative_probabilities(cumulative_probabilities)
        
        # Handle numerical issues
        probabilities[(probabilities < 1e-5) & (probabilities > -1e-5)] = 0.
        
        # Ensure probabilities sum to 1 along the category dimension
        prob_sum = probabilities.sum(dim=-1, keepdim=True)
        probabilities = probabilities / (prob_sum + 1e-10)
        
        return probabilities
    
    def calculate_linear_predictor(self, z):
        """Calculate linear predictor for cumulative link model"""

        batch_size, T, dz = z.shape
        z_flat = z.reshape(-1, dz)  # (batch_size * T, dz)
        
        # Compute linear predictor: z_flat @ beta.T
        lin_pred = z_flat @ self.beta.T  # (batch_size * T, dq)
        
        # Reshape back to (batch_size, T, dq)
        lin_pred = lin_pred.reshape(batch_size, T, -1)  # (batch_size, T, dq)
        
        # Get beta_0 and reshape for broadcasting
        beta_0 = self.reparameterize_beta_0()  # (dq, num_categories - 1)
        
        # Compute: beta_0 - lin_pred for each category
        # beta_0: (dq, num_categories - 1)
        # lin_pred: (batch_size, T, dq)
        # Result: (batch_size, T, dq, num_categories - 1)
        lin_pred = beta_0.unsqueeze(0).unsqueeze(0) - lin_pred.unsqueeze(-1)
        
        return lin_pred
    
    def inverse_logit_link_function(self, linear_predictor):
        """Apply inverse logit (sigmoid) link function"""
        # linear_predictor shape: (batch_size, T, dq, num_categories - 1)
        cumul_prob = torch.sigmoid(linear_predictor)
        
        # Add cumulative probability for the last category (always one)
        # Create a tensor of ones with the same shape as the last dimension
        batch_size, T, dq, num_cats_minus_1 = cumul_prob.shape
        ones = torch.ones(batch_size, T, dq, 1, device=cumul_prob.device)
        cumul_prob = torch.cat([cumul_prob, ones], dim=-1)
        
        return cumul_prob
    
    def calculate_probabilities_from_cumulative_probabilities(self, cumul_prob):
        """Convert cumulative probabilities to individual probabilities"""
        # cumul_prob shape: (batch_size, T, dq, num_categories)
        # Use torch.diff with prepend along the last dimension
        zeros = torch.zeros(cumul_prob.shape[:-1] + (1,), device=cumul_prob.device)
        prepended = torch.cat([zeros, cumul_prob], dim=-1)
        prob = torch.diff(prepended, dim=-1)
        return prob
    
    def reparameterize_beta_0(self):
        """Reparameterize beta_0 to ensure ordered parameters"""
        beta_0_tilde = torch.cumsum(torch.exp(self.beta_0), dim=1)
        beta_0_tilde = beta_0_tilde - torch.exp(self.beta_0[:, 0]).unsqueeze(1) + self.beta_0[:, 0].unsqueeze(1)
        return beta_0_tilde
    
    def log_likelihood(self, x, z):
        """Compute cumulative link log likelihood"""
        probabilities = self.get_category_probabilities(z)
        
        # probabilities shape: (batch_size, T, dq, num_categories)
        
        # For ordinal data, we typically have one output dimension (dq=1)
        # and we want to compute likelihood for each time step
        if probabilities.shape[2] == 1:  # dq = 1
            probabilities = probabilities.squeeze(2)  # (batch_size, T, num_categories)
        else:
            # If multiple output dimensions, we need to handle this case
            # For now, let's assume we're only interested in the first dimension
            probabilities = probabilities[:, :, 0, :]  # (batch_size, T, num_categories)
        
        # x shape: (batch_size, T, 1) - ordinal ratings 1-5 with NaN at odd time steps
        # probabilities shape: (batch_size, T, num_categories) - probabilities for each category
        
        # Only compute likelihood on valid (non-NaN) time steps (even indices: 0, 2, 4, ...)
        # Create mask for even time steps (0, 2, 4, ...)
        batch_size, T, num_categories = probabilities.shape
        even_mask = torch.zeros(T, dtype=torch.bool, device=probabilities.device)
        even_mask[::2] = True  # Set even indices to True
        
        # Also check for NaN values in the data
        nan_mask = ~torch.isnan(x).squeeze(-1)  # (batch_size, T)
        
        # Combine masks: we want even time steps AND non-NaN values
        valid_mask = even_mask.unsqueeze(0) & nan_mask  # (batch_size, T)
        
        if not valid_mask.any():
            # If no valid time steps, return 0
            return torch.tensor(0.0, device=probabilities.device)
        
        valid_probabilities = probabilities[valid_mask]  # (num_valid, num_categories)
        
        # For each valid data point find the correct categorical indices
        valid_x = x.squeeze(-1)[valid_mask]  # (num_valid,)
        categorical_indices = valid_x.long() - 1  # Convert 1-5 to 0-4
        if -1 in categorical_indices:
            categorical_indices[categorical_indices == -1] = 1
        
        # Gather the correct probabilities
        # valid_probabilities shape: (num_valid, num_categories)
        # categorical_indices shape: (num_valid,)
        batch_indices = torch.arange(valid_probabilities.shape[0], device=valid_probabilities.device)
        selected_probs = valid_probabilities[batch_indices, categorical_indices]
        
        # Handle numerical issues
        selected_probs[selected_probs < 1e-10] = 1e-10
        
        ll_cumulative_link = torch.sum(torch.log(selected_probs))
        
        assert torch.isfinite(ll_cumulative_link)
        return ll_cumulative_link
    
    def predict_spikes(self, z, n_samples=1):
        """Generate predictions from cumulative link model"""
        # For cumulative link, we return the most likely category
        predictions = self.forward(z)
        return predictions, predictions

class Decoder_softmax(BaseDecoder):
    """Categorical softmax decoder for ordinal data.

    Interface mirrors Decoder_cumulative_link (forward/log_likelihood/predict_spikes) so it is a
    drop-in swap, but each category gets its own free logit instead of shared ordered thresholds.
    """
    def __init__(self, dz, dq, num_categories=5):
        super().__init__(dz, dq)
        self.num_categories = num_categories

        self.weight = nn.Parameter(torch.randn(dq, num_categories, dz) * 0.1, requires_grad=True)
        self.bias = nn.Parameter(torch.randn(dq, num_categories) * 0.1, requires_grad=True)

    def forward(self, z):
        """Convert latent states to predicted category (most likely level)"""
        probabilities = self.get_category_probabilities(z)
        x = torch.argmax(probabilities, dim=-1) + 1
        x = x.float()
        x[torch.any(z.isnan(), dim=-1), :] = float('nan')
        return x

    def get_category_probabilities(self, z):
        """Get probabilities for each category"""
        logits = self.calculate_logits(z)
        probabilities = F.softmax(logits, dim=-1)
        return probabilities

    def calculate_logits(self, z):
        """Calculate per-category logits"""
        batch_size, T, dz = z.shape
        z_flat = z.reshape(-1, dz)  # (batch_size * T, dz)

        logits = torch.einsum('nd,qcd->nqc', z_flat, self.weight) + self.bias.unsqueeze(0)
        logits = logits.reshape(batch_size, T, self.dq, self.num_categories)
        return logits

    def log_likelihood(self, x, z):
        """Compute categorical softmax log likelihood"""
        probabilities = self.get_category_probabilities(z)

        if probabilities.shape[2] == 1:  # dq = 1
            probabilities = probabilities.squeeze(2)  # (batch_size, T, num_categories)
        else:
            probabilities = probabilities[:, :, 0, :]  # (batch_size, T, num_categories)

        batch_size, T, num_categories = probabilities.shape
        even_mask = torch.zeros(T, dtype=torch.bool, device=probabilities.device)
        even_mask[::2] = True  # Set even indices to True

        nan_mask = ~torch.isnan(x).squeeze(-1)  # (batch_size, T)
        valid_mask = even_mask.unsqueeze(0) & nan_mask  # (batch_size, T)

        if not valid_mask.any():
            return torch.tensor(0.0, device=probabilities.device)

        valid_probabilities = probabilities[valid_mask]  # (num_valid, num_categories)
        valid_x = x.squeeze(-1)[valid_mask]  # (num_valid,)
        categorical_indices = valid_x.long() - 1  # Convert 1-5 to 0-4
        if -1 in categorical_indices:
            categorical_indices[categorical_indices == -1] = 1

        batch_indices = torch.arange(valid_probabilities.shape[0], device=valid_probabilities.device)
        selected_probs = valid_probabilities[batch_indices, categorical_indices]

        selected_probs[selected_probs < 1e-10] = 1e-10

        ll_softmax = torch.sum(torch.log(selected_probs))

        assert torch.isfinite(ll_softmax)
        return ll_softmax

    def predict_spikes(self, z, n_samples=1):
        """Generate predictions from softmax model"""
        predictions = self.forward(z)
        return predictions, predictions

def get_decoder(decoder_type, dz, dq, num_categories=5):
    """Factory function to get decoder of specified type"""
    decoders = {
        'poisson': Decoder_Poisson,
        'generalized_poisson': Decoder_GeneralizedPoisson,
        'negative_binomial': Decoder_NegativeBinomial,
        'cumulative_link': lambda dz, dq: Decoder_cumulative_link(dz, dq, num_categories),
        'softmax': lambda dz, dq: Decoder_softmax(dz, dq, num_categories)
    }
    
    if decoder_type not in decoders:
        raise ValueError(f"Unknown decoder type: {decoder_type}. Choose from {list(decoders.keys())}")
    
    return decoders[decoder_type](dz, dq)