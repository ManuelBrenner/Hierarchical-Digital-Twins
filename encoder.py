import torch
import torch.nn as nn
import torch.nn.functional as F


class StackedConvolutions(nn.Module):
    def __init__(self, dim_x, dim_z, sample_rec=False, normalize_output=True, 
                 smoothness_weight=.2):
        """
        Encoder using stacked convolutions to map from spike counts to latent states.
        Includes temporal smoothing and explicit smoothness constraints.
        Maintains the same temporal resolution as the input.
        
        Args:
            dim_x: Input dimension (number of neurons)
            dim_z: Output dimension (latent dimension)
            sample_rec: Whether to sample from recognition distribution
            normalize_output: Whether to normalize the output using layer normalization
            smoothness_weight: Weight for the temporal smoothness loss
        """
        super(StackedConvolutions, self).__init__()
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.sample_rec = sample_rec
        self.smoothness_weight = smoothness_weight
        
        # Initial temporal smoothing layer with large kernel
        # For kernel_size=15, padding=7 maintains sequence length
        self.smooth = nn.Conv1d(dim_x, dim_x, kernel_size=15, padding=7, groups=dim_x)
        
        # Convolutional layers with increasing receptive fields
        # Calculate padding for each layer to maintain sequence length
        # For kernel_size=7:
        # - dilation=1: padding=3
        # - dilation=2: padding=6
        # - dilation=4: padding=12
        self.conv1 = nn.Conv1d(dim_x, 32, kernel_size=7, padding=3, dilation=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=7, padding=6, dilation=2)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=7, padding=12, dilation=4)
        
        # Final projection to latent space using 1x1 convolution
        self.proj = nn.Conv1d(128, dim_z, kernel_size=1, padding=0)  # 1x1 convolution maintains sequence length
        
        # Layer normalization for output
        self.normalize_output = normalize_output
        if normalize_output:
            self.layer_norm = nn.LayerNorm(dim_z)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Conv1d, nn.Linear)):
            if isinstance(module, nn.Conv1d) and module.groups == module.in_channels:
                # Initialize smoothing layer with Gaussian-like weights
                kernel_size = module.kernel_size[0]
                sigma = kernel_size / 6.0
                x = torch.arange(-(kernel_size//2), kernel_size//2 + 1)
                weights = torch.exp(-(x**2) / (2*sigma**2))
                weights = weights / weights.sum()
                module.weight.data = weights.view(1, 1, -1).repeat(module.in_channels, 1, 1)
            else:
                nn.init.xavier_uniform_(module.weight, gain=0.1)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def compute_smoothness_loss(self, z):
        """Compute temporal smoothness loss using finite differences"""
        # Compute first-order differences
        diff = z[:, 1:] - z[:, :-1]
        # L2 loss on differences
        smoothness_loss = torch.mean(diff**2)
        return smoothness_loss
    
    def forward(self, x):
        # Input shape: (batch_size, T, dim_x)
        # Reshape for convolutions: (batch_size, dim_x, T)
        x = x.transpose(1, 2)
        
        # Apply initial temporal smoothing
        x = F.relu(self.smooth(x))
        
        # Apply convolutions with ReLU activations
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        
        # Project to latent space using 1x1 convolution
        z = self.proj(h)  # (batch_size, dim_z, T)
        
        # Transpose back to (batch_size, T, dim_z)
        z = z.transpose(1, 2)
        
        # Apply layer normalization if enabled
        if self.normalize_output:
            z = self.layer_norm(z)
        
        # Sample from recognition distribution if enabled
        entropy = 0
        if self.sample_rec:
            # Add small noise for sampling
            z = z + torch.randn_like(z) * 0.1
            # Compute entropy (constant for Gaussian)
            entropy = 0.5 * torch.log(2 * torch.pi * torch.tensor(0.01)) * z.shape[1]
        
        # Compute smoothness loss
        smoothness_loss = self.compute_smoothness_loss(z)
        
        # Add smoothness loss to entropy (which is used in the total loss)
        entropy = entropy + self.smoothness_weight * smoothness_loss
        
        return z, entropy


# CNN encoder for ordinal data, dealing with missing values
class StackedConvolutions_ordinal(nn.Module):
    def __init__(self, dim_x, dim_z, use_ordinal_data=False, kernel_size=None, stride=None, padding=None, num_convs=(3, 1)):
        super(StackedConvolutions_ordinal, self).__init__()

        if padding is None:
            padding = [5, 3, 2, 1]
        if stride is None:
            stride = [1]
        if kernel_size is None:
            kernel_size = [11, 7, 5, 3]
            
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.dim_h = dim_z
        dim_q = dim_x

        if use_ordinal_data:
            self.multimodal = True
        else:
            self.multimodal = False
            self.dim_h = dim_z

        assert (len(kernel_size) == num_convs[0] + num_convs[1])
        assert (len(kernel_size) == len(stride) or len(stride) == 1)
        assert (len(kernel_size) == len(padding) or len(padding) == 1)

        if len(stride) == 1:
            stride *= len(kernel_size)
        if len(padding) == 1:
            padding *= len(kernel_size)

        mean_convs = []

        for i in range(num_convs[0]):
            mean_convs.append(nn.ReflectionPad1d((2 * padding[i], 0)))
            mean_convs.append(nn.Conv1d(
                in_channels=dim_x if i == 0 else self.dim_h,
                out_channels=self.dim_h,
                kernel_size=kernel_size[i],
                stride=stride[i],
                padding=0
            ))
        self.mean_conv = nn.Sequential(*mean_convs)

        logvar_convs = []
        for i in range(num_convs[1]):
            logvar_convs.append(nn.ReflectionPad1d((2 * padding[i], 0)))
            logvar_convs.append(nn.Conv1d(
                in_channels=self.dim_x if i == 0 else self.dim_h,
                out_channels=self.dim_h,
                kernel_size=kernel_size[i],
                stride=stride[i],
                padding=0
            ))
        self.logvar_conv = nn.Sequential(*logvar_convs)

        if self.multimodal:
            mean_q_convs = []

            for i in range(num_convs[0]):
                mean_q_convs.append(nn.ReflectionPad1d((2 * padding[i], 0)))
                mean_q_convs.append(nn.Conv1d(
                    in_channels=dim_q if i == 0 else self.dim_h,
                    out_channels=self.dim_h,
                    kernel_size=kernel_size[i],
                    stride=stride[i],
                    padding=0
                ))
            self.mean_q_conv = nn.Sequential(*mean_q_convs)

            logvar_q_convs = []
            for i in range(num_convs[1]):
                logvar_q_convs.append(nn.ReflectionPad1d((2 * padding[i], 0)))
                logvar_q_convs.append(nn.Conv1d(
                    in_channels=dim_q if i == 0 else self.dim_h,
                    out_channels=self.dim_h,
                    kernel_size=kernel_size[i],
                    stride=stride[i],
                    padding=0
                ))
            self.logvar_q_conv = nn.Sequential(*logvar_q_convs)

            self.mean_concat_nn = nn.Linear(2 * self.dim_h, self.dim_h)
            self.logvar_concat_nn = nn.Linear(2 * self.dim_h, self.dim_h)

            self.mean_concat_h1 = nn.Linear(self.dim_h, dim_z)
            self.logvar_concat_h1 = nn.Linear(self.dim_h, dim_z)

    def replace_nan_with_zero(self, x):
        return torch.nan_to_num(x, nan=0)

    def get_sample(self, mean, log_sqrt_var):
        sample = mean + torch.exp(log_sqrt_var) * torch.randn(mean.shape[0], mean.shape[1], self.dim_z)
        return sample

    @staticmethod
    def get_entropy(log_sqrt_var):
        entropy = torch.sum(log_sqrt_var) / log_sqrt_var.shape[0]
        return entropy

    def forward(self, x, q=None, sampling=False):
        """
        Forward pass for batched data.
        
        Args:
            x: Input tensor of shape (batch_size, T, dim_x)
            q: Optional second input tensor of shape (batch_size, T, dim_q) for multimodal case
            sampling: Whether to sample from the distribution
            
        Returns:
            If sampling=True: (sample, entropy)
            If sampling=False: (mean, entropy)
        """
        # Handle NaN values by forward-filling (carry forward last valid value)
        x_clean = x.clone()
        for b in range(x_clean.shape[0]):
            for d in range(x_clean.shape[2]):
                # Forward fill NaN values
                mask = torch.isnan(x_clean[b, :, d])
                if mask.any():
                    # Find valid values and their indices
                    valid_mask = ~mask
                    if valid_mask.any():
                        # Forward fill: each NaN gets the last valid value
                        valid_values = x_clean[b, valid_mask, d]
                        valid_indices = torch.where(valid_mask)[0]
                        
                        # For each position, find the last valid value
                        for i in range(x_clean.shape[1]):
                            if mask[i]:
                                # Find the last valid index before i
                                last_valid_idx = valid_indices[valid_indices < i]
                                if len(last_valid_idx) > 0:
                                    x_clean[b, i, d] = x_clean[b, last_valid_idx[-1], d]
                                else:
                                    # If no previous valid value, use 0
                                    x_clean[b, i, d] = 0.0
        
        # Input is already batched: (batch_size, T, dim_x)
        # Transpose to (batch_size, dim_x, T) for convolutions
        x_batched = x_clean.transpose(1, 2)
        
        mean = self.mean_conv(x_batched)
        log_sqrt_var = self.logvar_conv(x_batched)

        if self.multimodal:
            if q is None:
                q = x  # Use x as q if not provided
            q_batched = q.transpose(1, 2)

            mean_q = self.mean_q_conv(q_batched)
            log_sqrt_var_q = self.logvar_q_conv(q_batched)

            # Transpose back to (batch_size, T, dim_h)
            mean_concat = torch.cat([mean.transpose(1, 2), mean_q.transpose(1, 2)], dim=2)
            log_sqrt_var_concat = torch.cat([log_sqrt_var.transpose(1, 2), log_sqrt_var_q.transpose(1, 2)], dim=2)

            mean = F.relu(self.mean_concat_nn(mean_concat))
            mean = self.mean_concat_h1(mean)

            log_sqrt_var = F.relu(self.logvar_concat_nn(log_sqrt_var_concat))
            log_sqrt_var = self.logvar_concat_h1(log_sqrt_var)
        else:
            # Transpose back to (batch_size, T, dim_z)
            mean = mean.transpose(1, 2)
            log_sqrt_var = log_sqrt_var.transpose(1, 2)

        if sampling:
            sample = self.get_sample(mean, log_sqrt_var)
            entropy = self.get_entropy(log_sqrt_var)
            return sample, entropy
        else:
            entropy = torch.zeros(1, device=x.device)
            return mean, entropy
