import torch
import torch.nn as nn
import torch.nn.functional as F
from hierarchical_al_rnn import HierarchicalAL_RNN, predict_sequence_using_gtf, latent_likelihood
from non_hierarchical_al_rnn import NonHierarchicalAL_RNN
from decoders import get_decoder
from encoder import StackedConvolutions_ordinal
from hierarchical_decoder import HierarchicalDecoderCumulativeLink

class FullModel(nn.Module):
    def __init__(self, M, P, N, n_subjects, input_dim, N_feat=20, fix_R_z=False, 
                 learn_initial_states=True, num_categories=5, model_type='hierarchical', decoder_mode='shared'):
        """
        Full model using AL-RNN with cumulative link decoder.
        Supports both hierarchical and non-hierarchical versions.
        Args:
            M: Latent dimension
            P: Number of positive units
            N: Number of output dimensions (data dimensions)
            n_subjects: Number of subjects/trials
            input_dim: Dimension of external inputs
            N_feat: Number of features for hierarchical RNN (only used if model_type='hierarchical')
            fix_R_z: Whether to fix R_z parameter
            learn_initial_states: Whether to learn initial states
            num_categories: Number of categories for cumulative link decoder
            model_type: 'hierarchical' or 'non_hierarchical'
            decoder_mode: 'shared', 'individual', or 'hierarchical'
        """
        super(FullModel, self).__init__()
        self.M = M
        self.P = P
        self.N = N
        self.n_subjects = n_subjects
        self.input_dim = input_dim
        self.N_feat = N_feat
        self.learn_initial_states = learn_initial_states
        self.num_categories = num_categories
        self.model_type = model_type
        self.decoder_mode = decoder_mode
        
        # Initialize RNN based on model type
        if model_type == 'hierarchical':
            self.rnn = HierarchicalAL_RNN(
                M=self.M, 
                P=self.P, 
                N_feat=self.N_feat,
                n_trials=self.n_subjects,
                input_dim=self.input_dim,
                use_inputs=True,  # Always use inputs
                learn_initial_states=learn_initial_states
            )
        elif model_type == 'non_hierarchical':
            self.rnn = NonHierarchicalAL_RNN(
                M=self.M, 
                P=self.P, 
                n_subjects=self.n_subjects,
                input_dim=self.input_dim,
                use_inputs=True,  # Always use inputs
                learn_initial_states=learn_initial_states
            )
        else:
            raise ValueError(f"model_type must be 'hierarchical' or 'non_hierarchical', got {model_type}")
        
        # Decoder selection
        if decoder_mode in ['shared', 'individual', 'hierarchical']:
            self.decoder = HierarchicalDecoderCumulativeLink(
                dz=self.M,
                dq=self.N,
                num_categories=num_categories,
                n_subjects=self.n_subjects,
                decoder_mode=decoder_mode,
                feature_dim=self.N_feat if decoder_mode == 'hierarchical' else None
            )
        else:
            self.decoder = get_decoder('cumulative_link', dz=self.M, dq=self.N, num_categories=num_categories)
        self.encoder = StackedConvolutions_ordinal(dim_x=N, dim_z=self.M, use_ordinal_data=False)
        
        # Initialize R_z as a learnable parameter
        self.register_buffer('R_z', torch.ones(self.M))
        self.R_z_param = nn.Parameter(torch.ones(self.M))
        self.fix_R_z = fix_R_z
        
        print(f"FullModel initialized ({model_type}, decoder_mode={decoder_mode}):")
        print(f"- Latent dimension (M): {self.M}")
        print(f"- Number of positive units (P): {self.P}")
        print(f"- Output dimensions (N): {self.N}")
        print(f"- Number of subjects: {self.n_subjects}")
        print(f"- Input dimensions: {self.input_dim}")
        if model_type == 'hierarchical':
            print(f"- Feature dimensions (N_feat): {self.N_feat}")
        print(f"- Number of categories: {self.num_categories}")
        print(f"- Learn initial states: {self.learn_initial_states}")
        print(f"- Decoder mode: {self.decoder_mode}")
    
    def _get_decoder_feature_vectors(self, subject_ids):
        if self.decoder_mode == 'hierarchical' and hasattr(self.rnn, 'feature_vectors'):
            return self.rnn.feature_vectors[subject_ids]
        return None
    
    def forward(self, batch, subject_ids, alpha, n_interleave, beta_pred, beta_enc, beta_cons, beta_ent=0.1):
        # Update R_z from parameter if not fixed
        if not self.fix_R_z:
            self.R_z.copy_(self.R_z_param)
        data = batch['data']  # (batch_size, T, N)
        inputs = batch['inputs']  # (batch_size, T, input_dim)
        initial_states = self.rnn.get_initial_state(subject_ids)
        encoded_x, entropy = self.encoder(data, sampling=False)
        z_lat, logits = predict_sequence_using_gtf(
            self.rnn, initial_states, inputs,
            encoded_x, alpha, n_interleave, subject_ids
        )
        feature_vectors = self._get_decoder_feature_vectors(subject_ids)
        log_lik_pred = self.decoder.log_likelihood(data, z_lat, subject_ids=subject_ids, feature_vectors=feature_vectors)
        log_lik_enc = self.decoder.log_likelihood(data, encoded_x, subject_ids=subject_ids, feature_vectors=feature_vectors)
        kl_div = latent_likelihood(z_lat, encoded_x, self.R_z)
        log_lik_pred = log_lik_pred.mean()
        log_lik_enc = log_lik_enc.mean()
        kl_div = kl_div.mean()
        loss_pred = -beta_pred * log_lik_pred
        loss_enc = -beta_enc * log_lik_enc
        loss_cons = -beta_cons * kl_div
        loss_ent = beta_ent * entropy
        total_loss = loss_pred + loss_enc + loss_cons + loss_ent
        return {
            'total_loss': total_loss,
            'prediction_loss': loss_pred.item(),
            'encoder_loss': loss_enc.item(),
            'consistency_loss': loss_cons.item(),
            'entropy_loss': loss_ent,
            'log_lik_pred': log_lik_pred.item(),
            'log_lik_enc': log_lik_enc.item(),
            'kl_div': kl_div.item()
        }
    
    def fix_R_z_gradients(self, fix=True):
        """Toggle whether R_z should be updated during training"""
        self.fix_R_z = fix
        if fix:
            self.R_z_param.requires_grad_(False)
        else:
            self.R_z_param.requires_grad_(True)
    
    @torch.no_grad()
    def predict(self, batch, subject_ids=None, alpha=0.0, n_interleave=1):
        """
        Generate predictions from input data
        
        Args:
            batch: Dictionary containing:
                - data: Data tensor (batch_size, T, N)
                - inputs: External inputs (batch_size, T, input_dim)
            subject_ids: Tensor of subject IDs (batch_size,) (if None, uses all subjects)
            alpha: Teacher forcing parameter
            n_interleave: Teacher forcing interval
        """
        self.eval()
        
        data = batch['data']
        inputs = batch['inputs']
        
        if subject_ids is None:
            subject_ids = torch.arange(data.shape[0], device=data.device)
        
        initial_states = self.rnn.get_initial_state(subject_ids)
        encoded_x, _ = self.encoder(data, sampling=False)
        
        z_hat, _ = predict_sequence_using_gtf(
            self.rnn, initial_states, inputs,
            encoded_x, alpha, n_interleave, subject_ids
        )
        feature_vectors = self._get_decoder_feature_vectors(subject_ids)
        predictions = self.decoder.forward(z_hat, subject_ids=subject_ids, feature_vectors=feature_vectors)
        return predictions, z_hat 