import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from model import FullModel
from dataset import load_and_create_dataset
import os
import test_model  # Import the new functions
import argparse

def train_epoch(model, dataloader, optimizer, alpha, n_interleave, beta_pred, beta_enc, beta_cons, beta_ent):
    """
    Train for one epoch.
    
    Args:
        model: FullModel instance
        dataloader: DataLoader providing batches of data
        optimizer: Optimizer for updating model parameters
        alpha: Teacher forcing parameter
        n_interleave: Teacher forcing frequency
        beta_pred: Weight for prediction loss
        beta_enc: Weight for encoder loss
        beta_cons: Weight for consistency loss
        beta_ent: Weight for entropy loss
    """
    model.train()
    epoch_metrics = {
        'total_loss': 0,
        'prediction_loss': 0,
        'encoder_loss': 0,
        'consistency_loss': 0,
        'entropy_loss': 0,
        'log_lik_pred': 0,
        'log_lik_enc': 0,
        'kl_div': 0
    }
    
    n_batches = 0
    
    for batch_idx, batch in enumerate(dataloader):
        subject_ids = batch['subject_id']
        
        # Forward pass
        outputs = model(
            batch, subject_ids, alpha, n_interleave,
            beta_pred, beta_enc, beta_cons, beta_ent
        )
        
        # Backward pass
        optimizer.zero_grad()
        outputs['total_loss'].backward()
        optimizer.step()
        
        # Update metrics
        for k, v in outputs.items():
            if k != 'total_loss':
                epoch_metrics[k] += v
        epoch_metrics['total_loss'] += outputs['total_loss'].item()
        n_batches += 1
    
    # Average metrics
    for k in epoch_metrics:
        epoch_metrics[k] /= n_batches
    
    return epoch_metrics

def train_model(dataloader, model, training_params, dataset=None, validation_subjects=None, validation_interval=500, validation_results=None, run_dir=None, run_id=None):
    """
    Main training function with validation.
    Args:
        dataloader: DataLoader instance providing batches of data
        model: FullModel instance
        training_params: Dictionary of training parameters
        dataset: Dataset instance (for validation)
        validation_subjects: List of subject IDs (for validation)
        validation_interval: How often to run validation (epochs)
        validation_results: List to store validation results
        run_dir: Directory for this specific run's outputs
    """
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Set up parameter groups based on model type
    if model.model_type == 'hierarchical':
        param_groups = [
            {'params': [model.rnn.feature_vectors], 'lr': training_params['feature_lr']},
            {'params': [
                model.rnn.proj_A, 
                model.rnn.proj_W, 
                model.rnn.proj_h, 
                model.rnn.proj_z0,
                model.rnn.D,
                model.rnn.proj_C,
                model.R_z_param
            ], 'lr': training_params['projection_lr']},
            {'params': list(model.decoder.parameters()) + list(model.encoder.parameters()), 'lr': training_params['encoder_lr']}
        ]
    else:  # non_hierarchical
        param_groups = [
            {'params': [
                model.rnn.A_params, 
                model.rnn.W_params, 
                model.rnn.h_params, 
                model.rnn.z0_params,
                model.rnn.C_params if hasattr(model.rnn, 'C_params') else [],
                model.rnn.D,
                model.R_z_param
            ], 'lr': training_params['model_lr']},
            {'params': list(model.decoder.parameters()) + list(model.encoder.parameters()), 'lr': training_params['encoder_lr']}
        ]
    optimizer = torch.optim.Adam(param_groups)
    # Create checkpoint directory within the run directory
    checkpoint_dir = os.path.join(run_dir, 'checkpoints') if run_dir else 'checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    # Model ID based on model type
    if model.model_type == 'hierarchical':
        model_id = f"hierarchical_M{model.M}_P{model.P}_Nfeat{model.N_feat}_cats{model.num_categories}_decodermode_{model.decoder_mode}"
    else:  # non_hierarchical
        model_id = f"non_hierarchical_M{model.M}_P{model.P}_cats{model.num_categories}_decodermode_{model.decoder_mode}"
    
    if run_id:
        checkpoint_path = f'{checkpoint_dir}/{model_id}_run_{run_id}_best_model.pth'
    else:
        checkpoint_path = f'{checkpoint_dir}/{model_id}_best_model.pth'
    best_test_mae = float('inf')
    for epoch in range(training_params['n_epochs']):
        if training_params['use_alpha_scheduling']:
            progress = epoch / (training_params['n_epochs'] - 1)
            current_alpha = training_params['alpha_start'] + progress * (training_params['alpha_end'] - training_params['alpha_start'])
        else:
            current_alpha = training_params['alpha_start']
        metrics = train_epoch(
            model, dataloader, optimizer,
            current_alpha,
            training_params['n_interleave'],
            training_params['beta_pred'],
            training_params['beta_enc'],
            training_params['beta_cons'],
            training_params['beta_ent']
        )
        if (epoch + 1) % 100 == 0:
            print(f"\nEpoch {epoch + 1}/{training_params['n_epochs']}")
            print(f"Total Loss: {metrics['total_loss']:.4f}")
            print(f"Prediction Loss: {metrics['prediction_loss']:.4f}")
            print(f"Encoder Loss: {metrics['encoder_loss']:.4f}")
            print(f"Consistency Loss: {metrics['consistency_loss']:.4f}")
            print(f"Log Likelihood (pred): {metrics['log_lik_pred']:.4f}")
            print(f"Log Likelihood (enc): {metrics['log_lik_enc']:.4f}")
            print(f"KL Divergence: {metrics['kl_div']:.4f}")
        # Validation step
        if (epoch + 1) % validation_interval == 0 and dataset is not None and validation_subjects is not None and validation_results is not None:
            print(f"\nRunning validation at epoch {epoch + 1}...")
            results = test_model.generate_long_trajectories(model, dataset, alpha=0.0)
            val_metrics = test_model.evaluate_model_trajectories(
                results,
                subjects=validation_subjects,
                train_steps=dataset.train_time_steps,
                test_steps=dataset.test_time_steps,
                epoch=epoch + 1
            )
            validation_results.append(val_metrics)
            print(f"Validation mean train MAE: {val_metrics['mean_train_mae']:.4f}, mean test MAE: {val_metrics['mean_test_mae']:.4f}, mean test corr: {val_metrics['mean_test_corr']:.4f}")
            # Save model if test MAE is improved
            if val_metrics['mean_test_mae'] < best_test_mae:
                best_test_mae = val_metrics['mean_test_mae']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'metrics': metrics,
                    'model_config': {
                        'M': model.M,
                        'P': model.P,
                        'N': model.N,
                        'n_subjects': model.n_subjects,
                        'input_dim': model.input_dim,
                        'N_feat': model.N_feat,
                        'num_categories': model.num_categories,
                        'learn_initial_states': model.learn_initial_states,
                        'model_type': model.model_type,
                        'decoder_mode': model.decoder_mode
                    },
                    'training_config': training_params
                }, checkpoint_path)
                print(f"\nSaved best model to {checkpoint_path}")

def create_new_model(model_params, dataset):
    """Create a new model with the given parameters"""
    model = FullModel(
        M=model_params['M'],
        P=model_params['P'],
        N=1,  # Set to 1 for a single ordinal output variable
        n_subjects=dataset.num_subjects,
        input_dim=dataset.input_dim,
        N_feat=model_params.get('N_feat', 20),  # Only used for hierarchical
        fix_R_z=True,
        learn_initial_states=model_params['learn_initial_states'],
        num_categories=model_params['num_categories'],
        model_type=model_params.get('model_type', 'hierarchical'),
        decoder_mode=model_params.get('decoder_mode', 'shared')
    )
    
    return model

def load_model(checkpoint_path, dataset=None, learn_initial_states=None):
    """
    Load a model from checkpoint.
    
    Args:
        checkpoint_path: Path to the model checkpoint
        dataset: Optional dataset instance to get dimensions if not in checkpoint
        learn_initial_states: Whether to learn initial states (if None, uses checkpoint value)
    """
    checkpoint = torch.load(checkpoint_path)
    model_config = checkpoint['model_config']
    
    # Use checkpoint value for learn_initial_states if not specified
    if learn_initial_states is None:
        learn_initial_states = model_config.get('learn_initial_states', True)
    
    # Create model with current configuration
    model = FullModel(
        M=model_config['M'],
        P=model_config['P'],
        N=1,  # Set to 1 for a single ordinal output variable
        n_subjects=model_config.get('n_subjects', dataset.num_subjects if dataset else None),
        input_dim=model_config.get('input_dim', dataset.input_dim if dataset else 0),
        N_feat=model_config.get('N_feat', 20),
        fix_R_z=True,
        learn_initial_states=learn_initial_states,
        num_categories=model_config.get('num_categories', 5),
        model_type=model_config.get('model_type', 'hierarchical'),
        decoder_mode=model_config.get('decoder_mode', 'shared')
    )
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"\nLoaded model from checkpoint:")
    print(f"- Model type: {model.model_type}")
    print(f"- Latent dimension (M): {model.M}")
    print(f"- Number of positive units (P): {model.P}")
    print(f"- Output dimensions (N): {model.N}")
    print(f"- Number of subjects: {model.n_subjects}")
    print(f"- Input dimensions: {model.input_dim}")
    if model.model_type == 'hierarchical':
        print(f"- Feature dimensions (N_feat): {model.N_feat}")
    print(f"- Number of categories: {model.num_categories}")
    print(f"- Learn initial states: {model.learn_initial_states}")
    
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train AL-RNN model with validation (hierarchical or non-hierarchical).')
    parser.add_argument('--M', type=int, default=4, help='Latent dimension')
    parser.add_argument('--P', type=int, default=1, help='Number of positive units')
    parser.add_argument('--N_feat', type=int, default=10, help='Number of features for hierarchical RNN')
    parser.add_argument('--model_type', type=str, default='non_hierarchical', choices=['hierarchical', 'non_hierarchical'], 
                       help='Model type: hierarchical or non_hierarchical')
    parser.add_argument('--decoder_mode', type=str, default='individual', choices=['shared', 'individual', 'hierarchical'],
                       help='Decoder mode: shared, individual, or hierarchical')
    parser.add_argument('--output_dir', type=str, default='results', help='Base output directory for results')
    parser.add_argument('--n_epochs', type=int, default=5000, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--run_id', type=str, default=None, help='Unique run ID for this training run')
    parser.add_argument('--starting_run_id', type=int, default=1, help='Starting run ID (for launcher compatibility)')
    args = parser.parse_args()

    # Model parameters
    model_params = {
        'M': args.M,
        'P': args.P,
        'N_feat': args.N_feat,
        'learn_initial_states': True,  # whether to learn initial states
        'num_categories': 5,  # number of categories for cumulative link decoder
        'model_type': args.model_type,
        'decoder_mode': args.decoder_mode,
    }
    # Training parameters
    training_params = {
        'n_epochs': args.n_epochs,
        'batch_size': args.batch_size,
        'feature_lr': 8e-4,
        'projection_lr': 1e-4,
        'model_lr': 1e-4,  # Learning rate for non-hierarchical model parameters
        'encoder_lr': 1e-3,
        'alpha_start': 0.0,
        'alpha_end': 0.0,
        'use_alpha_scheduling': False,
        'n_interleave': 1,
        'beta_pred': 0.01,
        'beta_enc': 0.01,
        'beta_cons': 0.1,
        'beta_ent': 0.0,
    }

    # Output directory structure based on model type
    if args.model_type == 'hierarchical':
        base_run_dir = os.path.join(args.output_dir, f"M{args.M}_P{args.P}_Nfeat{args.N_feat}")
    else:  # non_hierarchical
        base_run_dir = os.path.join(args.output_dir, f"M{args.M}_P{args.P}")
    
    if args.run_id:
        run_dir = os.path.join(base_run_dir, f"run_{args.run_id}")
    else:
        run_dir = base_run_dir
    os.makedirs(run_dir, exist_ok=True)

    # Data parameters
    data_path = "data/stacked_invests.npy"
    inputs_path = "data/stacked_inputs.npy"
    load_existing_model = False  # Toggle this to load existing model
    
    if args.run_id:
        print(f"\n=== Starting {args.model_type} training run {args.run_id} ===")
        print(f"Model config: M={args.M}, P={args.P}")
        if args.model_type == 'hierarchical':
            print(f"N_feat={args.N_feat}")
        print(f"Output directory: {run_dir}")
    
    print(f"\nLoading data from {data_path} and {inputs_path}")
    dataset, dataloader = load_and_create_dataset(
        data_path=data_path,
        inputs_path=inputs_path,
        batch_size=training_params['batch_size'],
        shuffle=True
    )
    # Create or load model
    if load_existing_model:
        if args.model_type == 'hierarchical':
            if args.run_id:
                checkpoint_path = os.path.join(run_dir, 'checkpoints', f'hierarchical_M{model_params["M"]}_P{model_params["P"]}_Nfeat{model_params["N_feat"]}_cats{model_params["num_categories"]}_decodermode_{model_params["decoder_mode"]}_run_{args.run_id}_best_model.pth')
            else:
                checkpoint_path = os.path.join(run_dir, 'checkpoints', f'hierarchical_M{model_params["M"]}_P{model_params["P"]}_Nfeat{model_params["N_feat"]}_cats{model_params["num_categories"]}_decodermode_{model_params["decoder_mode"]}_best_model.pth')
        else:  # non_hierarchical
            if args.run_id:
                checkpoint_path = os.path.join(run_dir, 'checkpoints', f'non_hierarchical_M{model_params["M"]}_P{model_params["P"]}_cats{model_params["num_categories"]}_decodermode_{model_params["decoder_mode"]}_run_{args.run_id}_best_model.pth')
            else:
                checkpoint_path = os.path.join(run_dir, 'checkpoints', f'non_hierarchical_M{model_params["M"]}_P{model_params["P"]}_cats{model_params["num_categories"]}_decodermode_{model_params["decoder_mode"]}_best_model.pth')
        print(f"\nLoading model from {checkpoint_path}")
        model = load_model(checkpoint_path, dataset)
    else:
        print(f"\nCreating new {args.model_type} model")
        model = create_new_model(model_params, dataset)

    # Validation setup
    validation_subjects = list(range(dataset.num_subjects))
    validation_results = []
    # Train the model with validation
    train_model(
        dataloader, model, training_params,
        dataset=dataset,
        validation_subjects=validation_subjects,
        validation_interval=max(1, args.n_epochs // 10),  # 10 checkpoints per run
        validation_results=validation_results,
        run_dir=run_dir,
        run_id=args.run_id
    )
    # Save validation results, training and model params to JSON after training
    results_dict = {
        'model_params': model_params,
        'training_params': training_params,
        'validation_results': validation_results,
        'run_id': args.run_id
    }
    if args.run_id:
        out_json = os.path.join(run_dir, f'validation_metrics_run_{args.run_id}.json')
    else:
        out_json = os.path.join(run_dir, 'validation_metrics.json')
    test_model.save_results_to_json(results_dict, out_json)
    
    if args.run_id:
        print(f"\n=== Completed {args.model_type} training run {args.run_id} ===")
        print(f"Results saved to: {out_json}")
        print(f"Best model saved to: {run_dir}/checkpoints/") 