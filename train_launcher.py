import os
import subprocess
from multiprocessing import Pool

# Hyperparameter grid
Ms = [4]
Ps = [1]
N_feats = [12]

# Model type to train
model_type = 'hierarchical'  # 'hierarchical' or 'non_hierarchical'

n_threads_per_run = 1  # Limit threads per run
n_parallel = 10         # Number of parallel runs
runs_per_config = 10    # Number of runs per configuration
starting_run_id = 1    # Starting run ID (change this to continue from a specific run)

output_dir = 'results_M_P1'
# Number of epochs and batch size (can be changed as needed)
n_epochs = 3000
batch_size = 64

def launch(args):
    M, P, N_feat, run_id = args
    env = os.environ.copy()
    # Limit threads for each subprocess
    env["OMP_NUM_THREADS"] = str(n_threads_per_run)
    env["OPENBLAS_NUM_THREADS"] = str(n_threads_per_run)
    env["MKL_NUM_THREADS"] = str(n_threads_per_run)
    env["NUMEXPR_NUM_THREADS"] = str(n_threads_per_run)
    env["VECLIB_MAXIMUM_THREADS"] = str(n_threads_per_run)
    env["TORCH_NUM_THREADS"] = str(n_threads_per_run)
    
    # Set up run directory based on model type
    if model_type == 'hierarchical':
        run_dir = os.path.join(output_dir, f"M{M}_P{P}_Nfeat{N_feat}")
    else:  # non_hierarchical
        run_dir = os.path.join(output_dir, f"M{M}_P{P}")
    os.makedirs(run_dir, exist_ok=True)
    
    cmd = [
        "python", "train_model.py",
        "--M", str(M),
        "--P", str(P),
        "--model_type", model_type,
        "--output_dir", output_dir,
        "--n_epochs", str(n_epochs),
        "--batch_size", str(batch_size),
        "--run_id", str(run_id)
    ]
    
    # Add N_feat parameter only for hierarchical models
    if model_type == 'hierarchical':
        cmd.extend(["--N_feat", str(N_feat)])
    
    print(f"Launching: {' '.join(cmd)}")
    return subprocess.call(cmd, env=env)

if __name__ == "__main__":
    # Generate jobs with multiple runs per config
    jobs = []
    if model_type == 'hierarchical':
        for M in Ms:
            for P in Ps:
                for N_feat in N_feats:
                    for run_idx in range(runs_per_config):
                        # Use starting_run_id + run_idx for sequential run IDs
                        run_id = starting_run_id + run_idx
                        jobs.append((M, P, N_feat, run_id))
        total_configs = len(Ms) * len(Ps) * len(N_feats)
    else:  # non_hierarchical
        for M in Ms:
            for P in Ps:
                for run_idx in range(runs_per_config):
                    # Use starting_run_id + run_idx for sequential run IDs
                    run_id = starting_run_id + run_idx
                    # For non-hierarchical, N_feat is not used, so we pass 0 as placeholder
                    jobs.append((M, P, 0, run_id))
        total_configs = len(Ms) * len(Ps)
    
    print(f"Total jobs to run: {len(jobs)}")
    print(f"Configurations: {total_configs}")
    print(f"Runs per config: {runs_per_config}")
    print(f"Model type: {model_type}")
    print(f"Starting from run_id: {starting_run_id}")
    
    with Pool(processes=n_parallel) as pool:
        pool.map(launch, jobs) 