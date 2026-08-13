# config.py -> Experiment configuration and shared results-IO helpers
from pathlib import Path

import numpy as np
import pandas as pd

# TD3 max-average-return from paper with v1 results - for reference
PAPER_TD3_MAX_RETURN = {
    "HalfCheetah": 9636.95, "Hopper": 3564.07, "Walker2d": 4682.82, "Ant": 4372.44,
    "Reacher": -3.60, "InvertedPendulum": 1000.00, "InvertedDoublePendulum": 9337.47,
}

# Look up the paper TD3 max-return for an env id, matches on its base name
# ex. 'Hopper-v5' -> 'Hopper', used in plotting.py
def paper_reference(env_id):
    return PAPER_TD3_MAX_RETURN.get(env_id.split("-")[0])


# All hyperparameter for one experiment. Defaults match the paper TD3 code
# Overwrite as needed
class ExperimentConfig:
    def __init__(self, env_id="HalfCheetah-v5", algorithm="TD3", seed=0,
                 max_timesteps=1_000_000, start_timesteps=25_000, eval_freq=5_000,
                 eval_episodes=10, expl_noise=0.1, batch_size=256, discount=0.99,
                 tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2,
                 replay_buffer_size=1_000_000, actor_lr=3e-4, critic_lr=3e-4,
                 results_dir="results", save_model=False, snapshot_steps=None,
                 use_cdq=True):
        # What to run and how long
        self.env_id = env_id
        self.algorithm = algorithm # label used for logging / file paths
        self.seed = seed
        self.max_timesteps = max_timesteps
        self.start_timesteps = start_timesteps # random warm-up steps before learning

        # Evaluation cadence
        self.eval_freq = eval_freq
        self.eval_episodes = eval_episodes

        # TD3 / optimisation hyperparameters
        self.expl_noise = expl_noise # Gaussian exploration noise std (x max_action)
        self.batch_size = batch_size
        self.discount = discount
        self.tau = tau # target-network soft-update rate
        self.policy_noise = policy_noise # target-smoothing noise std (x max_action)
        self.noise_clip = noise_clip # clip range for that noise (x max_action)
        self.policy_freq = policy_freq # delayed policy-update period
        self.replay_buffer_size = replay_buffer_size
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr

        # Output stuff
        self.results_dir = results_dir
        self.save_model = save_model
        self.snapshot_steps = snapshot_steps # steps to snapshot for progress videos
        self.use_cdq = use_cdq # clipped double-Q

    # Make writing 'cfg' in a notebook lists the entire configuration
    def __repr__(self):
        fields = ", ".join(f"{k}={v!r}" for k, v in vars(self).items())
        return f"ExperimentConfig({fields})"

    # Milestone steps at which to snapshot the model (used to build progress videos)
    def default_snapshot_steps(self):
        if self.snapshot_steps is not None:
            steps = [int(s) for s in self.snapshot_steps]
        else:
            steps = [self.start_timesteps, self.max_timesteps // 10,
                     self.max_timesteps // 2, self.max_timesteps]
        valid = {s for s in steps if 0 < s <= self.max_timesteps}
        return sorted(valid)


# Builds and creates a results folder
def _subdir(results_dir, env_id, kind):
    path = Path(results_dir) / env_id / kind
    path.mkdir(parents=True, exist_ok=True)   # build the folder tree if it's missing
    return path


# Creates one standardized result record
# Build one results row with a fixed set of columns, so every algorithm CSV lines up
def _log_row(algorithm, env_id, seed, timestep, eval_return=np.nan, episode=np.nan,
             episode_return=np.nan, critic_loss=np.nan, actor_loss=np.nan):
    return {"timestep": timestep, "episode": episode, "eval_return": eval_return,
            "episode_return": episode_return, "critic_loss": critic_loss,
            "actor_loss": actor_loss, "algorithm": algorithm, "env_id": env_id, "seed": seed}


# Checks whether a training run has already finished
def _is_complete(csv_path, max_timesteps):
    if not csv_path.exists():
        return False
    return int(pd.read_csv(csv_path)["timestep"].max()) >= max_timesteps
