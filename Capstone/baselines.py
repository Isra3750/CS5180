# baselines.py -- Stable-Baselines3 baselines (TD3, DDPG, SAC, PPO, A2C)

from pathlib import Path

import numpy as np
import pandas as pd

from td3_algorithm import device
from config import _subdir, _log_row, _is_complete
from environment import make_env, reset_env, step_env, get_env_dims

SB3_OFF_POLICY = {"TD3", "DDPG", "SAC"} # these algo use a replay buffer
SB3_SUPPORTED = ["TD3", "DDPG", "SAC", "PPO", "A2C"]


# Import SB3 lazily, so this module still imports even when SB3 isn't installed
def _sb3_classes():
    from stable_baselines3 import TD3, DDPG, SAC, PPO, A2C
    return {"TD3": TD3, "DDPG": DDPG, "SAC": SAC, "PPO": PPO, "A2C": A2C}


# Construct an SB3 model, matching our TD3's hyperparameters where they apply
def _build_sb3_model(algo, env, cfg, seed, action_dim, max_action):
    from stable_baselines3.common.noise import NormalActionNoise
    Cls = _sb3_classes()[algo]

    # On-policy PPO / A2C: no replay buffer -> SB3 recommends CPU for small MLP policies
    if algo not in SB3_OFF_POLICY:
        return Cls("MlpPolicy", env, learning_rate=cfg.actor_lr, gamma=cfg.discount,
                   policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
                   seed=seed, device="cpu", verbose=0)

    # Off-policy TD3 / DDPG / SAC: shared replay-buffer + update settings
    kwargs = dict(
        learning_rate=cfg.actor_lr, buffer_size=cfg.replay_buffer_size,
        learning_starts=cfg.start_timesteps, batch_size=cfg.batch_size,
        tau=cfg.tau, gamma=cfg.discount, train_freq=(1, "step"), gradient_steps=1,
        policy_kwargs=dict(net_arch=[256, 256]), seed=seed, device=device.type, verbose=0,
    )
    # TD3 / DDPG explore with Gaussian action noise (note that SAC explores via its entropy term instead)
    if algo in ("TD3", "DDPG"):
        kwargs["action_noise"] = NormalActionNoise(
            np.zeros(action_dim), cfg.expl_noise * max_action * np.ones(action_dim))

    # TD3-only: target-policy smoothing and delayed updates
    if algo == "TD3":
        kwargs["policy_delay"] = cfg.policy_freq
        kwargs["target_policy_noise"] = cfg.policy_noise * max_action
        kwargs["target_noise_clip"] = cfg.noise_clip * max_action
    return Cls("MlpPolicy", env, **kwargs)


# Average deterministic return of an SB3 model over eval_episodes (mirrors evaluate_policy)
def _sb3_eval(model, env_id, seed, eval_episodes):
    eval_env = make_env(env_id, seed + 100)
    total = 0.0
    for episode in range(eval_episodes):
        obs = reset_env(eval_env, seed + 100 + episode)
        done = False
        while not done:
            action, _ = model.predict(np.asarray(obs), deterministic=True)
            obs, reward, done, _, _ = step_env(eval_env, action)
            total += reward
    eval_env.close()
    return float(total / eval_episodes)


# Train one SB3 baseline per seed, we log eval returns in our CSV schema
def run_sb3_multi_seed(base_cfg, seeds, algo="TD3", save_snapshots=False, verbose=True):
    algo = algo.upper()
    if algo not in SB3_SUPPORTED:
        raise ValueError(f"Unsupported SB3 algo {algo}; choose from {SB3_SUPPORTED}")
    algo_name = f"SB3-{algo}"
    logs_dir = _subdir(base_cfg.results_dir, base_cfg.env_id, "logs")
    models_dir = _subdir(base_cfg.results_dir, base_cfg.env_id, "models")

    dfs = []
    for seed in seeds:
        run_name = f"{algo_name}_{base_cfg.env_id}_seed{seed}"
        csv_path = logs_dir / f"{run_name}.csv"
        # Reload a finished run instead of retraining it
        if _is_complete(csv_path, base_cfg.max_timesteps):
            print(f"[skip] {run_name}: complete checkpoint found, loading.")
            dfs.append(pd.read_csv(csv_path))
            continue

        env = make_env(base_cfg.env_id, seed)
        _, action_dim, max_action = get_env_dims(env)
        model = _build_sb3_model(algo, env, base_cfg, seed, action_dim, max_action)
        print(f"Training {run_name} ...")

        rows = []
        # Eval the untrained model at t=0
        rows.append(_log_row(algo_name, base_cfg.env_id, seed, 0,
                             eval_return=_sb3_eval(model, base_cfg.env_id, seed, base_cfg.eval_episodes)))

        pending_snapshots = base_cfg.default_snapshot_steps() if save_snapshots else []
        steps_done = 0
        first_chunk = True

        # Train in eval_freq-sized chunks; evaluate (and maybe snapshot) after each chunk
        while steps_done < base_cfg.max_timesteps:
            chunk = min(base_cfg.eval_freq, base_cfg.max_timesteps - steps_done)

            # reset_num_timesteps only on the first chunk, so SB3's step counter keeps running
            model.learn(chunk, reset_num_timesteps=first_chunk, progress_bar=False)
            first_chunk = False
            steps_done += chunk

            val = _sb3_eval(model, base_cfg.env_id, seed, base_cfg.eval_episodes)
            rows.append(_log_row(algo_name, base_cfg.env_id, seed, steps_done, eval_return=val))
            if verbose:
                print(f"[{algo_name}] Eval | T={steps_done:>8} | Return={val:>10.3f}")

            # Once we cross a milestone, drop the milestones we passed and snapshot the model
            if pending_snapshots and steps_done >= pending_snapshots[0]:
                while pending_snapshots and steps_done >= pending_snapshots[0]:
                    pending_snapshots.pop(0)
                model.save(str(models_dir / f"{run_name}_step{steps_done}"))

        env.close()
        model.save(str(models_dir / run_name)) # final model
        df_seed = pd.DataFrame(rows)
        df_seed.to_csv(csv_path, index=False)
        print(f"Saved {algo_name} logs to: {csv_path}")
        dfs.append(df_seed)
    return pd.concat(dfs, ignore_index=True)


# Run several SB3 baselines and concatenate their eval logs
# snapshot_algos names the algos that also save milestone snapshots (for progress videos)
def run_all_baselines(base_cfg, seeds, algos=("TD3", "DDPG", "SAC", "PPO"),
                      snapshot_algos=("DDPG",)):
    snapshot_set = {a.upper() for a in snapshot_algos}
    frames = []
    for algo in algos:
        save_snapshots = algo.upper() in snapshot_set
        frames.append(run_sb3_multi_seed(base_cfg, seeds, algo=algo, save_snapshots=save_snapshots))
    return pd.concat(frames, ignore_index=True)
