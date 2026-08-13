# training.py -> Training driver for TD3 (paper main.py)
# Builds the agent, runs the env loop with warm-up + exploration noise, then evaluates
# Also periodically, writes a per-seed CSV and milestone model snapshots, and resumes finished seeds
import copy
import time
from pathlib import Path

import numpy as np
import pandas as pd

from td3_algorithm import device, seed_everything, ReplayBuffer, TD3Agent
from config import ExperimentConfig, _subdir, _log_row, _is_complete
from environment import make_env, reset_env, step_env, get_env_dims


# Build a TD3Agent from a config, scaling target/exploration noise by the max_action
def make_agent(algorithm, state_dim, action_dim, max_action, cfg):
    return TD3Agent(
        state_dim, action_dim, max_action,
        discount=cfg.discount, tau=cfg.tau,
        policy_noise=cfg.policy_noise * max_action,
        noise_clip=cfg.noise_clip * max_action,
        policy_freq=cfg.policy_freq, actor_lr=cfg.actor_lr, critic_lr=cfg.critic_lr,
        use_cdq=cfg.use_cdq,
    )


# Average return over deterministic episodes on a separate eval env
def evaluate_policy(agent, env_id, seed, eval_episodes=10):
    eval_env = make_env(env_id, seed + 100)
    total = 0.0
    for episode in range(eval_episodes):
        # Distinct seed per episode -> different start states, but reproducible across re-runs
        state = reset_env(eval_env, seed + 100 + episode)
        done = False
        while not done:
            action = agent.select_action(np.asarray(state)) # deterministic (no exploration noise)
            state, reward, done, _, _ = step_env(eval_env, action)
            total += reward
    eval_env.close()
    return float(total / eval_episodes)


# Make path prefix for a TD3 model snapshot at a given step
def _snapshot_path(cfg, seed, step):
    models = _subdir(cfg.results_dir, cfg.env_id, "models")
    return str(models / f"TD3_{cfg.env_id}_seed{seed}_step{step}")


# Train one TD3 seed fully -> also write the eval-log CSV and milestone model snapshots
def train_one_run(cfg):
    seed_everything(cfg.seed)
    env = make_env(cfg.env_id, cfg.seed)
    state_dim, action_dim, max_action = get_env_dims(env)

    agent = make_agent(cfg.algorithm, state_dim, action_dim, max_action, cfg)

    replay_buffer = ReplayBuffer(state_dim, action_dim, max_size=cfg.replay_buffer_size)

    run_name = f"{cfg.algorithm}_{cfg.env_id}_seed{cfg.seed}"
    snapshot_steps = cfg.default_snapshot_steps()
    print(f"Run: {run_name} | state_dim={state_dim}, action_dim={action_dim}, max_action={max_action} | snapshots at {snapshot_steps}")

    logs = []
    losses = {"critic_loss": np.nan, "actor_loss": np.nan}   # last-seen losses, logged per episode

    # Log the untrained policy's return at t=0 as a baseline point.
    init_return = evaluate_policy(agent, cfg.env_id, cfg.seed, cfg.eval_episodes)
    logs.append(_log_row(cfg.algorithm, cfg.env_id, cfg.seed, 0, eval_return=init_return, episode=0))

    state = reset_env(env, cfg.seed)
    episode_reward = 0.0
    episode_num = 0
    start_time = time.time()

    # Main env loop
    for t in range(cfg.max_timesteps):
        # Random warm-up to fill the buffer, then policy action + Gaussian exploration noise.
        if t < cfg.start_timesteps:
            action = env.action_space.sample()
        else:
            action = agent.select_action(np.asarray(state))
            noise = np.random.normal(0, max_action * cfg.expl_noise, size=action_dim) # exploration noise
            action = (action + noise).clip(-max_action, max_action)

        # Step the env and store the transition in the replay buffer
        next_state, reward, done, done_bool, _ = step_env(env, action)
        replay_buffer.add(state, action, next_state, reward, done_bool)
        state = next_state
        episode_reward += reward

        # One gradient update per step, but only after warm-up has filled the buffer
        if t >= cfg.start_timesteps:
            losses = agent.train(replay_buffer, cfg.batch_size)

        # At episode end: record the return, then reset for the next episode
        if done:
            episode_num += 1
            logs.append(_log_row(cfg.algorithm, cfg.env_id, cfg.seed, t + 1,
                                 episode=episode_num, episode_return=episode_reward,
                                 critic_loss=losses["critic_loss"], actor_loss=losses["actor_loss"]))
            state = reset_env(env)
            episode_reward = 0.0

        # Save weights at milestone steps (later replayed into the progress video)
        if (t + 1) in snapshot_steps:
            agent.save(_snapshot_path(cfg, cfg.seed, t + 1))

        # Periodic deterministic evaluation, logged as an eval row
        if (t + 1) % cfg.eval_freq == 0:
            val = evaluate_policy(agent, cfg.env_id, cfg.seed, cfg.eval_episodes)
            print(f"Eval | T={t + 1:>8} | Return={val:>10.3f} | Elapsed={time.time() - start_time:>.1f}s")
            logs.append(_log_row(cfg.algorithm, cfg.env_id, cfg.seed, t + 1,
                                 eval_return=val, episode=episode_num,
                                 critic_loss=losses["critic_loss"], actor_loss=losses["actor_loss"]))

    env.close()

    # Record the full log to <results>/<env>/logs/<run_name>.csv
    logs_dir = _subdir(cfg.results_dir, cfg.env_id, "logs")
    df = pd.DataFrame(logs)
    df.to_csv(logs_dir / f"{run_name}.csv", index=False)
    print(f"Saved logs to: {logs_dir / f'{run_name}.csv'}")
    return df


# Train our TD3 once per seed and concatenate the logs
# Note that finished seed's CSV is reloaded not retrained (resume via _is_complete)
def run_multi_seed(base_cfg, seeds):
    logs_dir = _subdir(base_cfg.results_dir, base_cfg.env_id, "logs")
    dfs = [] # all dataframe in all runs

    for seed in seeds:
        run_name = f"{base_cfg.algorithm}_{base_cfg.env_id}_seed{seed}"
        csv_path = logs_dir / f"{run_name}.csv"

        # Use a finished run instead of retraining it
        if _is_complete(csv_path, base_cfg.max_timesteps):
            print(f"[skip] {run_name}: complete checkpoint found, loading.")
            dfs.append(pd.read_csv(csv_path))
            continue
        
        # Copy the base config so per-seed edits don't leak across iterations
        cfg = copy.deepcopy(base_cfg)
        cfg.seed = seed
        dfs.append(train_one_run(cfg))

    return pd.concat(dfs, ignore_index=True)
