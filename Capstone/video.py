# video.py -> roll out a trained policy, render frames, and write GIF/MP4 progress videos
# make_progress_video (our TD3) and make_sb3_progress_video (an SB3 baseline) render the milestone snapshots
from pathlib import Path

import numpy as np

from td3_algorithm import TD3Agent
from config import _subdir
from environment import make_env, reset_env, step_env, get_env_dims
from baselines import _sb3_classes


# Run one ep and collect the rendered frames
def _rollout_frames(policy_fn, env_id, seed, max_steps=500):
    env = make_env(env_id, seed, render_mode="rgb_array")
    obs = reset_env(env, seed)
    frames = []
    done = False
    steps = 0
    while not done and steps < max_steps: # cap length so one bad episode can't run forever
        frames.append(np.asarray(env.render()))
        action = policy_fn(np.asarray(obs)) # get a deterministic action
        obs, _, done, _, _ = step_env(env, action)
        steps += 1
    env.close()
    return frames


# Draw a small caption bar on a frame; returns the frame unchanged if Pillow is missing
def _label_frame(frame, text):
    try:
        from PIL import Image, ImageDraw
    except Exception:
        return frame
    img = Image.fromarray(frame).convert("RGB")
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, img.width, 18], fill=(0, 0, 0)) # black strip along the top
    draw.text((4, 3), text, fill=(255, 255, 255))
    return np.asarray(img)


# Concatenate the labelled stage rollouts side by side and write a GIF (+ MP4 if ffmpeg is available)
def _write_progress_video(stages, out_dir, out_name, fps):
    import imageio
    if not stages:
        return None
    n_frames = max(len(frames) for _, frames in stages)
    combined = []
    for i in range(n_frames):
        panels = []
        for label, frames in stages:
            frame = frames[min(i, len(frames) - 1)] # hold the last frame for shorter rollouts
            panels.append(_label_frame(frame, label))
        combined.append(np.concatenate(panels, axis=1)) # stack panels horizontally
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gif_path = out_dir / f"{out_name}.gif"
    imageio.mimsave(gif_path, combined, fps=fps)
    try:
        imageio.mimsave(out_dir / f"{out_name}.mp4", combined, fps=fps) # MP4 needs ffmpeg
    except Exception as e:
        print(f"(MP4 skipped: {e}; GIF written.)")
    print(f"Saved progress video: {gif_path}")
    return str(gif_path)


# Find snapshot files named '<prefix><step><suffix>' and return their steps, sorted ascending
def _find_snapshot_steps(models_dir, prefix, suffix):
    steps = []
    for path in Path(models_dir).glob(f"{prefix}*{suffix}"):
        middle = path.name[len(prefix):len(path.name) - len(suffix)]   # the text between prefix and suffix
        if middle.isdigit():
            steps.append(int(middle))
    return sorted(set(steps))


# For TD3 algo -> render early / mid / late rollouts from the milestone snapshots, side by side
def make_progress_video(cfg, seed=0, max_steps=400, fps=30, out_name=None):
    env = make_env(cfg.env_id, seed)
    state_dim, action_dim, max_action = get_env_dims(env)
    env.close()

    models_dir = _subdir(cfg.results_dir, cfg.env_id, "models")
    prefix = f"TD3_{cfg.env_id}_seed{seed}_step"
    steps = _find_snapshot_steps(models_dir, prefix, "_actor.pt")

    stages = []
    for step in steps:
        # Reload each snapshot into a fresh agent and roll it out (seed+500 -> unseen start states)
        agent = TD3Agent(state_dim, action_dim, max_action)
        agent.load(str(models_dir / f"{prefix}{step}"))
        frames = _rollout_frames(agent.select_action, cfg.env_id, seed + 500, max_steps)
        stages.append((f"step {step:,}", frames))

    if not stages:
        print(f"No TD3 snapshots for {cfg.env_id} seed {seed}. Run train_one_run/run_multi_seed first.")
        return None
    videos_dir = _subdir(cfg.results_dir, cfg.env_id, "videos")
    return _write_progress_video(stages, videos_dir, out_name or f"progress_TD3_{cfg.env_id}_seed{seed}", fps)


# SB3 baseline -> use same idea, but load milestone .zip snapshots (needs save_snapshots=True at train time)
def make_sb3_progress_video(cfg, seed=0, algo="DDPG", max_steps=400, fps=30, out_name=None):
    algo = algo.upper()
    algo_name = f"SB3-{algo}"
    Cls = _sb3_classes()[algo]
    models_dir = _subdir(cfg.results_dir, cfg.env_id, "models")
    prefix = f"{algo_name}_{cfg.env_id}_seed{seed}_step"
    steps = _find_snapshot_steps(models_dir, prefix, ".zip")

    stages = []
    for step in steps:
        model = Cls.load(str(models_dir / f"{prefix}{step}"))

        # Wrap the SB3 predict() into a plain policy_fn(obs) -> action
        def policy(obs):
            action, _ = model.predict(obs, deterministic=True)
            return action

        frames = _rollout_frames(policy, cfg.env_id, seed + 500, max_steps)
        stages.append((f"step {step:,}", frames))

    if not stages:
        print(f"No {algo_name} snapshots for {cfg.env_id} seed {seed}. "
              f"Run run_sb3_multi_seed(..., algo='{algo}', save_snapshots=True) first.")
        return None
    videos_dir = _subdir(cfg.results_dir, cfg.env_id, "videos")
    return _write_progress_video(stages, videos_dir,
                                 out_name or f"progress_{algo_name}_{cfg.env_id}_seed{seed}", fps)


# Render a single rollout of a trained policy (our TD3Agent or an SB3 model) to a GIF
def record_final_video(env_id, model_or_agent, seed=0, max_steps=500, fps=30,
                       results_dir="results", out_name=None, is_sb3=False):
    import imageio

    # Pick the right way to query actions depending on the policy type
    if is_sb3:
        def policy(obs):
            action, _ = model_or_agent.predict(obs, deterministic=True)
            return action
    else:
        policy = model_or_agent.select_action

    frames = _rollout_frames(policy, env_id, seed, max_steps)
    videos_dir = _subdir(results_dir, env_id, "videos")
    out_name = out_name or f"final_{env_id}_seed{seed}"
    gif_path = videos_dir / f"{out_name}.gif"
    imageio.mimsave(gif_path, frames, fps=fps)
    print(f"Saved final-policy video: {gif_path}")
    return str(gif_path)
