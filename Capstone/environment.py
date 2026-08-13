# environment.py -> Gymnasium environment helpers
import gymnasium as gym

# Create a Gymnasium env and seed it (both the env and its action space) for reproducibility
def make_env(env_id, seed, render_mode=None):
    env = gym.make(env_id, render_mode=render_mode)
    env.reset(seed=seed) # seed the RNG behind resets / dynamics
    env.action_space.seed(seed) # seed action_space.sample(), this used during warm-up
    return env


# Reset the env and return just the obs
def reset_env(env, seed=None):
    if seed is not None:
        obs, _ = env.reset(seed=seed)
    else:
        obs, _ = env.reset()
    return obs


# Step the env and normalise the output to (next_state, reward, done, done_bool, info)
def step_env(env, action):
    next_state, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    return next_state, reward, done, float(terminated), info


# Read the continuous-control dimensions from an env
def get_env_dims(env):
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0]) # symmetric action bound (e.g. 1.0)

    return state_dim, action_dim, max_action
