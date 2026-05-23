# PS0 Q3 - Isra Chokwatana
# part a - environment, part d - noise
import gymnasium as gym
import numpy as np

# MDP Environment using gymnasium
class ApartmentEnv(gym.Env):
    def __init__(self, T: int, K: int, noise_std: float = 0, seed = None):
        super().__init__() # initialize

        self.T = T # Time horizon (current weeks) 
        self.K = K # max apartment quality
        self.noise_std = noise_std # noise for part d

        # Set seed if passed in
        if seed is not None:
            np.random.seed(seed)

        self.action_space = gym.spaces.Discrete(2) # actions can be accpept or reject

        # Observation includes -> current_week (t), observed_quality (U_t)
        # Can range from 1 to T
        self.observation_space = gym.spaces.Box(
            np.array([1, -np.inf]), # lower bound, noise can make obs quality -inf
            np.array([T, np.inf]), # upper bound, noise can make obs quality +inf
            dtype = np.float32
        )

    # Reset the environment
    def reset(self, seed = None, options = None):
        super().reset(seed = seed)

        if seed is not None:
            np.random.seed(seed)

        self.t = 1 # start at week 1
        self.u = np.random.randint(1, self.K + 1) # draw hidden true quality

        # Add noise (From part d)
        u_noise = self.u + np.random.normal(0, self.noise_std) # noisy observation

        obs = np.array([self.t, u_noise], dtype = np.float32)
        return obs, {}

    # Step the environment
    def step(self, action: int):
        if action == 1: # if accept
            reward = self.u
            terminated = True # end if accept
            obs = np.array([self.t, self.u], dtype = np.float32)
            return obs, float(reward), terminated, False, {}

        else: # if reject
            self.t += 1 # go to next week

            if self.t > self.T: # if out of time, fallback 
                reward = 0 # rejected everything -> 0 utility
                terminated = True
                obs = np.array([self.t, 0.0], dtype = np.float32)
                return (obs, float(reward), terminated, False, {})
            
            else: # continue searching
                self.u = np.random.randint(1, self.K + 1)  # new week, new draw
                obs_u = self.u + np.random.normal(0, self.noise_std) # noisy obs
                obs = np.array([self.t, obs_u], dtype = np.float32)
                return (obs, 0.0, False, False, {}) # 0 reward mid-search, not terminal