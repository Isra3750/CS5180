import time
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

# Import
from vi import value_iteration
from pi import policy_iteration

# part (c) comparison between VI and PI
if __name__ == "__main__":
    env = gym.make("FrozenLake-v1", is_slippery=True)
    P = env.unwrapped.P
    num_states = env.observation_space.n
    num_actions = env.action_space.n

    gamma_list = [0.5, 0.9, 0.99, 0.999] # required from assignment
    vi_iters = []
    pi_iters = []

    print("gamma | algorithm | iters | time (sec) | backups")
    for g in gamma_list:
        # Run Value Iteration
        start_time = time.time()
        _, _, iters_vi = value_iteration(P, gamma = g, theta = 1e-4)
        time_vi = time.time() - start_time

        vi_iters.append(iters_vi)
        vi_backups = iters_vi * num_states # calculate number of backups
        print("gamma =", g, "VI", iters_vi, round(time_vi, 5), vi_backups)

        # Run Policy Iteration
        start_time = time.time()
        _, _, iters_pi = policy_iteration(P, gamma = g)
        time_pi = time.time() - start_time

        pi_iters.append(iters_pi)
        pi_backups = iters_pi * (num_states + num_states**2 // num_actions) # calculate number of backups
        print("gamma =", g, "PI", iters_pi, round(time_pi, 5), pi_backups)
        print(" ")

    plt.plot(gamma_list, vi_iters, label='Value Iteration', c='forestgreen')
    plt.plot(gamma_list, pi_iters, label='Policy Iteration', c='orangered')

    plt.xlabel("Gamma (discount factor)")
    plt.ylabel("Count (iterations to convergence)")
    plt.title("Count vs Gamma")
    plt.legend()
    plt.show()