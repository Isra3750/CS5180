import numpy as np
import matplotlib.pyplot as plt

from env import ApartmentEnv
from policies import RandomPolicy, ThresholdPolicy, OptimalPolicy

# set seed
np.random.seed(67)

def evaluate_policy(env, policy, episodes = 10000):
    returns = np.zeros(episodes) # final utility
    reject_all_count = 0 
    
    for i in range(episodes):
        obs, _ = env.reset()
        terminated = False
        ep_return = 0
        while not terminated:
            action = policy.act(obs)
            obs, reward, terminated, _, _ = env.step(action)
            ep_return += reward # accumulate reward
            
        returns[i] = ep_return
        if ep_return == 0:
            reject_all_count += 1 # failure count
            
    mean_u = np.mean(returns) # EU
    std_err_u = np.std(returns) / np.sqrt(episodes)
    fraction_rejected = reject_all_count / episodes # this is risk of total rejection

    return mean_u, std_err_u, fraction_rejected, returns

if __name__ == "__main__":
    T, K, N = 4, 4, 10000
    env = ApartmentEnv(T = T, K = K, seed = 67)
    
    # part c - parameter sweep
    print("Sweeping u_min for threshold policy for part c: ")

    best_u_min, best_mean = -1, -1
    for u in [1, 2, 3, 4]:
        mean_u, _, _, _ = evaluate_policy(env, ThresholdPolicy(u), episodes = N)
        print(f"u_min = {u}: Mean Utility = {mean_u:.3f}")

        if mean_u > best_mean: # set best mean
            best_mean = mean_u
            best_u_min = u
    print(f"Best fixed threshold: {best_u_min}\n")

    print("Policy Comparison: ")
    policies = {
        "Random": RandomPolicy(T), # random policy
        f"Threshold (u_min = {best_u_min})": ThresholdPolicy(best_u_min), # threshold policy
        "Optimal": OptimalPolicy() # optimal policy
    }
    
    results = {}
    # evaluate each policies
    for name, pol in policies.items():
        m, se, frac, rets = evaluate_policy(env, pol, episodes = N)
        results[name] = rets
        print(f"{name}:\n  Mean = {m:.3f} +/- {se:.3f}, Fraction Rejected All = {frac:.3f}\n")

    # Part c - plot
    # Plot one figure with overlaid histograms of returns
    plt.figure()
    for name, rets in results.items():
        plt.hist(rets, bins = np.arange(-0.5, K + 1.5, 1), alpha = 0.3, label = f"{name} (mean = {np.mean(rets):.2f})", density = True)

    plt.xlabel("Utility received")
    plt.ylabel("Frequency")
    plt.title("Returns distribution (with N = 10000)")
    plt.legend()
    plt.savefig("Playground/Homework_5180/PS0/histogram_result.png") # change base on current dir
    
    # part d - robustness sweep
    print("Robustness Sweep for part d: ")
    for sigma in [0.0, 0.5, 1.0, 2.0]: # different noise level
        noisy_env = ApartmentEnv(T = T, K = K, noise_std = sigma, seed = 67)

        m_opt, _, _, _ = evaluate_policy(noisy_env, OptimalPolicy(), episodes = N)
        m_thr, _, _, _ = evaluate_policy(noisy_env, ThresholdPolicy(best_u_min), episodes = N)
        m_rnd, _, _, _ = evaluate_policy(noisy_env, RandomPolicy(T), episodes = N)

        print(f"Sigma = {sigma}: Optimal = {m_opt:.3f},  Threshold = {m_thr:.3f}, Random = {m_rnd:.3f}")