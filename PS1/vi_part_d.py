import numpy as np
import gymnasium as gym

# part (d) policy emergence checker script
# Reused vi.py (part a) with added policy tracking
def policy_track(P, gamma, theta):
    num_states = len(P)
    num_actions = len(P[0])

    V = np.zeros(num_states)
    policy = np.zeros(num_states, dtype=int)
    iterations = 0

    V_history = []
    pi_history = []

    while True:
        V_history.append(V.copy())

        delta = 0 # Change in value
        V_k_1 = np.zeros(num_states)

        # Bellman backup sweep
        for s in range(num_states):
            q_values = np.zeros(num_actions)
            for a in range(num_actions):
                # Calculate expected value for state-action pair Q(s,a)
                for prob, next_state, reward, terminated in P[s][a]:
                    v_next = 0.0 if terminated else V[next_state]
                    q_values[a] += prob * (reward + gamma * v_next)

            V_k_1[s] = np.max(q_values)
            policy[s] = np.argmax(q_values) # greedy policy pi_k from V_k
            delta = max(delta, abs(V_k_1[s] - V[s])) #

        pi_history.append(policy.copy())
        V = V_k_1
        iterations += 1

        if delta < (theta * (1 - gamma) / gamma): # termination condition
            break

    return V, pi_history[-1], V_history, pi_history

# quick test
if __name__ == "__main__":
    env = gym.make("FrozenLake-v1", is_slippery=True)
    P = env.unwrapped.P # P[s][a] = [(probability, next_state, reward, terminated)]

    V_final, pi_final, v_hist, pi_hist = policy_track(P, gamma = 0.99, theta = 1e-4)

    # Return result where first k* where pi_k = pi*
    k_star = None
    for k, pi_k in enumerate(pi_hist):
        if np.array_equal(pi_k, pi_final):
            k_star = k
            v_diff_at_k_star = np.max(np.abs(v_hist[k] - V_final))
            break

    print("Part (d):")
    print("First iteration k* where (pi_k = pi*) =", k_star)
    print("||V_k* - V*|| at that point =", round(v_diff_at_k_star, 4))