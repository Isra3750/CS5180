import numpy as np
import gymnasium as gym
from vi import value_iteration

# part (b) policy iteration
def policy_iteration(P, gamma):
    num_states = len(P)
    num_actions = len(P[0])
    
    # Init with a zero policy
    V = np.zeros(num_states)
    policy = np.zeros(num_states, dtype=int)
    iterations = 0
    
    while True:
        # Phase 1 -> Policy evaluation
        # Solve (I - gamma * P_pi) * V = R_pi
        I = np.eye(num_states)
        P_pi = np.zeros((num_states, num_states))
        R_pi = np.zeros(num_states)
        
        for s in range(num_states):
            a = policy[s]
            for prob, next_state, reward, terminated in P[s][a]:
                R_pi[s] += prob * reward
                if not terminated:
                    P_pi[s, next_state] += prob
                    
        V = np.linalg.solve(I - gamma * P_pi, R_pi)
        
        # Phase 2 -> Policy improvement
        policy_stable = True
        new_policy = np.zeros(num_states, dtype=int)
        
        for s in range(num_states):
            q_values = np.zeros(num_actions)
            for a in range(num_actions):
                for prob, next_state, reward, terminated in P[s][a]:
                    v_next = 0 if terminated else V[next_state]
                    q_values[a] += prob * (reward + gamma * v_next)
            
            best_action = np.argmax(q_values)
            new_policy[s] = best_action
            
            if best_action != policy[s]:
                policy_stable = False
                
        policy = new_policy
        iterations += 1
        
        if policy_stable:
            break
            
    return V, policy, iterations

if __name__ == "__main__":
    env = gym.make("FrozenLake-v1", is_slippery=True)
    P = env.unwrapped.P
    
    V_opt, pi_opt, iters = policy_iteration(P, gamma = 0.99)
    
    print("Policy Iteration: ")
    print(f"Iteration count to converge: {iters}")
    
    # Check if match with previous part VI results
    _, pi_opt_2, _ = value_iteration(P, gamma = 0.99, theta = 1e-4)
    print(f"VI and PI return the same policy? {np.array_equal(pi_opt, pi_opt_2)}")