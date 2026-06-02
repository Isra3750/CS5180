import numpy as np
import gymnasium as gym

# part (a) value iteration
def value_iteration(P, gamma, theta):
    num_states = len(P)
    num_actions = len(P[0])
    
    V = np.zeros(num_states)
    policy = np.zeros(num_states, dtype = int)
    iterations = 0 # k iteration

    while True: # for k iterations
        delta = 0 # Change in value
        V_k_1 = np.zeros(num_states) # initialize with zeros
        
        # Bellman backup sweep
        for s in range(num_states):
            q_values = np.zeros(num_actions)
            for a in range(num_actions):
                # Calculate expected value for state-action pair Q(s,a)
                for prob, next_state, reward, terminated in P[s][a]:
                    v_next = 0 if terminated else V[next_state]
                    q_values[a] += prob * (reward + gamma * v_next)
            
            V_k_1[s] = np.max(q_values) # next state is maximum of Q(s,a)
            delta = max(delta, abs(V_k_1[s] - V[s])) # difference between V_k and V_k+1
            
        V = V_k_1
        iterations += 1
        
        if delta < (theta * (1 - gamma) / gamma): # termination condition
            break
            
    # Extract the policy from the value function
    for s in range(num_states):
        q_values = np.zeros(num_actions)
        for a in range(num_actions):
            for prob, next_state, reward, terminated in P[s][a]:
                v_next = (0.0 if terminated else V[next_state]) # next state
                q_values[a] += prob * (reward + gamma * v_next)
        policy[s] = np.argmax(q_values)
        
    return V, policy, iterations

# quick test
if __name__ == "__main__":
    env = gym.make("FrozenLake-v1", is_slippery=True)
    P = env.unwrapped.P # Transition probability -> P[s][a] = [(probability, next_state, reward, terminated)]
    print(len(P))
    print(len(P[0]))
    print(env.unwrapped.desc)

    # Run Value Iteration and get V* and pi*
    V_table, Policy_opt, iters = value_iteration(P, gamma = 0.99, theta = 1e-4) # using required parameters from assignment
    
    print("Value Iteration: ")
    print(f"Iteration count to converge: {iters}")

    print("\nFull V* table (4x4 table):")
    print(np.round(V_table.reshape((4, 4)), 4))

    print("\nPolicy pi*:")
    mapping = {0: '<-', 1: '↓ ', 2: '->', 3: '↑ '}
    for row in Policy_opt.reshape((4, 4)):
        print(" ".join([mapping[a] for a in row]))