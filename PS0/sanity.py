# part a - sanity check
from env import ApartmentEnv

if __name__ == "__main__":
    # Create environment
    env = ApartmentEnv(T = 4, K = 4)

    # Reset
    obs, info = env.reset()
    
    print("Sanity Script - Random policy, one episode")

    terminated = False
    while not terminated:
        t, u_t = obs
        action = env.action_space.sample() # take random action
        
        next_obs, reward, terminated, truncated, info = env.step(action) # next step
        action_txt = "Accept" if action == 1 else "Reject"
        
        # print -> (t, U_t, action, reward, done) as required
        print(f"Week (t): {int(t)}, U_t: {u_t:.2f}, Action: {action_txt}, Reward: {reward}, Done: {terminated}")
        obs = next_obs