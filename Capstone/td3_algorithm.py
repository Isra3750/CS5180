# Based on https://github.com/sfujim/TD3/blob/master/TD3.py
# A re-implementation of the TD3 algorithm. Paper: https://arxiv.org/abs/1802.09477

import copy
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Seed everything for reproducibility
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Replay buffer (this is utils.py in the TD3 repo)
class ReplayBuffer:
    def __init__(self, state_dim: int, action_dim: int, max_size: int = 1_000_000):
        
        self.max_size = max_size # memory size
        self.ptr = 0 # pointer / index
        self.size = 0
        self.state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.action = np.zeros((max_size, action_dim), dtype=np.float32)
        self.next_state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.reward = np.zeros((max_size, 1), dtype=np.float32)
        self.not_done = np.zeros((max_size, 1), dtype=np.float32)

    def add(self, state, action, next_state, reward, done_bool): # store transitions
        # Store one transition, overwriting the oldest once full (ring buffer).
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1.0 - done_bool
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size): # sample buffer
        # Return a uniformly sampled minibatch as tensors on `device`.
        ind = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.as_tensor(self.state[ind], device=device),
            torch.as_tensor(self.action[ind], device=device),
            torch.as_tensor(self.next_state[ind], device=device),
            torch.as_tensor(self.reward[ind], device=device),
            torch.as_tensor(self.not_done[ind], device=device),
        )

# Define Actor network
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)

        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))

# Define Critic network (twin critics)
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        # Twin critics (component #1)
        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, action):
        # First critic only
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


# Original class is named `TD3`; kept as `TD3Agent` here
class TD3Agent(object):
    def __init__(self, state_dim, action_dim, max_action, discount=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2, actor_lr=3e-4, critic_lr=3e-4,
                 use_cdq=True): # added for ablation where use_cdq = False -> single-critic target (no clipped min)
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau # target network update rate
        self.policy_noise = policy_noise # TPS -> noise add to target action
        self.noise_clip = noise_clip # TPS -> Limits the target action noise
        self.policy_freq = policy_freq # policy update frequency
        self.use_cdq = use_cdq # clipped double-Q (as mentioned above)

        self.total_it = 0

    def select_action(self, state):
        # actor to select action
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size=256):
        # one complete TD3 learning update using replay buffer
        self.total_it += 1

        # Sample replay buffer, minibatch 
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        with torch.no_grad():
            # Select action according to policy and add clipped noise (component #3) -> target policy smoothing (TPS)
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)

            # Compute the target Q value (also component #1 -> use minimum of twin critics here)
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2) if self.use_cdq else target_Q1  # use_cdq=False disables clipped double-Q
            target_Q = reward + not_done * self.discount * target_Q

        # Get current Q estimates from twin critics
        current_Q1, current_Q2 = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss_value = None # this is so we can log it even on non-actor-update steps

        # Delayed policy updates (component #2)
        if self.total_it % self.policy_freq == 0:

            # Compute actor losse
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            actor_loss_value = float(actor_loss.detach().cpu().item())

            # Update the frozen target models (slowly with target network -> base on tau)
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        # (added) return the losses so the training loop can record them to CSV
        return {"critic_loss": float(critic_loss.detach().cpu().item()), "actor_loss": actor_loss_value}

    def save(self, filename):
        # Here differs from original -> only the weights are saved, we never resume mid-training
        # video.py needs just the actor/critic to replay a snapshot
        torch.save(self.critic.state_dict(), filename + "_critic.pt")
        torch.save(self.actor.state_dict(), filename + "_actor.pt")

    def load(self, filename):
        # Load weights back into TD3agent
        self.critic.load_state_dict(torch.load(filename + "_critic.pt", map_location=device))
        self.critic_target = copy.deepcopy(self.critic) # target network are recreated

        self.actor.load_state_dict(torch.load(filename + "_actor.pt", map_location=device))
        self.actor_target = copy.deepcopy(self.actor)
