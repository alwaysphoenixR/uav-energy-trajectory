"""
Proximal Policy Optimization (PPO) Actor-Critic Policy Network in PyTorch.
Upper-level tactical reference trajectory generator.
"""
import torch
import torch.nn as nn
from torch.distributions import Normal
import numpy as np

class ActorCritic(nn.Module):
    def __init__(self, state_dim: int = 14, action_dim: int = 3, hidden_dim1: int = 128, hidden_dim2: int = 64):
        super(ActorCritic, self).__init__()
        
        # Shared or decoupled Actor MLP (128 -> 64 -> 3)
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim1),
            nn.Tanh(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.Tanh(),
            nn.Linear(hidden_dim2, action_dim),
            nn.Tanh()  # Output bounded in [-1, 1]
        )
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        
        # Critic MLP (128 -> 64 -> 1)
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim1),
            nn.Tanh(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.Tanh(),
            nn.Linear(hidden_dim2, 1)
        )
        
    def forward(self, state: torch.Tensor):
        action_mean = self.actor(state)
        action_std = torch.exp(self.log_std)
        value = self.critic(state)
        return action_mean, action_std, value

    def act(self, state: np.ndarray, deterministic: bool = False):
        state_tensor = torch.as_tensor(state, dtype=torch.float32)
        action_mean, action_std, value = self.forward(state_tensor)
        
        if deterministic:
            action = action_mean
            log_prob = torch.tensor([0.0])
        else:
            dist = Normal(action_mean, action_std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
            
        action_clipped = torch.clamp(action, -1.0, 1.0)
        return action_clipped.detach().cpu().numpy(), log_prob.detach().cpu().numpy(), value.detach().cpu().item()

    def evaluate(self, states: torch.Tensor, actions: torch.Tensor):
        action_mean, action_std, values = self.forward(states)
        dist = Normal(action_mean, action_std)
        log_probs = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_probs, values.squeeze(-1), entropy


class PPOAgent:
    def __init__(self, state_dim: int = 14, action_dim: int = 3, lr: float = 3e-4, gamma: float = 0.99, gae_lambda: float = 0.95, clip_eps: float = 0.2):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        
        self.policy = ActorCritic(state_dim, action_dim)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.mse_loss = nn.MSELoss()
        
    def train_step(self, states, actions, old_log_probs, rewards, dones, epochs=5, batch_size=64):
        states_t = torch.tensor(np.array(states), dtype=torch.float32)
        actions_t = torch.tensor(np.array(actions), dtype=torch.float32)
        old_log_probs_t = torch.tensor(np.array(old_log_probs), dtype=torch.float32)
        
        # Compute GAE advantages and discounted returns
        with torch.no_grad():
            _, _, values = self.policy(states_t)
            values = values.squeeze(-1).numpy()
            
        advantages = np.zeros(len(rewards), dtype=np.float32)
        last_gae = 0.0
        for t in reversed(range(len(rewards))):
            next_val = 0.0 if t == len(rewards) - 1 or dones[t] else values[t + 1]
            delta = rewards[t] + self.gamma * next_val * (1.0 - dones[t]) - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * (1.0 - dones[t]) * last_gae
            advantages[t] = last_gae
            
        returns = advantages + values
        advantages_t = torch.tensor(advantages, dtype=torch.float32)
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)
        returns_t = torch.tensor(returns, dtype=torch.float32)
        
        # Policy update epochs
        dataset_size = len(states)
        for _ in range(epochs):
            indices = np.arange(dataset_size)
            np.random.shuffle(indices)
            for start in range(0, dataset_size, batch_size):
                batch_idx = indices[start:start + batch_size]
                b_s = states_t[batch_idx]
                b_a = actions_t[batch_idx]
                b_old_lp = old_log_probs_t[batch_idx]
                b_adv = advantages_t[batch_idx]
                b_ret = returns_t[batch_idx]
                
                new_lp, values_pred, entropy = self.policy.evaluate(b_s, b_a)
                ratios = torch.exp(new_lp - b_old_lp)
                
                surr1 = ratios * b_adv
                surr2 = torch.clamp(ratios, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * b_adv
                
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = 0.5 * self.mse_loss(values_pred, b_ret)
                entropy_loss = -0.01 * entropy.mean()
                
                total_loss = actor_loss + critic_loss + entropy_loss
                
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()
