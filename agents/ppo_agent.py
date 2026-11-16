"""
Module: agents.ppo_agent
PyTorch Actor-Critic PPO Architecture for UAV Tactical Heading Planning.
Trunk: 12 -> 128 -> 64
Actor: outputs 3D continuous heading displacement u_ppo with Tanh squashing in [-1, 1]
Critic: outputs scalar state value V(s)
"""
import torch
import torch.nn as nn
from torch.distributions import Normal
import numpy as np

class ActorCritic(nn.Module):
    def __init__(self, state_dim: int = 12, action_dim: int = 3, hidden1: int = 128, hidden2: int = 64):
        super(ActorCritic, self).__init__()
        
        # Shared feature trunk
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden1),
            nn.Tanh(),
            nn.Linear(hidden1, hidden2),
            nn.Tanh()
        )
        
        # Actor head: mean continuous action
        self.actor_head = nn.Linear(hidden2, action_dim)
        # Learnable log standard deviation
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.5))
        
        # Critic head: scalar state value V(s)
        self.critic_head = nn.Linear(hidden2, 1)

    def forward(self, state: torch.Tensor):
        feats = self.trunk(state)
        # Tanh squashed mean action [-1, 1]
        action_mean = torch.tanh(self.actor_head(feats))
        action_std = torch.exp(self.log_std).clamp(min=1e-3, max=1.0)
        value = self.critic_head(feats)
        return action_mean, action_std, value

    def get_action(self, state: np.ndarray, deterministic: bool = False) -> tuple[np.ndarray, float, float]:
        """
        Samples action from policy.
        Returns: (action, log_prob, value)
        """
        state_t = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        action_mean, action_std, val = self.forward(state_t)
        
        if deterministic:
            action = action_mean
            log_prob = torch.tensor([0.0])
        else:
            dist = Normal(action_mean, action_std)
            action = dist.sample()
            # Squashed clipping to [-1, 1]
            action = torch.clamp(action, -1.0, 1.0)
            log_prob = dist.log_prob(action).sum(dim=-1)
            
        return action.squeeze(0).detach().cpu().numpy(), float(log_prob.item()), float(val.item())

    def evaluate(self, states: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluates batch of states and actions for PPO gradient updates.
        """
        action_mean, action_std, values = self.forward(states)
        dist = Normal(action_mean, action_std)
        log_probs = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_probs, values.squeeze(-1), entropy
