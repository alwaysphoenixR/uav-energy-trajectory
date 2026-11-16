"""
Module: agents.qier_buffer
Quantum-Inspired Experience Replay / Advantage Buffer (QiER) for PPO rollouts.
Integrates Generalized Advantage Estimation (GAE) with Grover-rotation-based
quantum amplitude amplification to prioritize high-advantage transitions without
starving exploration.
"""
import numpy as np
import torch

class QuantumInspiredBuffer:
    """
    On-policy rollout buffer with Quantum-Inspired sampling weights.
    P(i) = sin^2((2*k + 1) * theta_i) / sum(sin^2(...))
    where theta_i = (p_i / max(p)) * (pi / 2)
    and p_i = |A_i| + epsilon.
    """
    def __init__(self, capacity: int, state_dim: int = 12, action_dim: int = 3, gamma: float = 0.99, gae_lambda: float = 0.95):
        self.capacity = capacity
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.values = np.zeros(capacity, dtype=np.float32)
        self.log_probs = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        
        self.advantages = np.zeros(capacity, dtype=np.float32)
        self.returns = np.zeros(capacity, dtype=np.float32)
        self.sampling_probs = np.zeros(capacity, dtype=np.float32)
        
        self.ptr = 0
        self.path_start_idx = 0

    def store(self, state: np.ndarray, action: np.ndarray, reward: float, value: float, log_prob: float, done: bool):
        """Stores a single step transition."""
        assert self.ptr < self.capacity, "Buffer overflow!"
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.log_probs[self.ptr] = log_prob
        self.dones[self.ptr] = 1.0 if done else 0.0
        self.ptr += 1

    def finish_path(self, last_val: float = 0.0):
        """
        Computes GAE-lambda advantages and rewards-to-go over the trajectory.
        """
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rewards[path_slice], last_val)
        vals = np.append(self.values[path_slice], last_val)
        dns = np.append(self.dones[path_slice], 0.0)
        
        # GAE-Lambda formulation:
        # delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)
        # A_t = sum_{l=0}^{T-t} (gamma * lambda)^l * delta_{t+l}
        deltas = rews[:-1] + self.gamma * vals[1:] * (1.0 - dns[:-1]) - vals[:-1]
        
        gae = 0.0
        advs = np.zeros_like(deltas)
        for t in reversed(range(len(deltas))):
            gae = deltas[t] + self.gamma * self.gae_lambda * (1.0 - dns[t]) * gae
            advs[t] = gae
            
        self.advantages[path_slice] = advs
        self.returns[path_slice] = advs + vals[:-1]
        self.path_start_idx = self.ptr

    def compute_quantum_probabilities(self, k_grover: int = 1, epsilon: float = 1e-4, use_quantum: bool = True):
        """
        Computes Grover-inspired probability distribution over stored transitions.
        """
        adv_slice = self.advantages[:self.ptr]
        if not use_quantum or self.ptr == 0:
            # Uniform distribution fallback
            self.sampling_probs[:self.ptr] = 1.0 / max(1, self.ptr)
            return

        priorities = np.abs(adv_slice) + epsilon
        max_p = np.max(priorities)
        if max_p < 1e-8:
            self.sampling_probs[:self.ptr] = 1.0 / self.ptr
            return
            
        # Map priorities to quantum state rotation angle theta in [0, pi/2]
        theta = (priorities / max_p) * (np.pi / 2.0)
        
        # Grover rotation amplitude amplification with k iterations
        grover_factor = (2 * k_grover + 1) * theta
        amplitudes_sq = np.sin(grover_factor) ** 2
        
        # Normalize to valid probability distribution
        total_amp = np.sum(amplitudes_sq)
        if total_amp < 1e-8 or np.isnan(total_amp):
            self.sampling_probs[:self.ptr] = 1.0 / self.ptr
        else:
            self.sampling_probs[:self.ptr] = amplitudes_sq / total_amp

    def get_batches(self, batch_size: int = 64, use_quantum: bool = True):
        """
        Generates mini-batches sampled with QiER probabilities or uniform permutation.
        """
        assert self.ptr == self.capacity, f"Buffer not full ({self.ptr}/{self.capacity})!"
        self.compute_quantum_probabilities(use_quantum=use_quantum)
        
        # Normalize advantages across entire buffer
        adv_mean = np.mean(self.advantages)
        adv_std = np.std(self.advantages) + 1e-8
        norm_advantages = (self.advantages - adv_mean) / adv_std
        
        indices = np.arange(self.capacity)
        
        if use_quantum:
            # Sample indices with replacement or weighted permutation
            sampled_indices = np.random.choice(indices, size=self.capacity, replace=True, p=self.sampling_probs)
        else:
            sampled_indices = np.random.permutation(indices)
            
        for start in range(0, self.capacity, batch_size):
            batch_idx = sampled_indices[start:start + batch_size]
            yield {
                'states': torch.as_tensor(self.states[batch_idx], dtype=torch.float32),
                'actions': torch.as_tensor(self.actions[batch_idx], dtype=torch.float32),
                'returns': torch.as_tensor(self.returns[batch_idx], dtype=torch.float32),
                'advantages': torch.as_tensor(norm_advantages[batch_idx], dtype=torch.float32),
                'log_probs': torch.as_tensor(self.log_probs[batch_idx], dtype=torch.float32)
            }

    def reset(self):
        """Resets pointers for next rollout."""
        self.ptr = 0
        self.path_start_idx = 0
