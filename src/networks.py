"""Neural network architectures for MAPPO."""

import torch
import torch.nn as nn
from torch.distributions import Categorical


class Actor(nn.Module):
    """Policy network: maps observation -> action distribution.

    Each agent has its own actor (parameter sharing optional).
    """

    def __init__(self, obs_dim, n_actions, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, obs):
        """obs: (..., obs_dim) -> logits: (..., n_actions)"""
        return self.net(obs)

    def get_action(self, obs, deterministic=False):
        """Sample action and return action, log_prob, entropy."""
        logits = self.forward(obs)
        dist = Categorical(logits=logits)
        if deterministic:
            action = logits.argmax(dim=-1)
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy

    def evaluate_action(self, obs, action):
        """Evaluate given action under current policy."""
        logits = self.forward(obs)
        dist = Categorical(logits=logits)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return log_prob, entropy


class Critic(nn.Module):
    """Centralized value function: maps global state -> value.

    Takes the concatenation of all agents' observations as input.
    """

    def __init__(self, global_state_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(global_state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, global_state):
        """global_state: (..., global_state_dim) -> value: (..., 1)"""
        return self.net(global_state)
