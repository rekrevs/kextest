"""Wrapper around PettingZoo Simple Spread to produce tensors for training."""

import numpy as np
import torch
from mpe2 import simple_spread_v3


class SimpleSpreadEnv:
    """Wraps Simple Spread parallel env for easy tensor-based interaction.

    Observations are stacked into a single tensor of shape (n_agents, obs_dim).
    Actions are expected as a tensor of shape (n_agents,) with integer values 0-4.
    """

    def __init__(self, n_agents=3, max_cycles=25, device="cpu"):
        self.n_agents = n_agents
        self.max_cycles = max_cycles
        self.device = device
        self.env = simple_spread_v3.parallel_env(
            N=n_agents, max_cycles=max_cycles, continuous_actions=False
        )
        self.agent_names = None
        self.obs_dim = None
        self.n_actions = 5  # no-op, left, right, down, up

    def reset(self, seed=None):
        obs_dict, infos = self.env.reset(seed=seed)
        self.agent_names = list(obs_dict.keys())
        obs_array = np.stack([obs_dict[a] for a in self.agent_names])
        self.obs_dim = obs_array.shape[1]
        obs_tensor = torch.tensor(obs_array, dtype=torch.float32, device=self.device)
        return obs_tensor  # (n_agents, obs_dim)

    def step(self, actions):
        """Take a step. actions: tensor of shape (n_agents,) with int actions."""
        action_dict = {}
        for i, agent in enumerate(self.agent_names):
            action_dict[agent] = int(actions[i].item())

        obs_dict, rew_dict, term_dict, trunc_dict, info_dict = self.env.step(action_dict)

        if not obs_dict:
            # Episode ended
            return None, None, True, {}

        obs_array = np.stack([obs_dict[a] for a in self.agent_names])
        obs_tensor = torch.tensor(obs_array, dtype=torch.float32, device=self.device)

        # Team reward: sum of all agent rewards
        rewards = [rew_dict[a] for a in self.agent_names]
        team_reward = sum(rewards)

        done = any(term_dict.values()) or any(trunc_dict.values())

        return obs_tensor, team_reward, done, {"individual_rewards": rewards}

    def get_global_state(self, obs):
        """Global state = concatenation of all observations. Shape: (n_agents * obs_dim,)"""
        return obs.reshape(-1)

    def close(self):
        self.env.close()

    @property
    def observation_space_dim(self):
        return self.obs_dim

    @property
    def global_state_dim(self):
        return self.n_agents * self.obs_dim
