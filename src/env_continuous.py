"""Wrapper for Simple Spread with continuous actions and variable N agents."""

import numpy as np
import torch
from mpe2 import simple_spread_v3


class SimpleSpreadContinuousEnv:
    """Simple Spread with continuous actions.

    Action space: 2D continuous force (x, y) per agent.
    The underlying MPE env uses 5D continuous action:
    [no_action, left, right, down, up] as force magnitudes.
    We simplify to 2D by mapping (fx, fy) -> 5D.
    """

    def __init__(self, n_agents=5, max_cycles=25, device="cpu"):
        self.n_agents = n_agents
        self.max_cycles = max_cycles
        self.device = device
        self.env = simple_spread_v3.parallel_env(
            N=n_agents, max_cycles=max_cycles, continuous_actions=True
        )
        self.agent_names = None
        self.obs_dim = None
        self.action_dim = 5  # MPE continuous action dim

    def reset(self, seed=None):
        obs_dict, infos = self.env.reset(seed=seed)
        self.agent_names = list(obs_dict.keys())
        obs_array = np.stack([obs_dict[a] for a in self.agent_names])
        self.obs_dim = obs_array.shape[1]
        return torch.tensor(obs_array, dtype=torch.float32, device=self.device)

    def step(self, actions):
        """actions: (n_agents, action_dim) tensor of continuous actions."""
        action_dict = {}
        for i, agent in enumerate(self.agent_names):
            action_dict[agent] = actions[i].detach().cpu().numpy()

        obs_dict, rew_dict, term_dict, trunc_dict, info_dict = self.env.step(action_dict)

        if not obs_dict:
            return None, None, True, {}

        obs_array = np.stack([obs_dict[a] for a in self.agent_names])
        obs_tensor = torch.tensor(obs_array, dtype=torch.float32, device=self.device)

        rewards = [rew_dict[a] for a in self.agent_names]
        team_reward = sum(rewards)
        done = any(term_dict.values()) or any(trunc_dict.values())

        return obs_tensor, team_reward, done, {"individual_rewards": rewards}

    def get_global_state(self, obs):
        return obs.reshape(-1)

    def close(self):
        self.env.close()

    @property
    def observation_space_dim(self):
        return self.obs_dim

    @property
    def global_state_dim(self):
        return self.n_agents * self.obs_dim
