"""Rollout buffer for storing trajectories and computing GAE."""

import torch


class RolloutBuffer:
    """Stores one episode of experience for all agents."""

    def __init__(self):
        self.obs = []          # (n_agents, obs_dim) per step
        self.actions = []      # (n_agents,) per step
        self.log_probs = []    # (n_agents,) per step
        self.rewards = []      # scalar per step (team reward)
        self.values = []       # scalar per step
        self.dones = []        # bool per step

    def store(self, obs, actions, log_probs, reward, value, done):
        self.obs.append(obs)
        self.actions.append(actions)
        self.log_probs.append(log_probs)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def compute_gae(self, last_value, gamma=0.99, gae_lambda=0.95):
        """Compute generalized advantage estimation.

        Returns:
            advantages: tensor (T, n_agents) - same advantage for all agents
            returns: tensor (T,)
        """
        T = len(self.rewards)
        n_agents = self.obs[0].shape[0]

        rewards = torch.tensor(self.rewards, dtype=torch.float32)
        values = torch.tensor(self.values, dtype=torch.float32)
        dones = torch.tensor(self.dones, dtype=torch.float32)

        advantages = torch.zeros(T)
        last_gae = 0.0

        for t in reversed(range(T)):
            if t == T - 1:
                next_value = last_value
            else:
                next_value = values[t + 1]
            next_non_terminal = 1.0 - dones[t]

            delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae

        returns = advantages + values

        # Broadcast advantage to all agents (shared team reward)
        advantages = advantages.unsqueeze(1).expand(T, n_agents)

        return advantages, returns

    def get_batches(self, last_value, gamma=0.99, gae_lambda=0.95):
        """Process buffer into training tensors.

        Returns dict with keys:
            obs: (T, n_agents, obs_dim)
            actions: (T, n_agents)
            old_log_probs: (T, n_agents)
            advantages: (T, n_agents)
            returns: (T,)
        """
        advantages, returns = self.compute_gae(last_value, gamma, gae_lambda)

        data = {
            "obs": torch.stack(self.obs),
            "actions": torch.stack(self.actions),
            "old_log_probs": torch.stack(self.log_probs),
            "advantages": advantages,
            "returns": returns,
        }
        return data

    def clear(self):
        self.obs.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()

    def __len__(self):
        return len(self.rewards)
