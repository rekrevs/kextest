"""MAPPO for continuous action spaces using Gaussian policy."""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path


class ContinuousActor(nn.Module):
    """Gaussian policy: maps observation -> mean and log_std of action distribution."""

    def __init__(self, obs_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, obs):
        h = self.net(obs)
        mean = torch.sigmoid(self.mean_head(h))  # Clamp to [0, 1] for MPE
        std = self.log_std.exp().expand_as(mean)
        return mean, std

    def get_action(self, obs, deterministic=False):
        mean, std = self.forward(obs)
        if deterministic:
            action = mean
        else:
            dist = torch.distributions.Normal(mean, std)
            action = dist.sample()
        action = action.clamp(0.0, 1.0)
        log_prob = torch.distributions.Normal(mean, std).log_prob(action).sum(-1)
        entropy = torch.distributions.Normal(mean, std).entropy().sum(-1)
        return action, log_prob, entropy

    def evaluate_action(self, obs, action):
        mean, std = self.forward(obs)
        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        return log_prob, entropy


class ContinuousCritic(nn.Module):
    def __init__(self, global_state_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(global_state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state):
        return self.net(state)


class ContinuousRolloutBuffer:
    """Buffer for continuous actions."""

    def __init__(self):
        self.obs = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []

    def store(self, obs, actions, log_probs, reward, value, done):
        self.obs.append(obs)
        self.actions.append(actions)
        self.log_probs.append(log_probs)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def compute_gae(self, last_value, gamma=0.99, gae_lambda=0.95):
        T = len(self.rewards)
        n_agents = self.obs[0].shape[0]
        rewards = torch.tensor(self.rewards, dtype=torch.float32)
        values = torch.tensor(self.values, dtype=torch.float32)
        dones = torch.tensor(self.dones, dtype=torch.float32)

        advantages = torch.zeros(T)
        last_gae = 0.0
        for t in reversed(range(T)):
            next_val = last_value if t == T - 1 else values[t + 1]
            delta = rewards[t] + gamma * next_val * (1 - dones[t]) - values[t]
            last_gae = delta + gamma * gae_lambda * (1 - dones[t]) * last_gae
            advantages[t] = last_gae

        returns = advantages + values
        advantages = advantages.unsqueeze(1).expand(T, n_agents)
        return advantages, returns

    def get_batches(self, last_value, gamma=0.99, gae_lambda=0.95):
        advantages, returns = self.compute_gae(last_value, gamma, gae_lambda)
        return {
            "obs": torch.stack(self.obs),
            "actions": torch.stack(self.actions),
            "old_log_probs": torch.stack(self.log_probs),
            "advantages": advantages,
            "returns": returns,
        }

    def clear(self):
        for lst in [self.obs, self.actions, self.log_probs, self.rewards, self.values, self.dones]:
            lst.clear()

    def __len__(self):
        return len(self.rewards)


class ContinuousMAPPO:
    """MAPPO for continuous action spaces."""

    def __init__(self, obs_dim, global_state_dim, action_dim, n_agents,
                 hidden_dim=64, lr_actor=3e-4, lr_critic=3e-4,
                 gamma=0.99, gae_lambda=0.95, clip_eps=0.2,
                 entropy_coef=0.005, ppo_epochs=10, device="cpu"):
        self.n_agents = n_agents
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.ppo_epochs = ppo_epochs
        self.device = device

        self.actor = ContinuousActor(obs_dim, action_dim, hidden_dim).to(device)
        self.critic = ContinuousCritic(global_state_dim, hidden_dim).to(device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

    @torch.no_grad()
    def act(self, obs, deterministic=False):
        actions, log_probs, _ = self.actor.get_action(obs, deterministic)
        value = self.critic(obs.reshape(1, -1)).squeeze()
        return actions, log_probs, value.item()

    def update(self, buffer, last_value):
        data = buffer.get_batches(last_value, self.gamma, self.gae_lambda)
        obs = data["obs"].to(self.device)
        actions = data["actions"].to(self.device)
        old_log_probs = data["old_log_probs"].to(self.device)
        advantages = data["advantages"].to(self.device)
        returns = data["returns"].to(self.device)

        T, n_agents, obs_dim = obs.shape
        action_dim = actions.shape[-1]

        adv_flat = advantages.reshape(-1)
        if adv_flat.std() > 1e-8:
            adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)
        advantages = adv_flat.reshape(T, n_agents)

        global_states = obs.reshape(T, -1)

        total_pl, total_vl, total_ent = 0, 0, 0

        for _ in range(self.ppo_epochs):
            new_lp, entropy = self.actor.evaluate_action(
                obs.reshape(T * n_agents, obs_dim),
                actions.reshape(T * n_agents, action_dim),
            )
            new_lp = new_lp.reshape(T, n_agents)
            entropy = entropy.reshape(T, n_agents)

            ratio = torch.exp(new_lp - old_log_probs)
            s1 = ratio * advantages
            s2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
            pl = -torch.min(s1, s2).mean()
            el = -entropy.mean()

            self.actor_optimizer.zero_grad()
            (pl + self.entropy_coef * el).backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            self.actor_optimizer.step()

            vals = self.critic(global_states).squeeze(-1)
            vl = nn.functional.mse_loss(vals, returns)
            self.critic_optimizer.zero_grad()
            vl.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
            self.critic_optimizer.step()

            total_pl += pl.item()
            total_vl += vl.item()
            total_ent += entropy.mean().item()

        n = self.ppo_epochs
        return {"policy_loss": total_pl / n, "value_loss": total_vl / n, "entropy": total_ent / n}

    def save(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.actor.state_dict(), path / "actor.pt")
        torch.save(self.critic.state_dict(), path / "critic.pt")

    def load(self, path):
        path = Path(path)
        self.actor.load_state_dict(torch.load(path / "actor.pt", weights_only=True))
        self.critic.load_state_dict(torch.load(path / "critic.pt", weights_only=True))
