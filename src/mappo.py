"""MAPPO trainer: Multi-Agent PPO with centralized critic."""

import torch
import torch.nn as nn
from pathlib import Path

from .networks import Actor, Critic
from .buffer import RolloutBuffer


class MAPPO:
    """Multi-Agent PPO with shared actor parameters and centralized critic."""

    def __init__(
        self,
        obs_dim,
        global_state_dim,
        n_actions,
        n_agents,
        hidden_dim=64,
        lr_actor=3e-4,
        lr_critic=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.2,
        entropy_coef=0.01,
        value_coef=0.5,
        max_grad_norm=0.5,
        ppo_epochs=10,
        device="cpu",
    ):
        self.n_agents = n_agents
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.device = device

        # Shared actor (parameter sharing across agents)
        self.actor = Actor(obs_dim, n_actions, hidden_dim).to(device)
        # Centralized critic
        self.critic = Critic(global_state_dim, hidden_dim).to(device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

    @torch.no_grad()
    def act(self, obs, deterministic=False):
        """Select actions for all agents.

        Args:
            obs: (n_agents, obs_dim)
            deterministic: if True, take argmax action

        Returns:
            actions: (n_agents,)
            log_probs: (n_agents,)
            value: scalar
        """
        actions, log_probs, _ = self.actor.get_action(obs, deterministic=deterministic)
        global_state = obs.reshape(1, -1)  # (1, n_agents * obs_dim)
        value = self.critic(global_state).squeeze()
        return actions, log_probs, value.item()

    def update(self, buffer: RolloutBuffer, last_value: float):
        """Run PPO update on collected rollout data.

        Returns dict of training metrics.
        """
        data = buffer.get_batches(last_value, self.gamma, self.gae_lambda)
        obs = data["obs"].to(self.device)             # (T, n_agents, obs_dim)
        actions = data["actions"].to(self.device)       # (T, n_agents)
        old_log_probs = data["old_log_probs"].to(self.device)  # (T, n_agents)
        advantages = data["advantages"].to(self.device) # (T, n_agents)
        returns = data["returns"].to(self.device)       # (T,)

        T, n_agents, obs_dim = obs.shape

        # Normalize advantages
        adv_flat = advantages.reshape(-1)
        if adv_flat.std() > 1e-8:
            adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)
        advantages = adv_flat.reshape(T, n_agents)

        # Global states for critic
        global_states = obs.reshape(T, -1)  # (T, n_agents * obs_dim)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0

        for _ in range(self.ppo_epochs):
            # Actor: evaluate all agents' actions
            new_log_probs, entropy = self.actor.evaluate_action(
                obs.reshape(T * n_agents, obs_dim),
                actions.reshape(T * n_agents),
            )
            new_log_probs = new_log_probs.reshape(T, n_agents)
            entropy = entropy.reshape(T, n_agents)

            # PPO clipped objective
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            entropy_loss = -entropy.mean()

            actor_loss = policy_loss + self.entropy_coef * entropy_loss

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.actor_optimizer.step()

            # Critic
            values = self.critic(global_states).squeeze(-1)  # (T,)
            value_loss = nn.functional.mse_loss(values, returns)

            self.critic_optimizer.zero_grad()
            value_loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.critic_optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.mean().item()

        n_updates = self.ppo_epochs
        return {
            "policy_loss": total_policy_loss / n_updates,
            "value_loss": total_value_loss / n_updates,
            "entropy": total_entropy / n_updates,
        }

    def save(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.actor.state_dict(), path / "actor.pt")
        torch.save(self.critic.state_dict(), path / "critic.pt")

    def load(self, path):
        path = Path(path)
        self.actor.load_state_dict(torch.load(path / "actor.pt", weights_only=True))
        self.critic.load_state_dict(torch.load(path / "critic.pt", weights_only=True))
