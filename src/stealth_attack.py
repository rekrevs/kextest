"""Stealth adversarial agent: minimize team reward while staying undetectable.

The adversarial agent is trained with a dual objective:
    max -R_team (minimize team reward)
    s.t. KL(pi_adv || pi_coop) < kl_threshold

This is implemented as a Lagrangian relaxation:
    L = -R_team + lambda * max(0, KL(pi_adv || pi_coop) - kl_threshold)

The agent learns to sabotage the team while keeping its action distribution
close to the cooperative policy, making it harder to detect.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from torch.distributions import Categorical

from .networks import Actor


class StealthAdversarialTrainer:
    """Trains a stealth adversarial agent with KL constraint."""

    def __init__(
        self,
        obs_dim,
        n_actions,
        cooperative_actor,
        hidden_dim=64,
        lr=3e-4,
        gamma=0.99,
        clip_eps=0.2,
        entropy_coef=0.01,
        ppo_epochs=10,
        kl_threshold=0.5,
        lambda_lr=1e-3,
        lambda_init=1.0,
    ):
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.ppo_epochs = ppo_epochs
        self.kl_threshold = kl_threshold

        # Adversarial actor
        self.actor = Actor(obs_dim, n_actions, hidden_dim)
        self.optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        # Frozen cooperative actor for KL reference
        self.coop_actor = cooperative_actor
        for p in self.coop_actor.parameters():
            p.requires_grad = False

        # Lagrange multiplier for KL constraint
        self.log_lambda = torch.tensor(np.log(lambda_init), requires_grad=True)
        self.lambda_optimizer = torch.optim.Adam([self.log_lambda], lr=lambda_lr)

    @property
    def lam(self):
        return self.log_lambda.exp()

    @torch.no_grad()
    def act(self, obs_single, deterministic=False):
        logits = self.actor(obs_single.unsqueeze(0)).squeeze(0)
        dist = Categorical(logits=logits)
        if deterministic:
            action = logits.argmax()
        else:
            action = dist.sample()
        return action, dist.log_prob(action)

    def compute_kl(self, obs):
        """Compute KL(pi_adv || pi_coop) for given observations."""
        adv_logits = self.actor(obs)
        coop_logits = self.coop_actor(obs)

        adv_dist = Categorical(logits=adv_logits)
        coop_dist = Categorical(logits=coop_logits)

        kl = torch.distributions.kl_divergence(adv_dist, coop_dist)
        return kl.mean()

    def update(self, obs_list, actions_list, old_log_probs_list, rewards_list, dones_list):
        """PPO update with KL constraint via Lagrangian relaxation."""
        obs = torch.stack(obs_list)
        actions = torch.stack(actions_list)
        old_log_probs = torch.stack(old_log_probs_list)

        T = len(rewards_list)
        rewards = torch.tensor(rewards_list, dtype=torch.float32)
        dones = torch.tensor(dones_list, dtype=torch.float32)

        # Compute discounted returns
        returns = torch.zeros(T)
        running_return = 0.0
        for t in reversed(range(T)):
            running_return = rewards[t] + self.gamma * running_return * (1 - dones[t])
            returns[t] = running_return

        advantages = returns - returns.mean()
        if advantages.std() > 1e-8:
            advantages = advantages / (advantages.std() + 1e-8)

        total_policy_loss = 0.0
        total_kl = 0.0

        for _ in range(self.ppo_epochs):
            logits = self.actor(obs)
            dist = Categorical(logits=logits)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy()

            # PPO clipped objective (for adversarial reward)
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            entropy_loss = -entropy.mean()

            # KL constraint
            kl = self.compute_kl(obs)

            # Total loss: adversarial objective + lambda * KL penalty
            actor_loss = (
                policy_loss
                + self.entropy_coef * entropy_loss
                + self.lam.detach() * torch.relu(kl - self.kl_threshold)
            )

            self.optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            self.optimizer.step()

            # Update Lagrange multiplier (dual ascent)
            lambda_loss = -self.lam * (kl.detach() - self.kl_threshold)
            self.lambda_optimizer.zero_grad()
            lambda_loss.backward()
            self.lambda_optimizer.step()

            total_policy_loss += policy_loss.item()
            total_kl += kl.item()

        n = self.ppo_epochs
        return {
            "policy_loss": total_policy_loss / n,
            "kl": total_kl / n,
            "lambda": self.lam.item(),
        }

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.actor.state_dict(), path)

    def load(self, path):
        self.actor.load_state_dict(torch.load(path, weights_only=True))
