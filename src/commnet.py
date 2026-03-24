"""CommNet: Learning Multiagent Communication with Backpropagation.

Extension of MAPPO where agents share learned communication messages.
Each agent encodes its observation, broadcasts a message, receives the
mean of other agents' messages, and uses this to make decisions.
"""

import torch
import torch.nn as nn
from torch.distributions import Categorical


class CommNetActor(nn.Module):
    """Actor with communication between agents.

    Architecture:
    1. Encode observation -> hidden state
    2. Communication round(s): hidden += mean(other agents' hidden)
    3. Decode hidden -> action logits
    """

    def __init__(self, obs_dim, n_actions, n_agents, hidden_dim=64, n_comm_rounds=2):
        super().__init__()
        self.n_agents = n_agents
        self.n_comm_rounds = n_comm_rounds
        self.hidden_dim = hidden_dim

        # Encoder: obs -> hidden
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
        )

        # Communication layers (one per round)
        self.comm_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(n_comm_rounds)
        ])

        # Update layers after receiving message (one per round)
        self.update_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Tanh(),
            ) for _ in range(n_comm_rounds)
        ])

        # Decoder: hidden -> action logits
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, obs):
        """obs: (n_agents, obs_dim) or (batch, n_agents, obs_dim)

        Returns: logits (same leading dims, n_actions)
        """
        single = obs.dim() == 2
        if single:
            obs = obs.unsqueeze(0)  # (1, n_agents, obs_dim)

        batch_size = obs.size(0)
        n_agents = obs.size(1)

        # Encode
        h = self.encoder(obs)  # (batch, n_agents, hidden_dim)

        # Communication rounds
        for r in range(self.n_comm_rounds):
            # Generate messages
            messages = self.comm_layers[r](h)  # (batch, n_agents, hidden_dim)

            # Each agent receives mean of OTHER agents' messages
            msg_sum = messages.sum(dim=1, keepdim=True)  # (batch, 1, hidden_dim)
            # Subtract own message, divide by (n-1)
            received = (msg_sum - messages) / max(n_agents - 1, 1)

            # Update hidden with received message
            h = self.update_layers[r](torch.cat([h, received], dim=-1))

        logits = self.decoder(h)  # (batch, n_agents, n_actions)

        if single:
            logits = logits.squeeze(0)

        return logits

    def get_action(self, obs, deterministic=False):
        """obs: (n_agents, obs_dim) -> actions, log_probs, entropy"""
        logits = self.forward(obs)
        dist = Categorical(logits=logits)
        if deterministic:
            actions = logits.argmax(dim=-1)
        else:
            actions = dist.sample()
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return actions, log_probs, entropy

    def evaluate_action(self, obs, actions):
        """Evaluate actions under current policy.

        obs: (T*n_agents, obs_dim) -- needs reshaping
        actions: (T*n_agents,)
        """
        # We need to handle the flat input - reshape for communication
        # This is called with already-flattened tensors from MAPPO update
        # We process per-agent independently (no communication during training update)
        # For proper CommNet training, we need the grouped format
        logits = self.forward_flat(obs)
        dist = Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy()

    def forward_flat(self, obs):
        """Process observations without communication (for compatibility with MAPPO update).

        obs: (batch, obs_dim)
        """
        h = self.encoder(obs)
        # Skip communication - just use zero messages
        for r in range(self.n_comm_rounds):
            zero_msg = torch.zeros_like(h)
            h = self.update_layers[r](torch.cat([h, zero_msg], dim=-1))
        return self.decoder(h)


class CommMAPPO:
    """MAPPO variant with CommNet actor."""

    def __init__(
        self,
        obs_dim,
        global_state_dim,
        n_actions,
        n_agents,
        hidden_dim=64,
        n_comm_rounds=2,
        lr_actor=3e-4,
        lr_critic=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.2,
        entropy_coef=0.01,
        ppo_epochs=10,
        max_grad_norm=0.5,
        device="cpu",
    ):
        from .networks import Critic

        self.n_agents = n_agents
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.ppo_epochs = ppo_epochs
        self.max_grad_norm = max_grad_norm
        self.device = device

        self.actor = CommNetActor(obs_dim, n_actions, n_agents, hidden_dim, n_comm_rounds).to(device)
        self.critic = Critic(global_state_dim, hidden_dim).to(device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

    @torch.no_grad()
    def act(self, obs, deterministic=False):
        """obs: (n_agents, obs_dim)"""
        actions, log_probs, _ = self.actor.get_action(obs, deterministic)
        global_state = obs.reshape(1, -1)
        value = self.critic(global_state).squeeze()
        return actions, log_probs, value.item()

    def update(self, buffer, last_value):
        """PPO update using CommNet actor with full communication."""
        from .buffer import RolloutBuffer

        data = buffer.get_batches(last_value, self.gamma, self.gae_lambda)
        obs = data["obs"].to(self.device)             # (T, n_agents, obs_dim)
        actions = data["actions"].to(self.device)       # (T, n_agents)
        old_log_probs = data["old_log_probs"].to(self.device)
        advantages = data["advantages"].to(self.device)
        returns = data["returns"].to(self.device)

        T, n_agents, obs_dim = obs.shape

        # Normalize advantages
        adv_flat = advantages.reshape(-1)
        if adv_flat.std() > 1e-8:
            adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)
        advantages = adv_flat.reshape(T, n_agents)

        global_states = obs.reshape(T, -1)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0

        for _ in range(self.ppo_epochs):
            # CommNet forward with communication (batch of timesteps)
            logits = self.actor(obs)  # (T, n_agents, n_actions)
            dist = Categorical(logits=logits)
            new_log_probs = dist.log_prob(actions)  # (T, n_agents)
            entropy = dist.entropy()

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

            values = self.critic(global_states).squeeze(-1)
            value_loss = nn.functional.mse_loss(values, returns)

            self.critic_optimizer.zero_grad()
            value_loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.critic_optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.mean().item()

        n = self.ppo_epochs
        return {
            "policy_loss": total_policy_loss / n,
            "value_loss": total_value_loss / n,
            "entropy": total_entropy / n,
        }

    def save(self, path):
        from pathlib import Path as P
        path = P(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.actor.state_dict(), path / "actor.pt")
        torch.save(self.critic.state_dict(), path / "critic.pt")

    def load(self, path):
        from pathlib import Path as P
        path = P(path)
        self.actor.load_state_dict(torch.load(path / "actor.pt", weights_only=True))
        self.critic.load_state_dict(torch.load(path / "critic.pt", weights_only=True))
