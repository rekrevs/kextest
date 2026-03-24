"""Train an adversarial agent that tries to minimize team reward.

The adversarial agent replaces agent 0. The remaining agents use the
pre-trained MAPPO policy (frozen). The adversarial agent is trained
with PPO using the NEGATED team reward as its objective.
"""

import sys
sys.path.insert(0, "/Users/sverker/repos/kextest")

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter

from src.env_wrapper import SimpleSpreadEnv
from src.mappo import MAPPO
from src.networks import Actor


class AdversarialTrainer:
    """Trains a single adversarial agent against frozen cooperative agents."""

    def __init__(self, obs_dim, n_actions, hidden_dim=64, lr=3e-4,
                 gamma=0.99, gae_lambda=0.95, clip_eps=0.2,
                 entropy_coef=0.01, ppo_epochs=10):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.ppo_epochs = ppo_epochs

        self.actor = Actor(obs_dim, n_actions, hidden_dim)
        self.optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

    @torch.no_grad()
    def act(self, obs_single, deterministic=False):
        logits = self.actor(obs_single.unsqueeze(0)).squeeze(0)
        dist = Categorical(logits=logits)
        if deterministic:
            action = logits.argmax()
        else:
            action = dist.sample()
        return action, dist.log_prob(action)

    def update(self, obs_list, actions_list, old_log_probs_list, rewards_list, dones_list):
        """PPO update for the adversarial agent.

        All inputs are lists of per-step values from multiple episodes.
        """
        obs = torch.stack(obs_list)
        actions = torch.stack(actions_list)
        old_log_probs = torch.stack(old_log_probs_list)

        # Compute returns and advantages with GAE
        T = len(rewards_list)
        rewards = torch.tensor(rewards_list, dtype=torch.float32)
        dones = torch.tensor(dones_list, dtype=torch.float32)

        # Simple return computation (no value function baseline for simplicity)
        # Use discounted returns as advantage estimate
        returns = torch.zeros(T)
        running_return = 0.0
        for t in reversed(range(T)):
            running_return = rewards[t] + self.gamma * running_return * (1 - dones[t])
            returns[t] = running_return

        advantages = returns - returns.mean()
        if advantages.std() > 1e-8:
            advantages = advantages / (advantages.std() + 1e-8)

        total_loss = 0.0
        for _ in range(self.ppo_epochs):
            logits = self.actor(obs)
            dist = Categorical(logits=logits)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy()

            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            entropy_loss = -entropy.mean()

            loss = policy_loss + self.entropy_coef * entropy_loss

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / self.ppo_epochs

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.actor.state_dict(), path)

    def load(self, path):
        self.actor.load_state_dict(torch.load(path, weights_only=True))


def train_adversarial(
    cooperative_model_path="models/mappo_baseline/best",
    adv_agent_idx=0,
    n_agents=3,
    max_cycles=25,
    total_episodes=5000,
    episodes_per_batch=10,
    hidden_dim=64,
    lr=5e-4,
    seed=123,
    log_interval=200,
    model_dir="models/adversarial",
    log_dir="logs/adversarial",
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = SimpleSpreadEnv(n_agents=n_agents, max_cycles=max_cycles)
    obs = env.reset(seed=seed)

    # Load frozen cooperative agents
    coop = MAPPO(
        obs_dim=env.obs_dim,
        global_state_dim=env.global_state_dim,
        n_actions=env.n_actions,
        n_agents=n_agents,
        hidden_dim=128,
    )
    coop.load(cooperative_model_path)
    coop.actor.eval()
    for p in coop.actor.parameters():
        p.requires_grad = False

    # Adversarial agent trainer
    adv = AdversarialTrainer(
        obs_dim=env.obs_dim,
        n_actions=env.n_actions,
        hidden_dim=hidden_dim,
        lr=lr,
    )

    writer = SummaryWriter(log_dir)
    all_rewards = []
    best_min_reward = float("inf")  # Lower = better attack
    ep_count = 0

    while ep_count < total_episodes:
        # Collect batch
        batch_obs = []
        batch_actions = []
        batch_log_probs = []
        batch_rewards = []
        batch_dones = []
        batch_team_rewards = []

        for _ in range(episodes_per_batch):
            obs = env.reset(seed=seed + ep_count)
            done = False
            ep_team_reward = 0.0

            while not done:
                # Cooperative agents choose actions
                with torch.no_grad():
                    coop_actions, _, _ = coop.act(obs, deterministic=True)

                # Adversarial agent chooses its own action
                adv_action, adv_log_prob = adv.act(obs[adv_agent_idx])
                coop_actions[adv_agent_idx] = adv_action

                next_obs, team_reward, done, info = env.step(coop_actions)

                # Store adversarial agent's experience with NEGATED reward
                batch_obs.append(obs[adv_agent_idx].clone())
                batch_actions.append(adv_action)
                batch_log_probs.append(adv_log_prob)
                batch_rewards.append(-team_reward)  # NEGATE: adversary wants to minimize team reward
                batch_dones.append(done)

                if next_obs is None:
                    break
                obs = next_obs
                ep_team_reward += team_reward

            batch_team_rewards.append(ep_team_reward)
            ep_count += 1

        # Update adversarial agent
        loss = adv.update(batch_obs, batch_actions, batch_log_probs, batch_rewards, batch_dones)

        all_rewards.extend(batch_team_rewards)
        avg_team = np.mean(batch_team_rewards)
        writer.add_scalar("reward/team_under_attack", avg_team, ep_count)
        writer.add_scalar("loss/adversarial", loss, ep_count)

        if ep_count % log_interval < episodes_per_batch:
            window = min(log_interval, len(all_rewards))
            recent = all_rewards[-window:]
            avg = np.mean(recent)
            print(f"Episode {ep_count:6d} | Team reward under attack: {avg:.2f} | Loss: {loss:.4f}")

            if avg < best_min_reward:
                best_min_reward = avg
                adv.save(f"{model_dir}/best_adversary.pt")
                print(f"  -> New best attack! (team reward: {avg:.2f})")

    adv.save(f"{model_dir}/final_adversary.pt")
    writer.close()
    env.close()

    print(f"\nAdversarial training complete.")
    print(f"Best attack brought team reward down to: {best_min_reward:.2f}")
    return all_rewards


if __name__ == "__main__":
    train_adversarial()
