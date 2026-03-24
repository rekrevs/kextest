"""Train MAPPO on Simple Spread and log results."""

import sys
sys.path.insert(0, "/Users/sverker/repos/kextest")

import torch
import numpy as np
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

from src.env_wrapper import SimpleSpreadEnv
from src.mappo import MAPPO
from src.buffer import RolloutBuffer


def train(
    n_agents=3,
    max_cycles=25,
    total_episodes=20000,
    episodes_per_batch=10,
    hidden_dim=128,
    lr_actor=5e-4,
    lr_critic=5e-4,
    gamma=0.99,
    gae_lambda=0.95,
    clip_eps=0.2,
    entropy_coef=0.01,
    ppo_epochs=15,
    seed=42,
    save_interval=2000,
    log_interval=200,
    model_dir="models/mappo_baseline",
    log_dir="logs/mappo_baseline",
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = SimpleSpreadEnv(n_agents=n_agents, max_cycles=max_cycles)
    obs = env.reset(seed=seed)

    agent = MAPPO(
        obs_dim=env.obs_dim,
        global_state_dim=env.global_state_dim,
        n_actions=env.n_actions,
        n_agents=n_agents,
        hidden_dim=hidden_dim,
        lr_actor=lr_actor,
        lr_critic=lr_critic,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_eps=clip_eps,
        entropy_coef=entropy_coef,
        ppo_epochs=ppo_epochs,
    )

    writer = SummaryWriter(log_dir)
    model_path = Path(model_dir)

    all_episode_rewards = []
    best_avg_reward = -float("inf")
    ep_count = 0

    while ep_count < total_episodes:
        # Collect a batch of episodes
        buffer = RolloutBuffer()
        batch_rewards = []

        for _ in range(episodes_per_batch):
            obs = env.reset(seed=seed + ep_count)
            done = False
            ep_reward = 0.0

            while not done:
                actions, log_probs, value = agent.act(obs)
                next_obs, reward, done, info = env.step(actions)

                if next_obs is None:
                    buffer.store(obs, actions, log_probs, reward if reward else 0.0, value, True)
                    break

                buffer.store(obs, actions, log_probs, reward, value, done)
                obs = next_obs
                ep_reward += reward

            batch_rewards.append(ep_reward)
            ep_count += 1

            # Mark episode boundary (done=True) already stored above

        # Last value for GAE
        if done:
            last_value = 0.0
        else:
            _, _, last_value = agent.act(obs)

        # PPO update on full batch
        if len(buffer) > 0:
            metrics = agent.update(buffer, last_value)

        all_episode_rewards.extend(batch_rewards)

        # Logging per batch
        avg_batch = np.mean(batch_rewards)
        writer.add_scalar("reward/batch_avg", avg_batch, ep_count)
        if len(buffer) > 0:
            writer.add_scalar("loss/policy", metrics["policy_loss"], ep_count)
            writer.add_scalar("loss/value", metrics["value_loss"], ep_count)
            writer.add_scalar("loss/entropy", metrics["entropy"], ep_count)

        if ep_count % log_interval < episodes_per_batch:
            window = min(log_interval, len(all_episode_rewards))
            recent = all_episode_rewards[-window:]
            avg = np.mean(recent)
            std = np.std(recent)
            writer.add_scalar("reward/avg_recent", avg, ep_count)
            print(f"Episode {ep_count:6d} | Avg reward (last {window}): {avg:7.2f} +/- {std:.2f} | "
                  f"PL: {metrics['policy_loss']:.4f} | VL: {metrics['value_loss']:.4f} | "
                  f"Ent: {metrics['entropy']:.4f}")

            if avg > best_avg_reward:
                best_avg_reward = avg
                agent.save(model_path / "best")
                print(f"  -> New best! (avg: {avg:.2f})")

        if ep_count % save_interval < episodes_per_batch:
            agent.save(model_path / f"checkpoint_{ep_count}")

    # Save final model
    agent.save(model_path / "final")
    writer.close()
    env.close()

    print(f"\nTraining complete after {ep_count} episodes.")
    print(f"Best avg reward: {best_avg_reward:.2f}")
    print(f"Models saved to {model_path}")

    return all_episode_rewards


if __name__ == "__main__":
    train()
