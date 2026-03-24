"""Train CommNet-MAPPO on Simple Spread."""

import sys
sys.path.insert(0, "/Users/sverker/repos/kextest")

import torch
import numpy as np
from pathlib import Path
from src.env_wrapper import SimpleSpreadEnv
from src.commnet import CommMAPPO
from src.buffer import RolloutBuffer


def train(
    n_agents=3,
    max_cycles=25,
    total_episodes=20000,
    episodes_per_batch=10,
    hidden_dim=128,
    n_comm_rounds=2,
    lr_actor=5e-4,
    lr_critic=5e-4,
    gamma=0.99,
    gae_lambda=0.95,
    clip_eps=0.2,
    entropy_coef=0.01,
    ppo_epochs=15,
    seed=42,
    log_interval=200,
    save_interval=2000,
    model_dir="models/commnet_baseline",
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = SimpleSpreadEnv(n_agents=n_agents, max_cycles=max_cycles)
    obs = env.reset(seed=seed)

    agent = CommMAPPO(
        obs_dim=env.obs_dim,
        global_state_dim=env.global_state_dim,
        n_actions=env.n_actions,
        n_agents=n_agents,
        hidden_dim=hidden_dim,
        n_comm_rounds=n_comm_rounds,
        lr_actor=lr_actor,
        lr_critic=lr_critic,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_eps=clip_eps,
        entropy_coef=entropy_coef,
        ppo_epochs=ppo_epochs,
    )

    all_rewards = []
    best_avg = -float("inf")
    ep_count = 0

    while ep_count < total_episodes:
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

        if done:
            last_value = 0.0
        else:
            _, _, last_value = agent.act(obs)

        if len(buffer) > 0:
            metrics = agent.update(buffer, last_value)

        all_rewards.extend(batch_rewards)

        if ep_count % log_interval < episodes_per_batch:
            window = min(log_interval, len(all_rewards))
            recent = all_rewards[-window:]
            avg = np.mean(recent)
            print(f"Episode {ep_count:6d} | Avg reward: {avg:.2f} | "
                  f"PL: {metrics['policy_loss']:.4f} | VL: {metrics['value_loss']:.4f}")

            if avg > best_avg:
                best_avg = avg
                agent.save(f"{model_dir}/best")
                print(f"  -> New best! ({avg:.2f})")

        if ep_count % save_interval < episodes_per_batch:
            agent.save(f"{model_dir}/checkpoint_{ep_count}")

    agent.save(f"{model_dir}/final")
    env.close()
    print(f"\nCommNet training complete. Best avg: {best_avg:.2f}")
    return all_rewards


if __name__ == "__main__":
    train()
