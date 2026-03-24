"""Train MAPPO on Simple Spread with N=5 agents and continuous actions."""

import sys
sys.path.insert(0, "/Users/sverker/repos/kextest")

import torch
import numpy as np
from src.env_continuous import SimpleSpreadContinuousEnv
from src.mappo_continuous import ContinuousMAPPO, ContinuousRolloutBuffer


def train(
    n_agents=5,
    max_cycles=25,
    total_episodes=20000,
    episodes_per_batch=10,
    hidden_dim=128,
    lr=5e-4,
    seed=42,
    log_interval=200,
    model_dir="models/mappo_continuous_n5",
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = SimpleSpreadContinuousEnv(n_agents=n_agents, max_cycles=max_cycles)
    obs = env.reset(seed=seed)

    agent = ContinuousMAPPO(
        obs_dim=env.obs_dim,
        global_state_dim=env.global_state_dim,
        action_dim=env.action_dim,
        n_agents=n_agents,
        hidden_dim=hidden_dim,
        lr_actor=lr,
        lr_critic=lr,
    )

    all_rewards = []
    best_avg = -float("inf")
    ep_count = 0

    while ep_count < total_episodes:
        buffer = ContinuousRolloutBuffer()
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

        last_value = 0.0 if done else agent.act(obs)[2]

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

    agent.save(f"{model_dir}/final")
    env.close()
    print(f"\nContinuous N={n_agents} training complete. Best avg: {best_avg:.2f}")
    return all_rewards


if __name__ == "__main__":
    train()
