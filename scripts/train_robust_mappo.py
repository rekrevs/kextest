"""Train MAPPO with adversarial training (random observation perturbation during training)."""

import sys
sys.path.insert(0, "/Users/sverker/repos/kextest")

import torch
import numpy as np
from src.env_wrapper import SimpleSpreadEnv
from src.mappo import MAPPO
from src.buffer import RolloutBuffer
from src.countermeasures import adversarial_training_step


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
    adv_epsilon=0.2,
    adv_prob=0.5,
    seed=42,
    log_interval=200,
    save_interval=2000,
    model_dir="models/mappo_robust",
):
    """Train with random perturbations applied during data collection."""
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
                # Adversarial training: randomly perturb observations
                if np.random.random() < adv_prob:
                    obs_for_action = adversarial_training_step(agent.actor, obs, adv_epsilon)
                else:
                    obs_for_action = obs

                actions, log_probs, value = agent.act(obs_for_action)
                next_obs, reward, done, info = env.step(actions)

                if next_obs is None:
                    buffer.store(obs, actions, log_probs, reward if reward else 0.0, value, True)
                    break

                # Store clean obs for learning, but actions were based on perturbed
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

    agent.save(f"{model_dir}/final")
    env.close()
    print(f"\nRobust MAPPO training complete. Best avg: {best_avg:.2f}")
    return all_rewards


if __name__ == "__main__":
    train()
