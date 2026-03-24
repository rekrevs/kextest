"""Run Simple Spread with random agents to establish a lower baseline."""

import sys
sys.path.insert(0, "/Users/sverker/repos/kextest")

import numpy as np
from src.env_wrapper import SimpleSpreadEnv


def run_random(n_episodes=100, n_agents=3, max_cycles=25, seed=42):
    env = SimpleSpreadEnv(n_agents=n_agents, max_cycles=max_cycles)
    rng = np.random.default_rng(seed)

    episode_rewards = []

    for ep in range(n_episodes):
        obs = env.reset(seed=int(rng.integers(0, 100000)))
        total_reward = 0.0
        done = False

        while not done:
            actions = rng.integers(0, env.n_actions, size=env.n_agents)
            import torch
            obs, reward, done, info = env.step(torch.tensor(actions))
            if obs is None:
                break
            total_reward += reward

        episode_rewards.append(total_reward)

    env.close()

    rewards = np.array(episode_rewards)
    print(f"Random baseline over {n_episodes} episodes:")
    print(f"  Mean reward:   {rewards.mean():.2f}")
    print(f"  Std reward:    {rewards.std():.2f}")
    print(f"  Min reward:    {rewards.min():.2f}")
    print(f"  Max reward:    {rewards.max():.2f}")
    print(f"  Median reward: {np.median(rewards):.2f}")
    return rewards


if __name__ == "__main__":
    run_random()
