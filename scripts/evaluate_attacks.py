"""Evaluate adversarial attacks against trained MAPPO agents."""

import sys
sys.path.insert(0, "/Users/sverker/repos/kextest")

import torch
import numpy as np
from pathlib import Path

from src.env_wrapper import SimpleSpreadEnv
from src.mappo import MAPPO
from src.attacks import FGSMAttack, IterativeFGSMAttack, RandomNoiseAttack


def evaluate_no_attack(agent, env, n_episodes=100, seed=0):
    """Evaluate clean (no attack) performance."""
    rewards = []
    for ep in range(n_episodes):
        obs = env.reset(seed=seed + ep)
        done = False
        ep_reward = 0.0
        while not done:
            actions, _, _ = agent.act(obs, deterministic=True)
            obs, reward, done, info = env.step(actions)
            if obs is None:
                break
            ep_reward += reward
        rewards.append(ep_reward)
    return np.array(rewards)


def evaluate_observation_attack(agent, env, attack, n_episodes=100, target_agents=None, seed=0):
    """Evaluate with observation perturbation attack."""
    rewards = []
    for ep in range(n_episodes):
        obs = env.reset(seed=seed + ep)
        done = False
        ep_reward = 0.0
        while not done:
            # Attack: perturb observations before agent decision
            obs_perturbed = attack.perturb(obs, target_agents=target_agents)
            actions, _, _ = agent.act(obs_perturbed, deterministic=True)
            obs, reward, done, info = env.step(actions)
            if obs is None:
                break
            ep_reward += reward
        rewards.append(ep_reward)
    return np.array(rewards)


def evaluate_adversarial_agent(agent, env, adv_agent, adv_idx=0, n_episodes=100, seed=0):
    """Evaluate with one agent replaced by adversarial agent."""
    rewards = []
    for ep in range(n_episodes):
        obs = env.reset(seed=seed + ep)
        done = False
        ep_reward = 0.0
        while not done:
            # Normal agents act from trained policy
            actions, _, _ = agent.act(obs, deterministic=True)
            # Override adversarial agent's action
            actions[adv_idx] = adv_agent.act(obs[adv_idx], deterministic=True)
            obs, reward, done, info = env.step(actions)
            if obs is None:
                break
            ep_reward += reward
        rewards.append(ep_reward)
    return np.array(rewards)


def main(model_path="models/mappo_baseline/best", n_episodes=100):
    env = SimpleSpreadEnv(n_agents=3, max_cycles=25)

    # Initialize and load trained agent
    obs = env.reset(seed=0)
    agent = MAPPO(
        obs_dim=env.obs_dim,
        global_state_dim=env.global_state_dim,
        n_actions=env.n_actions,
        n_agents=3,
        hidden_dim=128,
    )
    agent.load(model_path)
    agent.actor.eval()
    agent.critic.eval()

    print("=" * 70)
    print("ADVERSARIAL ATTACK EVALUATION ON MAPPO")
    print("=" * 70)

    # 1. Clean baseline
    print("\n--- Clean (No Attack) ---")
    clean = evaluate_no_attack(agent, env, n_episodes)
    print(f"  Mean: {clean.mean():.2f} +/- {clean.std():.2f}")

    # 2. Random noise baseline (for comparison)
    print("\n--- Random Noise Attack ---")
    for eps in [0.05, 0.1, 0.2, 0.5, 1.0]:
        attack = RandomNoiseAttack(epsilon=eps)
        rews = evaluate_observation_attack(agent, env, attack, n_episodes)
        pct = (1 - rews.mean() / clean.mean()) * 100
        print(f"  eps={eps:.2f}: Mean={rews.mean():.2f} +/- {rews.std():.2f} (reward change: {pct:+.1f}%)")

    # 3. FGSM attack
    print("\n--- FGSM Attack (all agents) ---")
    for eps in [0.05, 0.1, 0.2, 0.5, 1.0]:
        attack = FGSMAttack(agent.actor, epsilon=eps)
        rews = evaluate_observation_attack(agent, env, attack, n_episodes)
        pct = (1 - rews.mean() / clean.mean()) * 100
        print(f"  eps={eps:.2f}: Mean={rews.mean():.2f} +/- {rews.std():.2f} (reward change: {pct:+.1f}%)")

    # 4. FGSM attack on single agent
    print("\n--- FGSM Attack (agent 0 only) ---")
    for eps in [0.05, 0.1, 0.2, 0.5, 1.0]:
        attack = FGSMAttack(agent.actor, epsilon=eps)
        rews = evaluate_observation_attack(agent, env, attack, n_episodes, target_agents=[0])
        pct = (1 - rews.mean() / clean.mean()) * 100
        print(f"  eps={eps:.2f}: Mean={rews.mean():.2f} +/- {rews.std():.2f} (reward change: {pct:+.1f}%)")

    # 5. Iterative FGSM (PGD) attack
    print("\n--- Iterative FGSM / PGD Attack (all agents, 10 steps) ---")
    for eps in [0.05, 0.1, 0.2, 0.5, 1.0]:
        attack = IterativeFGSMAttack(agent.actor, epsilon=eps, n_steps=10)
        rews = evaluate_observation_attack(agent, env, attack, n_episodes)
        pct = (1 - rews.mean() / clean.mean()) * 100
        print(f"  eps={eps:.2f}: Mean={rews.mean():.2f} +/- {rews.std():.2f} (reward change: {pct:+.1f}%)")

    # 6. Number of attacked agents analysis
    print("\n--- FGSM eps=0.2: Varying number of attacked agents ---")
    attack = FGSMAttack(agent.actor, epsilon=0.2)
    for n_attacked in range(1, 4):
        targets = list(range(n_attacked))
        rews = evaluate_observation_attack(agent, env, attack, n_episodes, target_agents=targets)
        pct = (1 - rews.mean() / clean.mean()) * 100
        print(f"  {n_attacked} agent(s) attacked: Mean={rews.mean():.2f} +/- {rews.std():.2f} "
              f"(reward change: {pct:+.1f}%)")

    env.close()
    print("\n" + "=" * 70)
    print("Evaluation complete.")


if __name__ == "__main__":
    main()
