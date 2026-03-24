"""Visualize attack results with plots."""

import sys
sys.path.insert(0, "/Users/sverker/repos/kextest")

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.env_wrapper import SimpleSpreadEnv
from src.mappo import MAPPO
from src.attacks import FGSMAttack, IterativeFGSMAttack, RandomNoiseAttack, AdversarialAgent
from src.networks import Actor
from scripts.evaluate_attacks import (
    evaluate_no_attack,
    evaluate_observation_attack,
    evaluate_adversarial_agent,
)


def plot_epsilon_sweep(agent, env, n_episodes=100, save_path="results/epsilon_sweep.png"):
    """Plot reward vs epsilon for different attack methods."""
    epsilons = [0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]

    clean = evaluate_no_attack(agent, env, n_episodes)
    clean_mean = clean.mean()

    results = {
        "Random Noise": [],
        "FGSM": [],
        "Iterative FGSM (PGD)": [],
    }

    for eps in epsilons:
        if eps == 0.0:
            for key in results:
                results[key].append(clean_mean)
        else:
            rn = RandomNoiseAttack(epsilon=eps)
            fgsm = FGSMAttack(agent.actor, epsilon=eps)
            ifgsm = IterativeFGSMAttack(agent.actor, epsilon=eps, n_steps=10)

            results["Random Noise"].append(
                evaluate_observation_attack(agent, env, rn, n_episodes).mean()
            )
            results["FGSM"].append(
                evaluate_observation_attack(agent, env, fgsm, n_episodes).mean()
            )
            results["Iterative FGSM (PGD)"].append(
                evaluate_observation_attack(agent, env, ifgsm, n_episodes).mean()
            )
            print(f"  eps={eps:.2f} done")

    fig, ax = plt.subplots(figsize=(10, 6))
    for label, values in results.items():
        ax.plot(epsilons, values, marker="o", label=label)

    ax.axhline(y=clean_mean, color="green", linestyle="--", alpha=0.5, label="Clean baseline")
    ax.set_xlabel("Perturbation epsilon", fontsize=12)
    ax.set_ylabel("Mean Team Reward", fontsize=12)
    ax.set_title("Effect of Observation Perturbation Attacks on MAPPO", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_agents_attacked(agent, env, n_episodes=100, save_path="results/agents_attacked.png"):
    """Plot reward vs number of agents attacked."""
    clean = evaluate_no_attack(agent, env, n_episodes)
    clean_mean = clean.mean()

    epsilons = [0.1, 0.2, 0.5]
    n_agents_list = [0, 1, 2, 3]

    fig, ax = plt.subplots(figsize=(10, 6))

    for eps in epsilons:
        attack = FGSMAttack(agent.actor, epsilon=eps)
        means = [clean_mean]
        for n in range(1, 4):
            targets = list(range(n))
            rews = evaluate_observation_attack(agent, env, attack, n_episodes, target_agents=targets)
            means.append(rews.mean())
        ax.plot(n_agents_list, means, marker="s", label=f"FGSM eps={eps}")

    ax.set_xlabel("Number of Agents Attacked", fontsize=12)
    ax.set_ylabel("Mean Team Reward", fontsize=12)
    ax.set_title("Impact of Attacking Different Numbers of Agents", fontsize=14)
    ax.set_xticks([0, 1, 2, 3])
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_attack_comparison(agent, env, adv_actor_path=None, n_episodes=100,
                           save_path="results/attack_comparison.png"):
    """Bar chart comparing all attack types."""
    clean = evaluate_no_attack(agent, env, n_episodes)

    attack_results = {"Clean": clean}

    # Observation attacks with eps=0.2
    fgsm = FGSMAttack(agent.actor, epsilon=0.2)
    attack_results["FGSM (eps=0.2)"] = evaluate_observation_attack(agent, env, fgsm, n_episodes)

    ifgsm = IterativeFGSMAttack(agent.actor, epsilon=0.2, n_steps=10)
    attack_results["PGD (eps=0.2)"] = evaluate_observation_attack(agent, env, ifgsm, n_episodes)

    rn = RandomNoiseAttack(epsilon=0.2)
    attack_results["Random (eps=0.2)"] = evaluate_observation_attack(agent, env, rn, n_episodes)

    # Adversarial agent
    if adv_actor_path and Path(adv_actor_path).exists():
        adv_actor = Actor(env.obs_dim, env.n_actions, hidden_dim=64)
        adv_actor.load_state_dict(torch.load(adv_actor_path, weights_only=True))
        adv_actor.eval()
        adv_agent = AdversarialAgent(adv_actor)
        attack_results["Adversarial Agent"] = evaluate_adversarial_agent(
            agent, env, adv_agent, adv_idx=0, n_episodes=n_episodes
        )

    # Random agent as adversary baseline
    from src.networks import Actor as ActorNet
    random_actor = ActorNet(env.obs_dim, env.n_actions, hidden_dim=64)  # untrained
    random_adv = AdversarialAgent(random_actor)
    attack_results["Random Agent Replace"] = evaluate_adversarial_agent(
        agent, env, random_adv, adv_idx=0, n_episodes=n_episodes
    )

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    names = list(attack_results.keys())
    means = [v.mean() for v in attack_results.values()]
    stds = [v.std() for v in attack_results.values()]

    colors = ["green"] + ["steelblue"] * (len(names) - 1)
    bars = ax.bar(names, means, yerr=stds, capsize=5, color=colors, alpha=0.8)
    ax.set_ylabel("Mean Team Reward", fontsize=12)
    ax.set_title("Attack Comparison on MAPPO Simple Spread", fontsize=14)
    ax.grid(True, alpha=0.3, axis="y")
    plt.xticks(rotation=30, ha="right")

    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def main(model_path="models/mappo_baseline/best", adv_model_path="models/adversarial/best_adversary.pt"):
    env = SimpleSpreadEnv(n_agents=3, max_cycles=25)
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

    print("Generating plots...")

    print("\n1/3: Epsilon sweep")
    plot_epsilon_sweep(agent, env)

    print("\n2/3: Agents attacked")
    plot_agents_attacked(agent, env)

    print("\n3/3: Attack comparison")
    plot_attack_comparison(agent, env, adv_actor_path=adv_model_path)

    env.close()
    print("\nAll plots saved to results/")


if __name__ == "__main__":
    main()
