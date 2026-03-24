"""Deep analysis: trajectory visualization, action distributions, ablation studies."""

import sys
sys.path.insert(0, "/Users/sverker/repos/kextest")

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from pathlib import Path
from torch.distributions import Categorical

from src.env_wrapper import SimpleSpreadEnv
from src.mappo import MAPPO
from src.attacks import FGSMAttack, IterativeFGSMAttack


def collect_trajectory(agent, env, attack=None, target_agents=None, seed=42):
    """Collect one episode trajectory for visualization."""
    obs = env.reset(seed=seed)
    positions = [[] for _ in range(env.n_agents)]
    landmark_pos = None  # will extract from first obs

    done = False
    while not done:
        # Extract positions from observations
        # In Simple Spread, obs = [self_vel(2), self_pos(2), landmark_rel_pos(2*N), other_agent_rel_pos(2*(N-1))]
        for i in range(env.n_agents):
            pos = obs[i, 2:4].detach().numpy()
            positions[i].append(pos.copy())

        # Extract landmark positions from first agent's perspective
        if landmark_pos is None:
            self_pos = obs[0, 2:4].detach().numpy()
            landmark_pos = []
            for j in range(env.n_agents):
                rel = obs[0, 4 + 2 * j:4 + 2 * (j + 1)].detach().numpy()
                landmark_pos.append(self_pos + rel)

        obs_for_action = obs
        if attack is not None:
            obs_for_action = attack.perturb(obs, target_agents=target_agents)

        actions, _, _ = agent.act(obs_for_action, deterministic=True)
        obs, reward, done, info = env.step(actions)
        if obs is None:
            break

    return positions, landmark_pos


def plot_trajectories(agent, env, save_dir="results"):
    """Plot agent trajectories: clean vs attacked."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    scenarios = [
        ("Clean", None, None),
        ("FGSM eps=0.5", FGSMAttack(agent.actor, epsilon=0.5), None),
        ("FGSM eps=0.5 (agent 0)", FGSMAttack(agent.actor, epsilon=0.5), [0]),
    ]

    colors = ["tab:blue", "tab:orange", "tab:green"]

    for ax, (title, attack, targets) in zip(axes, scenarios):
        positions, landmarks = collect_trajectory(agent, env, attack, targets, seed=42)

        # Plot landmarks
        for j, lm in enumerate(landmarks):
            ax.plot(lm[0], lm[1], "k*", markersize=15, zorder=5)
            ax.add_patch(Circle(lm, 0.05, color="gray", alpha=0.3))

        # Plot trajectories
        for i, (traj, color) in enumerate(zip(positions, colors)):
            traj = np.array(traj)
            ax.plot(traj[:, 0], traj[:, 1], "-", color=color, alpha=0.7, linewidth=2,
                    label=f"Agent {i}")
            ax.plot(traj[0, 0], traj[0, 1], "o", color=color, markersize=8)
            ax.plot(traj[-1, 0], traj[-1, 1], "s", color=color, markersize=10)

        ax.set_title(title, fontsize=14)
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    fig.suptitle("Agent Trajectories Under Different Attack Scenarios", fontsize=16)
    plt.tight_layout()
    path = f"{save_dir}/trajectories.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_action_distributions(agent, env, save_dir="results"):
    """Compare action distributions under clean vs attacked observations."""
    obs = env.reset(seed=42)
    action_names = ["No-op", "Left", "Right", "Down", "Up"]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    attacks = [
        ("Clean", None),
        ("FGSM eps=0.2", FGSMAttack(agent.actor, epsilon=0.2)),
        ("FGSM eps=1.0", FGSMAttack(agent.actor, epsilon=1.0)),
    ]

    for col, (title, attack) in enumerate(attacks):
        obs_used = obs if attack is None else attack.perturb(obs)

        for row in range(2):  # agent 0 and agent 1
            ax = axes[row, col]
            logits = agent.actor(obs_used[row].unsqueeze(0)).squeeze(0)
            probs = torch.softmax(logits, dim=-1).detach().numpy()

            bars = ax.bar(action_names, probs, color="steelblue", alpha=0.8)
            ax.set_ylim(0, 1)
            ax.set_ylabel("Probability")
            ax.set_title(f"Agent {row} - {title}", fontsize=11)

            # Highlight the chosen action
            best = probs.argmax()
            bars[best].set_color("coral")

    fig.suptitle("Action Probability Distributions Under Attack", fontsize=14)
    plt.tight_layout()
    path = f"{save_dir}/action_distributions.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def analyze_fgsm_vs_pgd(agent, env, n_episodes=50, save_dir="results"):
    """Analyze WHY PGD was weaker than FGSM in our experiments.

    Hypothesis: In discrete action spaces, PGD steps oscillate around
    decision boundaries without crossing them, while FGSM's single
    large step crosses more boundaries.
    """
    from scripts.evaluate_attacks import evaluate_observation_attack

    epsilons = [0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]

    # Count action changes (how many actions differ from clean)
    fgsm_changes = []
    pgd_changes = []

    for eps in epsilons:
        fgsm_attack = FGSMAttack(agent.actor, epsilon=eps)
        pgd_attack = IterativeFGSMAttack(agent.actor, epsilon=eps, n_steps=10)

        fgsm_changed = 0
        pgd_changed = 0
        total = 0

        for ep in range(n_episodes):
            obs = env.reset(seed=ep)
            done = False
            while not done:
                clean_actions = agent.actor(obs).argmax(dim=-1)

                fgsm_obs = fgsm_attack.perturb(obs)
                fgsm_actions = agent.actor(fgsm_obs).argmax(dim=-1)

                pgd_obs = pgd_attack.perturb(obs)
                pgd_actions = agent.actor(pgd_obs).argmax(dim=-1)

                fgsm_changed += (fgsm_actions != clean_actions).sum().item()
                pgd_changed += (pgd_actions != clean_actions).sum().item()
                total += clean_actions.numel()

                actions, _, _ = agent.act(obs, deterministic=True)
                obs, _, done, _ = env.step(actions)
                if obs is None:
                    break

        fgsm_changes.append(fgsm_changed / total)
        pgd_changes.append(pgd_changed / total)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epsilons, fgsm_changes, "o-", label="FGSM", linewidth=2)
    ax.plot(epsilons, pgd_changes, "s-", label="PGD (10 steps)", linewidth=2)
    ax.set_xlabel("Perturbation epsilon", fontsize=12)
    ax.set_ylabel("Fraction of Actions Changed", fontsize=12)
    ax.set_title("FGSM vs PGD: How Often Do Actions Actually Change?", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    path = f"{save_dir}/fgsm_vs_pgd_analysis.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_perturbation_heatmap(agent, env, save_dir="results"):
    """Visualize which observation dimensions are most perturbed by FGSM."""
    obs = env.reset(seed=42)
    obs_adv = obs.clone().detach().requires_grad_(True)

    logits = agent.actor(obs_adv)
    best_actions = logits.argmax(dim=-1).detach()
    dist = Categorical(logits=logits)
    loss = dist.log_prob(best_actions).sum()
    loss.backward()

    grad = obs_adv.grad.abs().detach().numpy()

    fig, ax = plt.subplots(figsize=(14, 4))
    im = ax.imshow(grad, aspect="auto", cmap="hot")
    ax.set_ylabel("Agent")
    ax.set_xlabel("Observation Dimension")
    ax.set_yticks(range(env.n_agents))
    ax.set_yticklabels([f"Agent {i}" for i in range(env.n_agents)])
    ax.set_title("Gradient Magnitude per Observation Dimension (FGSM Sensitivity)", fontsize=13)

    # Label observation dimensions
    obs_labels = ["vel_x", "vel_y", "pos_x", "pos_y"]
    for i in range(env.n_agents):
        obs_labels.extend([f"lm{i}_dx", f"lm{i}_dy"])
    for i in range(env.n_agents - 1):
        obs_labels.extend([f"ag{i}_dx", f"ag{i}_dy"])

    if len(obs_labels) <= grad.shape[1]:
        ax.set_xticks(range(len(obs_labels)))
        ax.set_xticklabels(obs_labels, rotation=45, ha="right", fontsize=8)

    plt.colorbar(im, ax=ax, label="Gradient magnitude")
    plt.tight_layout()
    path = f"{save_dir}/gradient_heatmap.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def main(model_path="models/mappo_baseline/best"):
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

    Path("results").mkdir(exist_ok=True)

    print("1/4: Trajectories")
    plot_trajectories(agent, env)

    print("2/4: Action distributions")
    plot_action_distributions(agent, env)

    print("3/4: FGSM vs PGD analysis")
    analyze_fgsm_vs_pgd(agent, env)

    print("4/4: Gradient heatmap")
    plot_perturbation_heatmap(agent, env)

    env.close()
    print("\nDeep analysis complete.")


if __name__ == "__main__":
    main()
