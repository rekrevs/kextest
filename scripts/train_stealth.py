"""Train stealth adversarial agent with KL constraint."""

import sys
sys.path.insert(0, "/Users/sverker/repos/kextest")

import torch
import numpy as np
from src.env_wrapper import SimpleSpreadEnv
from src.mappo import MAPPO
from src.stealth_attack import StealthAdversarialTrainer


def train(
    cooperative_model_path="models/mappo_baseline/best",
    adv_agent_idx=0,
    n_agents=3,
    max_cycles=25,
    total_episodes=5000,
    episodes_per_batch=10,
    hidden_dim=64,
    lr=5e-4,
    kl_threshold=0.5,
    seed=123,
    log_interval=200,
    model_dir="models/stealth",
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = SimpleSpreadEnv(n_agents=n_agents, max_cycles=max_cycles)
    obs = env.reset(seed=seed)

    # Load cooperative agents
    coop = MAPPO(
        obs_dim=env.obs_dim,
        global_state_dim=env.global_state_dim,
        n_actions=env.n_actions,
        n_agents=n_agents,
        hidden_dim=128,
    )
    coop.load(cooperative_model_path)
    coop.actor.eval()

    # Stealth adversary
    adv = StealthAdversarialTrainer(
        obs_dim=env.obs_dim,
        n_actions=env.n_actions,
        cooperative_actor=coop.actor,
        hidden_dim=hidden_dim,
        lr=lr,
        kl_threshold=kl_threshold,
    )

    all_rewards = []
    best_min = float("inf")
    ep_count = 0

    while ep_count < total_episodes:
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
                with torch.no_grad():
                    coop_actions, _, _ = coop.act(obs, deterministic=True)

                adv_action, adv_log_prob = adv.act(obs[adv_agent_idx])
                coop_actions[adv_agent_idx] = adv_action

                next_obs, team_reward, done, info = env.step(coop_actions)

                batch_obs.append(obs[adv_agent_idx].clone())
                batch_actions.append(adv_action)
                batch_log_probs.append(adv_log_prob)
                batch_rewards.append(-team_reward)  # Negate
                batch_dones.append(done)

                if next_obs is None:
                    break
                obs = next_obs
                ep_team_reward += team_reward

            batch_team_rewards.append(ep_team_reward)
            ep_count += 1

        metrics = adv.update(batch_obs, batch_actions, batch_log_probs, batch_rewards, batch_dones)
        all_rewards.extend(batch_team_rewards)

        if ep_count % log_interval < episodes_per_batch:
            window = min(log_interval, len(all_rewards))
            recent = all_rewards[-window:]
            avg = np.mean(recent)
            print(f"Episode {ep_count:6d} | Team reward: {avg:.2f} | "
                  f"KL: {metrics['kl']:.4f} | Lambda: {metrics['lambda']:.4f}")

            if avg < best_min:
                best_min = avg
                adv.save(f"{model_dir}/best_stealth.pt")
                print(f"  -> New best stealth attack! ({avg:.2f})")

    adv.save(f"{model_dir}/final_stealth.pt")

    # Also train with different KL thresholds for comparison
    print(f"\nStealth training (kl_thresh={kl_threshold}) complete. Best: {best_min:.2f}")

    env.close()
    return all_rewards


if __name__ == "__main__":
    # Train with multiple KL thresholds
    for kl_thresh in [0.1, 0.5, 2.0]:
        print(f"\n{'='*60}")
        print(f"Training stealth adversary with KL threshold = {kl_thresh}")
        print(f"{'='*60}")
        train(
            kl_threshold=kl_thresh,
            model_dir=f"models/stealth_kl{kl_thresh}",
        )
