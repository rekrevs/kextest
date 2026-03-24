"""Train MAPPO with multiple seeds for statistical rigor."""

import sys
sys.path.insert(0, "/Users/sverker/repos/kextest")

import torch
import numpy as np
from src.env_wrapper import SimpleSpreadEnv
from src.mappo import MAPPO
from src.buffer import RolloutBuffer


def train_single(seed, model_dir, n_episodes=20000, episodes_per_batch=10,
                 hidden_dim=128, lr=5e-4, adv_training=False, adv_epsilon=0.1):
    """Train one MAPPO instance. Returns final eval rewards."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = SimpleSpreadEnv(n_agents=3, max_cycles=25)
    obs = env.reset(seed=seed)

    agent = MAPPO(
        obs_dim=env.obs_dim,
        global_state_dim=env.global_state_dim,
        n_actions=env.n_actions,
        n_agents=3,
        hidden_dim=hidden_dim,
        lr_actor=lr, lr_critic=lr,
        ppo_epochs=15,
    )

    if adv_training:
        from src.countermeasures import fgsm_adversarial_training_step

    ep_count = 0
    while ep_count < n_episodes:
        buffer = RolloutBuffer()
        for _ in range(episodes_per_batch):
            obs = env.reset(seed=seed * 100000 + ep_count)
            done = False
            while not done:
                if adv_training and np.random.random() < 0.5:
                    obs_for_action = fgsm_adversarial_training_step(
                        agent.actor, obs, epsilon=adv_epsilon
                    )
                else:
                    obs_for_action = obs
                actions, log_probs, value = agent.act(obs_for_action)
                next_obs, reward, done, info = env.step(actions)
                if next_obs is None:
                    buffer.store(obs, actions, log_probs, reward if reward else 0.0, value, True)
                    break
                buffer.store(obs, actions, log_probs, reward, value, done)
                obs = next_obs
            ep_count += 1

        last_value = 0.0 if done else agent.act(obs)[2]
        if len(buffer) > 0:
            agent.update(buffer, last_value)

    agent.save(f"{model_dir}/seed_{seed}")
    env.close()
    return agent


def main():
    seeds = [42, 123, 456, 789, 1024]

    print("=" * 60)
    print("MULTI-SEED MAPPO TRAINING")
    print("=" * 60)

    for seed in seeds:
        print(f"\n--- Training MAPPO seed={seed} ---")
        train_single(seed, "models/mappo_multiseed", n_episodes=20000)
        print(f"  Done: seed={seed}")

    print("\n--- Training FGSM-Adversarial MAPPO (5 seeds) ---")
    for seed in seeds:
        print(f"\n--- Training Robust MAPPO seed={seed} ---")
        train_single(seed, "models/mappo_robust_fgsm", n_episodes=20000,
                     adv_training=True, adv_epsilon=0.2)
        print(f"  Done: seed={seed}")

    print("\nAll multi-seed training complete.")


if __name__ == "__main__":
    main()
