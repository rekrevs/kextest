"""Train QMIX on Simple Spread."""

import sys
sys.path.insert(0, "/Users/sverker/repos/kextest")

import torch
import numpy as np
from src.env_wrapper import SimpleSpreadEnv
from src.qmix import QMIX


def train(
    n_agents=3,
    max_cycles=25,
    total_episodes=20000,
    hidden_dim=64,
    lr=5e-4,
    gamma=0.99,
    batch_size=32,
    buffer_size=5000,
    target_update=200,
    seed=42,
    log_interval=200,
    save_interval=2000,
    model_dir="models/qmix_baseline",
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = SimpleSpreadEnv(n_agents=n_agents, max_cycles=max_cycles)
    obs = env.reset(seed=seed)

    agent = QMIX(
        obs_dim=env.obs_dim,
        global_state_dim=env.global_state_dim,
        n_actions=env.n_actions,
        n_agents=n_agents,
        hidden_dim=hidden_dim,
        lr=lr,
        gamma=gamma,
        batch_size=batch_size,
        buffer_size=buffer_size,
        target_update_interval=target_update,
    )

    all_rewards = []
    best_avg = -float("inf")

    for ep in range(1, total_episodes + 1):
        obs = env.reset(seed=seed + ep)
        hidden = agent.agent_net.init_hidden(1).expand(n_agents, -1).contiguous()
        done = False
        ep_reward = 0.0

        episode_data = {
            "obs": [], "actions": [], "rewards": [],
            "next_obs": [], "dones": [], "states": [], "next_states": [],
        }

        while not done:
            state = env.get_global_state(obs)
            actions, hidden = agent.act(obs, hidden)
            next_obs, reward, done, info = env.step(actions)

            if next_obs is None:
                next_obs_store = obs.clone()
                next_state = state.clone()
                actual_done = 1.0
            else:
                next_obs_store = next_obs.clone()
                next_state = env.get_global_state(next_obs)
                actual_done = float(done)

            episode_data["obs"].append(obs)
            episode_data["actions"].append(actions)
            episode_data["rewards"].append(reward if reward else 0.0)
            episode_data["next_obs"].append(next_obs_store)
            episode_data["dones"].append(actual_done)
            episode_data["states"].append(state)
            episode_data["next_states"].append(next_state)

            if next_obs is not None:
                obs = next_obs
            ep_reward += reward if reward else 0.0

        agent.replay_buffer.store_episode(episode_data)

        # Train
        loss = agent.update()

        all_rewards.append(ep_reward)

        if ep % log_interval == 0:
            recent = all_rewards[-log_interval:]
            avg = np.mean(recent)
            loss_str = f"{loss:.4f}" if loss is not None else "N/A"
            print(f"Episode {ep:6d} | Avg reward: {avg:.2f} | Loss: {loss_str} | Eps: {agent.get_epsilon():.3f}")

            if avg > best_avg:
                best_avg = avg
                agent.save(f"{model_dir}/best")
                print(f"  -> New best! ({avg:.2f})")

        if ep % save_interval == 0:
            agent.save(f"{model_dir}/checkpoint_{ep}")

    agent.save(f"{model_dir}/final")
    env.close()
    print(f"\nQMIX training complete. Best avg: {best_avg:.2f}")
    return all_rewards


if __name__ == "__main__":
    train()
