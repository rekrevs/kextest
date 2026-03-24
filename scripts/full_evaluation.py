"""Full evaluation: all research angles combined.

1. Transferability: cross-algorithm attack testing (MAPPO <-> QMIX)
2. Communication robustness: MAPPO vs CommNet under attack
3. Stealth: compare stealth (KL-constrained) vs unconstrained adversary
4. Countermeasures: robust MAPPO and observation smoothing
5. Scaled environment: N=5 continuous results
"""

import sys
sys.path.insert(0, "/Users/sverker/repos/kextest")

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torch.distributions import Categorical

from src.env_wrapper import SimpleSpreadEnv
from src.mappo import MAPPO
from src.attacks import FGSMAttack, AdversarialAgent, RandomNoiseAttack
from src.networks import Actor
from src.countermeasures import ObservationSmoother, ActionDivergenceDetector


# ─── Helpers ─────────────────────────────────────────────────────────

def eval_clean(agent, env, n_ep=100, seed=0):
    rewards = []
    for ep in range(n_ep):
        obs = env.reset(seed=seed + ep)
        done, r = False, 0.0
        while not done:
            a, _, _ = agent.act(obs, deterministic=True)
            obs, rew, done, _ = env.step(a)
            if obs is None: break
            r += rew
        rewards.append(r)
    return np.array(rewards)


def eval_fgsm(agent, env, actor_for_grad, eps=0.2, n_ep=100, seed=0):
    attack = FGSMAttack(actor_for_grad, epsilon=eps)
    rewards = []
    for ep in range(n_ep):
        obs = env.reset(seed=seed + ep)
        done, r = False, 0.0
        while not done:
            obs_p = attack.perturb(obs)
            a, _, _ = agent.act(obs_p, deterministic=True)
            obs, rew, done, _ = env.step(a)
            if obs is None: break
            r += rew
        rewards.append(r)
    return np.array(rewards)


def eval_adversary(agent, env, adv_actor, adv_idx=0, n_ep=100, seed=0):
    adv = AdversarialAgent(adv_actor)
    rewards = []
    for ep in range(n_ep):
        obs = env.reset(seed=seed + ep)
        done, r = False, 0.0
        while not done:
            a, _, _ = agent.act(obs, deterministic=True)
            a[adv_idx] = adv.act(obs[adv_idx], deterministic=True)
            obs, rew, done, _ = env.step(a)
            if obs is None: break
            r += rew
        rewards.append(r)
    return np.array(rewards)


def eval_with_smoother(agent, env, attack, alpha=0.3, n_ep=100, seed=0):
    smoother = ObservationSmoother(env.n_agents, env.obs_dim, alpha=alpha)
    rewards = []
    for ep in range(n_ep):
        obs = env.reset(seed=seed + ep)
        smoother.reset()
        done, r = False, 0.0
        while not done:
            obs_attacked = attack.perturb(obs)
            obs_smoothed = smoother.smooth(obs_attacked)
            a, _, _ = agent.act(obs_smoothed, deterministic=True)
            obs, rew, done, _ = env.step(a)
            if obs is None: break
            r += rew
        rewards.append(r)
    return np.array(rewards)


def eval_detection(agent, env, adv_actor, adv_idx=0, n_ep=50, seed=0):
    """Evaluate anomaly detection against adversarial agent."""
    adv = AdversarialAgent(adv_actor)
    detector = ActionDivergenceDetector(agent.actor, threshold=2.0, window_size=5)

    true_pos, false_neg = 0, 0  # detecting adv agent
    true_neg, false_pos = 0, 0  # not flagging clean agents
    total_steps = 0

    for ep in range(n_ep):
        obs = env.reset(seed=seed + ep)
        detector.reset()
        done = False
        while not done:
            a, _, _ = agent.act(obs, deterministic=True)
            a[adv_idx] = adv.act(obs[adv_idx], deterministic=True)

            results = detector.check(obs, a)
            for idx, (is_anom, _) in results.items():
                if idx == adv_idx:
                    if is_anom: true_pos += 1
                    else: false_neg += 1
                else:
                    if is_anom: false_pos += 1
                    else: true_neg += 1

            obs, _, done, _ = env.step(a)
            if obs is None: break
            total_steps += 1

    tpr = true_pos / max(true_pos + false_neg, 1)
    fpr = false_pos / max(false_pos + true_neg, 1)
    return tpr, fpr


# ─── Main sections ───────────────────────────────────────────────────

def section_transferability(env, n_ep=100):
    """Test attack transferability between MAPPO and QMIX."""
    print("\n" + "=" * 70)
    print("SECTION 1: ATTACK TRANSFERABILITY (MAPPO <-> QMIX)")
    print("=" * 70)

    obs = env.reset(seed=0)

    mappo = MAPPO(obs_dim=env.obs_dim, global_state_dim=env.global_state_dim,
                  n_actions=env.n_actions, n_agents=3, hidden_dim=128)
    mappo.load("models/mappo_baseline/best")
    mappo.actor.eval()

    # QMIX uses a different actor architecture - load it
    qmix_path = Path("models/qmix_baseline/best")
    if not qmix_path.exists():
        print("  QMIX model not found, skipping transferability.")
        return {}

    from src.qmix import QMIX
    qmix = QMIX(obs_dim=env.obs_dim, global_state_dim=env.global_state_dim,
                 n_actions=env.n_actions, n_agents=3, hidden_dim=64)
    qmix.load("models/qmix_baseline/best")
    qmix.agent_net.eval()

    results = {}

    # Baselines
    mappo_clean = eval_clean(mappo, env, n_ep)
    print(f"  MAPPO clean:  {mappo_clean.mean():.2f}")
    results["MAPPO clean"] = mappo_clean.mean()

    # QMIX clean eval uses different act method
    qmix_rewards = []
    for ep in range(n_ep):
        obs = env.reset(seed=ep)
        hidden = qmix.agent_net.init_hidden(1).expand(3, -1).contiguous()
        done, r = False, 0.0
        while not done:
            a, hidden = qmix.act(obs, hidden, deterministic=True)
            obs, rew, done, _ = env.step(a)
            if obs is None: break
            r += rew
        qmix_rewards.append(r)
    qmix_clean = np.array(qmix_rewards)
    print(f"  QMIX clean:   {qmix_clean.mean():.2f}")
    results["QMIX clean"] = qmix_clean.mean()

    # FGSM trained on MAPPO, tested on MAPPO
    r = eval_fgsm(mappo, env, mappo.actor, eps=0.5, n_ep=n_ep)
    print(f"  FGSM(MAPPO)->MAPPO: {r.mean():.2f}")
    results["FGSM(MAPPO)->MAPPO"] = r.mean()

    # FGSM trained on MAPPO, tested on QMIX (transferability!)
    qmix_under_mappo_fgsm = []
    fgsm_mappo = FGSMAttack(mappo.actor, epsilon=0.5)
    for ep in range(n_ep):
        obs = env.reset(seed=ep)
        hidden = qmix.agent_net.init_hidden(1).expand(3, -1).contiguous()
        done, r = False, 0.0
        while not done:
            obs_p = fgsm_mappo.perturb(obs)
            a, hidden = qmix.act(obs_p, hidden, deterministic=True)
            obs, rew, done, _ = env.step(a)
            if obs is None: break
            r += rew
        qmix_under_mappo_fgsm.append(r)
    r = np.array(qmix_under_mappo_fgsm)
    print(f"  FGSM(MAPPO)->QMIX:  {r.mean():.2f} (transferability)")
    results["FGSM(MAPPO)->QMIX"] = r.mean()

    # Adversarial agent trained against MAPPO, tested on QMIX
    adv_path = "models/adversarial/best_adversary.pt"
    if Path(adv_path).exists():
        adv_actor = Actor(env.obs_dim, env.n_actions, hidden_dim=64)
        adv_actor.load_state_dict(torch.load(adv_path, weights_only=True))
        adv_actor.eval()

        r_mappo = eval_adversary(mappo, env, adv_actor, n_ep=n_ep)
        print(f"  AdvAgent(MAPPO)->MAPPO: {r_mappo.mean():.2f}")
        results["AdvAgent(MAPPO)->MAPPO"] = r_mappo.mean()

        # Test against QMIX
        qmix_under_adv = []
        adv = AdversarialAgent(adv_actor)
        for ep in range(n_ep):
            obs = env.reset(seed=ep)
            hidden = qmix.agent_net.init_hidden(1).expand(3, -1).contiguous()
            done, r = False, 0.0
            while not done:
                a, hidden = qmix.act(obs, hidden, deterministic=True)
                a[0] = adv.act(obs[0], deterministic=True)
                obs, rew, done, _ = env.step(a)
                if obs is None: break
                r += rew
            qmix_under_adv.append(r)
        r = np.array(qmix_under_adv)
        print(f"  AdvAgent(MAPPO)->QMIX:  {r.mean():.2f} (transferability)")
        results["AdvAgent(MAPPO)->QMIX"] = r.mean()

    return results


def section_communication(env, n_ep=100):
    """Compare MAPPO vs CommNet robustness."""
    print("\n" + "=" * 70)
    print("SECTION 2: COMMUNICATION ROBUSTNESS (MAPPO vs CommNet)")
    print("=" * 70)

    obs = env.reset(seed=0)

    mappo = MAPPO(obs_dim=env.obs_dim, global_state_dim=env.global_state_dim,
                  n_actions=env.n_actions, n_agents=3, hidden_dim=128)
    mappo.load("models/mappo_baseline/best")
    mappo.actor.eval()

    comm_path = Path("models/commnet_baseline/best")
    if not comm_path.exists():
        print("  CommNet model not found, skipping communication analysis.")
        return {}

    from src.commnet import CommMAPPO
    commnet = CommMAPPO(obs_dim=env.obs_dim, global_state_dim=env.global_state_dim,
                        n_actions=env.n_actions, n_agents=3, hidden_dim=128, n_comm_rounds=2)
    commnet.load("models/commnet_baseline/best")
    commnet.actor.eval()

    results = {}

    # Clean
    mappo_c = eval_clean(mappo, env, n_ep)
    comm_c = eval_clean(commnet, env, n_ep)
    print(f"  MAPPO clean:   {mappo_c.mean():.2f}")
    print(f"  CommNet clean: {comm_c.mean():.2f}")
    results["MAPPO clean"] = mappo_c.mean()
    results["CommNet clean"] = comm_c.mean()

    # FGSM attack on both
    for eps in [0.1, 0.2, 0.5, 1.0]:
        fgsm_m = FGSMAttack(mappo.actor, epsilon=eps)
        r_m = eval_fgsm(mappo, env, mappo.actor, eps=eps, n_ep=n_ep)

        # For CommNet, gradient goes through commnet actor
        fgsm_c = FGSMAttack(commnet.actor, epsilon=eps)
        # We can't use eval_fgsm directly since CommNet actor expects grouped input
        # Use the flat forward for gradient, but evaluate with full comm
        comm_rewards = []
        for ep in range(n_ep):
            obs = env.reset(seed=ep)
            done, r = False, 0.0
            while not done:
                obs_p = fgsm_c.perturb(obs)
                a, _, _ = commnet.act(obs_p, deterministic=True)
                obs, rew, done, _ = env.step(a)
                if obs is None: break
                r += rew
            comm_rewards.append(r)
        r_c = np.array(comm_rewards)

        pct_m = (r_m.mean() - mappo_c.mean()) / abs(mappo_c.mean()) * 100
        pct_c = (r_c.mean() - comm_c.mean()) / abs(comm_c.mean()) * 100
        print(f"  FGSM eps={eps:.1f}: MAPPO={r_m.mean():.2f} ({pct_m:+.1f}%), "
              f"CommNet={r_c.mean():.2f} ({pct_c:+.1f}%)")
        results[f"MAPPO FGSM eps={eps}"] = r_m.mean()
        results[f"CommNet FGSM eps={eps}"] = r_c.mean()

    return results


def section_stealth(env, n_ep=100):
    """Compare stealth vs unconstrained adversary."""
    print("\n" + "=" * 70)
    print("SECTION 3: STEALTH ATTACKS (KL-CONSTRAINED)")
    print("=" * 70)

    obs = env.reset(seed=0)

    mappo = MAPPO(obs_dim=env.obs_dim, global_state_dim=env.global_state_dim,
                  n_actions=env.n_actions, n_agents=3, hidden_dim=128)
    mappo.load("models/mappo_baseline/best")
    mappo.actor.eval()

    results = {}

    # Clean
    clean = eval_clean(mappo, env, n_ep)
    print(f"  Clean: {clean.mean():.2f}")
    results["Clean"] = clean.mean()

    # Unconstrained adversary
    adv_path = "models/adversarial/best_adversary.pt"
    if Path(adv_path).exists():
        adv_actor = Actor(env.obs_dim, env.n_actions, hidden_dim=64)
        adv_actor.load_state_dict(torch.load(adv_path, weights_only=True))
        adv_actor.eval()
        r = eval_adversary(mappo, env, adv_actor, n_ep=n_ep)
        print(f"  Unconstrained adversary: {r.mean():.2f}")
        results["Unconstrained"] = r.mean()

        # Detection rate
        tpr, fpr = eval_detection(mappo, env, adv_actor, n_ep=50)
        print(f"    Detection: TPR={tpr:.2%}, FPR={fpr:.2%}")
        results["Unconstrained TPR"] = tpr
        results["Unconstrained FPR"] = fpr

    # Stealth adversaries with different KL thresholds
    for kl in [0.1, 0.5, 2.0]:
        stealth_path = f"models/stealth_kl{kl}/best_stealth.pt"
        if Path(stealth_path).exists():
            s_actor = Actor(env.obs_dim, env.n_actions, hidden_dim=64)
            s_actor.load_state_dict(torch.load(stealth_path, weights_only=True))
            s_actor.eval()
            r = eval_adversary(mappo, env, s_actor, n_ep=n_ep)

            tpr, fpr = eval_detection(mappo, env, s_actor, n_ep=50)

            print(f"  Stealth (KL<={kl}): reward={r.mean():.2f}, TPR={tpr:.2%}, FPR={fpr:.2%}")
            results[f"Stealth KL={kl}"] = r.mean()
            results[f"Stealth KL={kl} TPR"] = tpr
        else:
            print(f"  Stealth (KL<={kl}): model not found")

    return results


def section_countermeasures(env, n_ep=100):
    """Evaluate countermeasures: robust MAPPO + observation smoothing."""
    print("\n" + "=" * 70)
    print("SECTION 4: COUNTERMEASURES")
    print("=" * 70)

    obs = env.reset(seed=0)

    mappo = MAPPO(obs_dim=env.obs_dim, global_state_dim=env.global_state_dim,
                  n_actions=env.n_actions, n_agents=3, hidden_dim=128)
    mappo.load("models/mappo_baseline/best")
    mappo.actor.eval()

    results = {}

    # Standard MAPPO under attack
    clean = eval_clean(mappo, env, n_ep)
    fgsm_std = eval_fgsm(mappo, env, mappo.actor, eps=0.5, n_ep=n_ep)
    print(f"  Standard MAPPO - clean: {clean.mean():.2f}, FGSM(0.5): {fgsm_std.mean():.2f}")
    results["Std clean"] = clean.mean()
    results["Std FGSM"] = fgsm_std.mean()

    # Robust MAPPO
    robust_path = Path("models/mappo_robust/best")
    if robust_path.exists():
        robust = MAPPO(obs_dim=env.obs_dim, global_state_dim=env.global_state_dim,
                       n_actions=env.n_actions, n_agents=3, hidden_dim=128)
        robust.load("models/mappo_robust/best")
        robust.actor.eval()

        robust_clean = eval_clean(robust, env, n_ep)
        robust_fgsm = eval_fgsm(robust, env, robust.actor, eps=0.5, n_ep=n_ep)
        print(f"  Robust MAPPO  - clean: {robust_clean.mean():.2f}, FGSM(0.5): {robust_fgsm.mean():.2f}")
        results["Robust clean"] = robust_clean.mean()
        results["Robust FGSM"] = robust_fgsm.mean()

        # Also test with FGSM grad from standard model (black-box transfer)
        robust_fgsm_transfer = eval_fgsm(robust, env, mappo.actor, eps=0.5, n_ep=n_ep)
        print(f"  Robust MAPPO  - FGSM(transfer): {robust_fgsm_transfer.mean():.2f}")
        results["Robust FGSM transfer"] = robust_fgsm_transfer.mean()
    else:
        print("  Robust MAPPO model not found.")

    # Observation smoothing
    print("\n  Observation smoothing defense:")
    for alpha in [0.2, 0.5, 0.8]:
        fgsm_attack = FGSMAttack(mappo.actor, epsilon=0.5)

        # Clean + smoothing (cost of defense)
        smoother = ObservationSmoother(env.n_agents, env.obs_dim, alpha=alpha)
        smooth_clean = []
        for ep in range(n_ep):
            obs = env.reset(seed=ep)
            smoother.reset()
            done, r = False, 0.0
            while not done:
                obs_s = smoother.smooth(obs)
                a, _, _ = mappo.act(obs_s, deterministic=True)
                obs, rew, done, _ = env.step(a)
                if obs is None: break
                r += rew
            smooth_clean.append(r)

        # FGSM + smoothing
        smooth_fgsm = eval_with_smoother(mappo, env, fgsm_attack, alpha=alpha, n_ep=n_ep)

        print(f"    alpha={alpha}: clean={np.mean(smooth_clean):.2f}, FGSM(0.5)={smooth_fgsm.mean():.2f}")
        results[f"Smooth a={alpha} clean"] = np.mean(smooth_clean)
        results[f"Smooth a={alpha} FGSM"] = smooth_fgsm.mean()

    return results


def generate_summary_plots(all_results, save_dir="results"):
    """Generate publication-quality summary plots."""
    Path(save_dir).mkdir(exist_ok=True)

    # ─── Plot 1: Transferability bar chart ─────────────
    if "transferability" in all_results and all_results["transferability"]:
        r = all_results["transferability"]
        fig, ax = plt.subplots(figsize=(12, 6))

        categories = list(r.keys())
        values = list(r.values())
        colors = ["green" if "clean" in k.lower() else
                  "coral" if "AdvAgent" in k else "steelblue"
                  for k in categories]

        ax.bar(categories, values, color=colors, alpha=0.8)
        ax.set_ylabel("Mean Team Reward", fontsize=12)
        ax.set_title("Attack Transferability: MAPPO vs QMIX", fontsize=14)
        plt.xticks(rotation=30, ha="right", fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")
        fig.savefig(f"{save_dir}/transferability.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {save_dir}/transferability.png")

    # ─── Plot 2: Communication robustness ─────────────
    if "communication" in all_results and all_results["communication"]:
        r = all_results["communication"]
        epsilons = [0.1, 0.2, 0.5, 1.0]

        mappo_vals = [r.get("MAPPO clean", 0)] + [r.get(f"MAPPO FGSM eps={e}", 0) for e in epsilons]
        comm_vals = [r.get("CommNet clean", 0)] + [r.get(f"CommNet FGSM eps={e}", 0) for e in epsilons]

        fig, ax = plt.subplots(figsize=(10, 6))
        x = [0] + epsilons
        ax.plot(x, mappo_vals, "o-", label="MAPPO (no comm)", linewidth=2, markersize=8)
        ax.plot(x, comm_vals, "s-", label="CommNet", linewidth=2, markersize=8)
        ax.set_xlabel("FGSM epsilon", fontsize=12)
        ax.set_ylabel("Mean Team Reward", fontsize=12)
        ax.set_title("Communication Robustness: MAPPO vs CommNet Under Attack", fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        fig.savefig(f"{save_dir}/comm_robustness.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {save_dir}/comm_robustness.png")

    # ─── Plot 3: Stealth trade-off ─────────────
    if "stealth" in all_results and all_results["stealth"]:
        r = all_results["stealth"]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        labels = ["Clean"]
        rewards = [r.get("Clean", 0)]
        tprs = [0]

        if "Unconstrained" in r:
            labels.append("Unconstrained")
            rewards.append(r["Unconstrained"])
            tprs.append(r.get("Unconstrained TPR", 0))

        for kl in [2.0, 0.5, 0.1]:
            key = f"Stealth KL={kl}"
            if key in r:
                labels.append(f"KL≤{kl}")
                rewards.append(r[key])
                tprs.append(r.get(f"Stealth KL={kl} TPR", 0))

        colors_r = ["green"] + ["coral"] * (len(labels) - 1)
        ax1.bar(labels, rewards, color=colors_r, alpha=0.8)
        ax1.set_ylabel("Mean Team Reward", fontsize=12)
        ax1.set_title("Attack Effectiveness vs Stealth", fontsize=13)
        ax1.grid(True, alpha=0.3, axis="y")

        colors_t = ["green"] + ["steelblue"] * (len(labels) - 1)
        ax2.bar(labels, tprs, color=colors_t, alpha=0.8)
        ax2.set_ylabel("True Positive Rate (Detection)", fontsize=12)
        ax2.set_title("Detectability of Adversarial Agent", fontsize=13)
        ax2.set_ylim(0, 1.05)
        ax2.grid(True, alpha=0.3, axis="y")

        fig.suptitle("Stealth-Effectiveness Trade-off", fontsize=15)
        plt.tight_layout()
        fig.savefig(f"{save_dir}/stealth_tradeoff.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {save_dir}/stealth_tradeoff.png")

    # ─── Plot 4: Countermeasures ─────────────
    if "countermeasures" in all_results and all_results["countermeasures"]:
        r = all_results["countermeasures"]
        fig, ax = plt.subplots(figsize=(12, 6))

        groups = []
        clean_vals = []
        fgsm_vals = []

        groups.append("Standard MAPPO")
        clean_vals.append(r.get("Std clean", 0))
        fgsm_vals.append(r.get("Std FGSM", 0))

        if "Robust clean" in r:
            groups.append("Robust MAPPO")
            clean_vals.append(r["Robust clean"])
            fgsm_vals.append(r.get("Robust FGSM", 0))

        for alpha in [0.2, 0.5, 0.8]:
            k = f"Smooth a={alpha}"
            if f"{k} clean" in r:
                groups.append(f"Smoothing α={alpha}")
                clean_vals.append(r[f"{k} clean"])
                fgsm_vals.append(r[f"{k} FGSM"])

        x = np.arange(len(groups))
        w = 0.35
        ax.bar(x - w / 2, clean_vals, w, label="Clean", color="green", alpha=0.8)
        ax.bar(x + w / 2, fgsm_vals, w, label="Under FGSM (eps=0.5)", color="coral", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(groups, rotation=15, ha="right")
        ax.set_ylabel("Mean Team Reward", fontsize=12)
        ax.set_title("Countermeasure Effectiveness", fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis="y")
        fig.savefig(f"{save_dir}/countermeasures.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {save_dir}/countermeasures.png")


def main():
    env = SimpleSpreadEnv(n_agents=3, max_cycles=25)

    all_results = {}

    all_results["transferability"] = section_transferability(env, n_ep=100)
    all_results["communication"] = section_communication(env, n_ep=100)
    all_results["stealth"] = section_stealth(env, n_ep=100)
    all_results["countermeasures"] = section_countermeasures(env, n_ep=100)

    print("\n" + "=" * 70)
    print("GENERATING SUMMARY PLOTS")
    print("=" * 70)
    generate_summary_plots(all_results)

    env.close()
    print("\n" + "=" * 70)
    print("FULL EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
