"""Complete evaluation: all results across 5 MAPPO seeds with CIs.

Fills the gaps: adversarial agent, stealth, communication, transferability
all evaluated against the 5 trained MAPPO seeds.
"""

import sys
sys.path.insert(0, "/Users/sverker/repos/kextest")

import torch
import numpy as np
from pathlib import Path
from scipy import stats
from torch.distributions import Categorical

from src.env_wrapper import SimpleSpreadEnv
from src.mappo import MAPPO
from src.attacks import FGSMAttack, AdversarialAgent
from src.networks import Actor
from src.commnet import CommMAPPO
from src.countermeasures import ObservationSmoother, noise_augmentation


def load_mappo(path, env):
    a = MAPPO(obs_dim=env.obs_dim, global_state_dim=env.global_state_dim,
              n_actions=env.n_actions, n_agents=3, hidden_dim=128)
    a.load(path)
    a.actor.eval(); a.critic.eval()
    return a


def ci95(vals):
    m = np.mean(vals)
    if len(vals) < 2:
        return m, m, m
    lo, hi = stats.t.interval(0.95, df=len(vals)-1, loc=m, scale=stats.sem(vals))
    return m, lo, hi


def fmt(m, lo, hi):
    return f"{m:.1f} [{lo:.1f}, {hi:.1f}]"


def eval_ep(agent, env, n_ep=100, seed=50000, attack=None, adv_agent=None, adv_idx=0,
            smoother_alpha=None):
    rewards = []
    for ep in range(n_ep):
        obs = env.reset(seed=seed+ep)
        sm = ObservationSmoother(3, env.obs_dim, smoother_alpha) if smoother_alpha else None
        done, r = False, 0.0
        while not done:
            o = obs
            if attack: o = attack.perturb(o)
            if sm: o = sm.smooth(o)
            a, _, _ = agent.act(o, deterministic=True)
            if adv_agent is not None:
                a[adv_idx] = adv_agent.act(obs[adv_idx], deterministic=True)
            obs, rew, done, _ = env.step(a)
            if obs is None: break
            r += rew
        rewards.append(r)
    return np.array(rewards)


def multiseed(env, label, eval_fn, model_dir="models/mappo_multiseed"):
    """Run eval_fn(agent, env) for each seed, return per-seed means."""
    seeds = sorted(Path(model_dir).glob("seed_*"))
    if not seeds:
        print(f"  {label}: NO MODELS")
        return None
    per_seed = []
    for sd in seeds:
        agent = load_mappo(str(sd), env)
        r = eval_fn(agent, env)
        per_seed.append(r.mean())
    arr = np.array(per_seed)
    m, lo, hi = ci95(arr)
    print(f"  {label}: {fmt(m, lo, hi)} (n={len(arr)})")
    return arr


# ─── Stealth KL measurement ──────────────────────────────────────

def measure_stealth_kl(env, stealth_path, coop_agent, n_ep=100, seed=50000):
    """Measure achieved KL(stealth || coop) over evaluation episodes."""
    s_actor = Actor(env.obs_dim, env.n_actions, hidden_dim=64)
    s_actor.load_state_dict(torch.load(stealth_path, weights_only=True))
    s_actor.eval()

    kl_values = []
    for ep in range(n_ep):
        obs = env.reset(seed=seed+ep)
        done = False
        while not done:
            with torch.no_grad():
                s_logits = s_actor(obs[0].unsqueeze(0)).squeeze(0)
                c_logits = coop_agent.actor(obs[0].unsqueeze(0)).squeeze(0)
                s_dist = Categorical(logits=s_logits)
                c_dist = Categorical(logits=c_logits)
                kl = torch.distributions.kl_divergence(s_dist, c_dist).item()
                kl_values.append(kl)
            a, _, _ = coop_agent.act(obs, deterministic=True)
            obs, _, done, _ = env.step(a)
            if obs is None: break
    return np.mean(kl_values), np.std(kl_values)


def main():
    env = SimpleSpreadEnv(n_agents=3, max_cycles=25)
    env.reset(seed=0)

    print("=" * 70)
    print("COMPLETE MULTI-SEED EVALUATION")
    print("=" * 70)

    # ─── 1. Baselines ────────────────────────────────────────────
    print("\n--- Baselines ---")
    clean = multiseed(env, "Clean", lambda a, e: eval_ep(a, e))
    fgsm05 = multiseed(env, "FGSM eps=0.5", lambda a, e: eval_ep(a, e, attack=FGSMAttack(a.actor, 0.5)))
    fgsm10 = multiseed(env, "FGSM eps=1.0", lambda a, e: eval_ep(a, e, attack=FGSMAttack(a.actor, 1.0)))

    # ─── 2. Adversarial agent (across 5 MAPPO seeds) ─────────────
    print("\n--- Adversarial Agent (single adv model, 5 MAPPO seeds) ---")
    adv_path = "models/adversarial/best_adversary.pt"
    if Path(adv_path).exists():
        adv_actor = Actor(env.obs_dim, env.n_actions, hidden_dim=64)
        adv_actor.load_state_dict(torch.load(adv_path, weights_only=True))
        adv_actor.eval()
        adv_agent = AdversarialAgent(adv_actor)

        multiseed(env, "Adversarial agent",
                  lambda a, e: eval_ep(a, e, adv_agent=adv_agent))

    # ─── 3. Stealth (across 5 MAPPO seeds + KL measurement) ──────
    print("\n--- Stealth Attacks (5 MAPPO seeds) ---")
    for delta in [0.1, 0.5, 2.0]:
        sp = f"models/stealth_kl{delta}/best_stealth.pt"
        if Path(sp).exists():
            s_actor = Actor(env.obs_dim, env.n_actions, hidden_dim=64)
            s_actor.load_state_dict(torch.load(sp, weights_only=True))
            s_actor.eval()
            s_adv = AdversarialAgent(s_actor)

            multiseed(env, f"Stealth delta={delta}",
                      lambda a, e, sa=s_adv: eval_ep(a, e, adv_agent=sa))

            # Measure achieved KL against first seed
            first_seed = sorted(Path("models/mappo_multiseed").glob("seed_*"))[0]
            coop = load_mappo(str(first_seed), env)
            kl_mean, kl_std = measure_stealth_kl(env, sp, coop)
            print(f"    Achieved KL: {kl_mean:.3f} +/- {kl_std:.3f} (target delta={delta})")

    # ─── 4. Communication robustness (across 5 MAPPO seeds) ──────
    print("\n--- Communication Robustness ---")
    comm_path = Path("models/commnet_baseline/best")
    if comm_path.exists():
        commnet = CommMAPPO(obs_dim=env.obs_dim, global_state_dim=env.global_state_dim,
                            n_actions=env.n_actions, n_agents=3, hidden_dim=128, n_comm_rounds=2)
        commnet.load(str(comm_path))
        commnet.actor.eval()

        # CommNet clean (single model, 100 eps)
        r = eval_ep(commnet, env)
        print(f"  CommNet clean: {r.mean():.1f} +/- {r.std():.1f} (single model)")

        for eps in [0.1, 0.2, 0.5, 1.0]:
            attack = FGSMAttack(commnet.actor, epsilon=eps)
            r = eval_ep(commnet, env, attack=attack)
            print(f"  CommNet FGSM eps={eps}: {r.mean():.1f} +/- {r.std():.1f} (single model)")

        # MAPPO under same attacks for comparison
        print("  (MAPPO comparison across 5 seeds:)")
        for eps in [0.1, 0.2, 0.5, 1.0]:
            multiseed(env, f"MAPPO FGSM eps={eps}",
                      lambda a, e, ep=eps: eval_ep(a, e, attack=FGSMAttack(a.actor, ep)))

    # ─── 5. Countermeasures unified ──────────────────────────────
    print("\n--- Countermeasures (all 5 seeds) ---")
    multiseed(env, "Standard clean", lambda a, e: eval_ep(a, e))
    multiseed(env, "Standard FGSM 0.5", lambda a, e: eval_ep(a, e, attack=FGSMAttack(a.actor, 0.5)))

    # Noise augmentation (old robust model, single seed)
    rob_path = Path("models/mappo_robust/best")
    if rob_path.exists():
        rob = load_mappo(str(rob_path), env)
        r_clean = eval_ep(rob, env)
        r_fgsm = eval_ep(rob, env, attack=FGSMAttack(rob.actor, 0.5))
        print(f"  Noise augmentation clean: {r_clean.mean():.1f} (single seed)")
        print(f"  Noise augmentation FGSM 0.5: {r_fgsm.mean():.1f} (single seed)")

    # FGSM adv training (5 seeds)
    if Path("models/mappo_robust_fgsm/seed_42").exists():
        multiseed(env, "FGSM adv.train clean",
                  lambda a, e: eval_ep(a, e),
                  model_dir="models/mappo_robust_fgsm")
        multiseed(env, "FGSM adv.train FGSM 0.5",
                  lambda a, e: eval_ep(a, e, attack=FGSMAttack(a.actor, 0.5)),
                  model_dir="models/mappo_robust_fgsm")

    # Smoothing (across 5 seeds)
    for alpha in [0.5, 0.8]:
        multiseed(env, f"Smooth a={alpha} clean",
                  lambda a, e, al=alpha: eval_ep(a, e, smoother_alpha=al))
        multiseed(env, f"Smooth a={alpha} FGSM 0.5",
                  lambda a, e, al=alpha: eval_ep(a, e, attack=FGSMAttack(a.actor, 0.5), smoother_alpha=al))
        # Adaptive
        eff = 0.5 / alpha
        multiseed(env, f"Smooth a={alpha} adaptive (eff_eps={eff:.2f})",
                  lambda a, e, al=alpha, ep=eff: eval_ep(a, e, attack=FGSMAttack(a.actor, ep), smoother_alpha=al))

    # ─── 6. Significance tests ───────────────────────────────────
    print("\n--- Key significance tests ---")
    if clean is not None and fgsm05 is not None:
        t, p = stats.ttest_rel(clean, fgsm05)
        print(f"  Clean vs FGSM(0.5): t={t:.3f}, p={p:.4f}")
    if clean is not None and fgsm10 is not None:
        t, p = stats.ttest_rel(clean, fgsm10)
        print(f"  Clean vs FGSM(1.0): t={t:.3f}, p={p:.4f}")

    env.close()
    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
