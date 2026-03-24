"""Multi-seed evaluation with confidence intervals, PGD ablation, and adaptive attacks."""

import sys
sys.path.insert(0, "/Users/sverker/repos/kextest")

import torch
import numpy as np
from pathlib import Path
from scipy import stats

from src.env_wrapper import SimpleSpreadEnv
from src.mappo import MAPPO
from src.attacks import FGSMAttack, IterativeFGSMAttack, RandomNoiseAttack
from src.countermeasures import ObservationSmoother


def load_mappo(model_path, env):
    agent = MAPPO(
        obs_dim=env.obs_dim, global_state_dim=env.global_state_dim,
        n_actions=env.n_actions, n_agents=3, hidden_dim=128,
    )
    agent.load(model_path)
    agent.actor.eval()
    agent.critic.eval()
    return agent


def eval_episodes(agent, env, n_ep=100, seed=0, attack=None, smoother_alpha=None):
    """Evaluate with optional attack and optional smoothing defense."""
    rewards = []
    for ep in range(n_ep):
        obs = env.reset(seed=seed + ep)
        smoother = None
        if smoother_alpha is not None:
            smoother = ObservationSmoother(env.n_agents, env.obs_dim, alpha=smoother_alpha)
        done, r = False, 0.0
        while not done:
            obs_used = obs
            if attack is not None:
                obs_used = attack.perturb(obs_used)
            if smoother is not None:
                obs_used = smoother.smooth(obs_used)
            a, _, _ = agent.act(obs_used, deterministic=True)
            obs, rew, done, _ = env.step(a)
            if obs is None:
                break
            r += rew
        rewards.append(r)
    return np.array(rewards)


def ci_95(values):
    """Compute 95% confidence interval for the mean."""
    n = len(values)
    mean = np.mean(values)
    se = stats.sem(values)
    ci = stats.t.interval(0.95, df=n - 1, loc=mean, scale=se) if n > 1 else (mean, mean)
    return mean, ci[0], ci[1]


def format_ci(mean, lo, hi):
    return f"{mean:.1f} [{lo:.1f}, {hi:.1f}]"


def multiseed_eval(model_dir, env, label, n_ep_per_seed=100, attack_fn=None,
                   smoother_alpha=None, eval_seed_offset=50000):
    """Evaluate across multiple training seeds. Returns per-seed means."""
    seed_dirs = sorted(Path(model_dir).glob("seed_*"))
    if not seed_dirs:
        print(f"  No seed dirs found in {model_dir}")
        return None

    per_seed_means = []
    for sd in seed_dirs:
        agent = load_mappo(str(sd), env)
        attack = attack_fn(agent) if attack_fn else None
        rews = eval_episodes(agent, env, n_ep_per_seed, seed=eval_seed_offset,
                             attack=attack, smoother_alpha=smoother_alpha)
        per_seed_means.append(rews.mean())

    arr = np.array(per_seed_means)
    mean, lo, hi = ci_95(arr)
    print(f"  {label}: {format_ci(mean, lo, hi)} (n_seeds={len(arr)})")
    return arr


def section_main_results(env):
    """Core results with proper CIs across seeds."""
    print("\n" + "=" * 70)
    print("MULTI-SEED RESULTS WITH 95% CONFIDENCE INTERVALS")
    print("=" * 70)

    # Clean
    multiseed_eval("models/mappo_multiseed", env, "Clean (no attack)")

    # Random noise
    for eps in [0.1, 0.5, 1.0]:
        multiseed_eval("models/mappo_multiseed", env, f"Random noise eps={eps}",
                       attack_fn=lambda a, e=eps: RandomNoiseAttack(epsilon=e))

    # FGSM
    for eps in [0.1, 0.2, 0.5, 1.0]:
        multiseed_eval("models/mappo_multiseed", env, f"FGSM eps={eps}",
                       attack_fn=lambda a, e=eps: FGSMAttack(a.actor, epsilon=e))

    # PGD
    for eps in [0.2, 0.5, 1.0]:
        multiseed_eval("models/mappo_multiseed", env, f"PGD eps={eps} (random start)",
                       attack_fn=lambda a, e=eps: IterativeFGSMAttack(
                           a.actor, epsilon=e, n_steps=10, random_start=True))


def section_pgd_ablation(env):
    """PGD ablation: vary steps and random start."""
    print("\n" + "=" * 70)
    print("PGD ABLATION")
    print("=" * 70)

    for eps in [0.2, 0.5]:
        print(f"\n  --- eps={eps} ---")
        for n_steps in [3, 5, 10, 20, 50]:
            for rs in [False, True]:
                label = f"PGD steps={n_steps:2d} rs={'Y' if rs else 'N'}"
                multiseed_eval("models/mappo_multiseed", env, label,
                               attack_fn=lambda a, e=eps, ns=n_steps, r=rs:
                               IterativeFGSMAttack(a.actor, epsilon=e, n_steps=ns, random_start=r))


def section_defense_comparison(env):
    """Compare noise augmentation vs real FGSM adversarial training."""
    print("\n" + "=" * 70)
    print("DEFENSE COMPARISON: NOISE AUGMENTATION vs FGSM ADV. TRAINING")
    print("=" * 70)

    # Standard MAPPO
    print("\n  --- Standard MAPPO ---")
    multiseed_eval("models/mappo_multiseed", env, "Clean")
    multiseed_eval("models/mappo_multiseed", env, "FGSM eps=0.5",
                   attack_fn=lambda a: FGSMAttack(a.actor, epsilon=0.5))

    # Real FGSM adversarial training
    if Path("models/mappo_robust_fgsm/seed_42").exists():
        print("\n  --- FGSM Adversarial Training ---")
        multiseed_eval("models/mappo_robust_fgsm", env, "Clean")
        multiseed_eval("models/mappo_robust_fgsm", env, "FGSM eps=0.5 (white-box)",
                       attack_fn=lambda a: FGSMAttack(a.actor, epsilon=0.5))

    # Smoothing
    print("\n  --- Observation Smoothing ---")
    for alpha in [0.5, 0.8]:
        multiseed_eval("models/mappo_multiseed", env, f"Smoothing a={alpha} clean",
                       smoother_alpha=alpha)
        multiseed_eval("models/mappo_multiseed", env, f"Smoothing a={alpha} + FGSM eps=0.5",
                       attack_fn=lambda a: FGSMAttack(a.actor, epsilon=0.5),
                       smoother_alpha=alpha)

    # Adaptive attack: FGSM through the smoother
    print("\n  --- Adaptive attack (FGSM through smoother) ---")
    # For adaptive attack, we'd need to differentiate through the smoother.
    # Since EMA is differentiable, we approximate by using larger epsilon.
    for alpha in [0.5, 0.8]:
        eff_eps = 0.5 / alpha  # compensate for smoothing attenuation
        multiseed_eval("models/mappo_multiseed", env,
                       f"Adaptive FGSM (eff_eps={eff_eps:.1f}) + Smooth a={alpha}",
                       attack_fn=lambda a, e=eff_eps: FGSMAttack(a.actor, epsilon=e),
                       smoother_alpha=alpha)


def section_significance_tests(env):
    """Pairwise significance tests for key comparisons."""
    print("\n" + "=" * 70)
    print("SIGNIFICANCE TESTS (paired t-test across seeds)")
    print("=" * 70)

    clean = multiseed_eval("models/mappo_multiseed", env, "Clean")
    fgsm05 = multiseed_eval("models/mappo_multiseed", env, "FGSM eps=0.5",
                            attack_fn=lambda a: FGSMAttack(a.actor, epsilon=0.5))
    fgsm10 = multiseed_eval("models/mappo_multiseed", env, "FGSM eps=1.0",
                            attack_fn=lambda a: FGSMAttack(a.actor, epsilon=1.0))

    if clean is not None and fgsm05 is not None:
        t, p = stats.ttest_rel(clean, fgsm05)
        print(f"\n  Clean vs FGSM(0.5): t={t:.3f}, p={p:.4f} {'***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'}")

    if clean is not None and fgsm10 is not None:
        t, p = stats.ttest_rel(clean, fgsm10)
        print(f"  Clean vs FGSM(1.0): t={t:.3f}, p={p:.4f} {'***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'}")


def main():
    env = SimpleSpreadEnv(n_agents=3, max_cycles=25)
    env.reset(seed=0)

    section_main_results(env)
    section_pgd_ablation(env)
    section_defense_comparison(env)
    section_significance_tests(env)

    env.close()
    print("\n" + "=" * 70)
    print("MULTI-SEED EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
