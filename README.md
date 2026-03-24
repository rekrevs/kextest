# Adversarial Attacks on Cooperative Multi-Agent Reinforcement Learning

**Bachelor Thesis** | KTH Royal Institute of Technology | DD143X, 15 ECTS

Supervisors: Axel Andersson, György Dán

---

## Overview

How robust are cooperative AI teams when one member turns hostile — or when their sensors are compromised?

This project systematically investigates adversarial vulnerabilities in cooperative multi-agent reinforcement learning (c-MARL). We train teams of agents to solve a cooperative navigation task, then attack them in two fundamentally different ways: perturbing what they *see* (observation attacks) and replacing *who they are* (behavioral attacks).

### Key findings

| Finding | Details |
|---------|---------|
| **Behavioral attacks dominate** | A single adversarial agent degrades team performance by 42%, vs 25% for the strongest observation perturbation — but the two attacks assume very different threat models |
| **FGSM > PGD in discrete action spaces** | Single-step attacks cross more decision boundaries than iterative ones (robust across 3–50 step ablation) |
| **Communication amplifies adversarial noise** | CommNet is *more* vulnerable at low perturbation levels, as corrupted messages propagate to all agents |
| **Stealth-effectiveness trade-off** | KL-constrained adversaries can evade detection (TPR 37%) while still causing 10% damage |
| **No tested defense fully mitigates FGSM** | Adversarial training destabilizes RL; observation smoothing fails against adaptive attackers |

## Architecture

```
src/
├── env_wrapper.py          # PettingZoo Simple Spread (discrete)
├── env_continuous.py        # Continuous-action variant
├── networks.py              # Actor-Critic networks
├── mappo.py                 # Multi-Agent PPO (CTDE)
├── qmix.py                 # QMIX with hypernetwork mixer
├── commnet.py               # CommNet + MAPPO integration
├── buffer.py                # GAE rollout buffer
├── attacks.py               # FGSM, PGD, random noise, adversarial agent
├── stealth_attack.py        # KL-constrained stealth adversary
├── countermeasures.py       # Smoothing, anomaly detection, adversarial training
└── mappo_continuous.py      # Gaussian-policy MAPPO

scripts/
├── train_mappo.py           # Baseline cooperative training
├── train_qmix.py            # QMIX baseline
├── train_commnet.py         # CommNet baseline
├── train_adversarial.py     # Unconstrained adversarial agent
├── train_stealth.py         # Stealth adversaries (δ ∈ {0.1, 0.5, 2.0})
├── train_robust_mappo.py    # Noise augmentation defense
├── train_multiseed.py       # 5-seed training (standard + FGSM adv. training)
├── train_continuous.py      # N=5 continuous-action training
├── evaluate_attacks.py      # Single-seed attack evaluation
├── eval_multiseed.py        # Multi-seed eval with CIs + significance tests
├── deep_analysis.py         # Trajectories, gradients, FGSM vs PGD
├── full_evaluation.py       # All research sections combined
├── visualize_results.py     # Plot generation
└── run_random.py            # Random agent baseline

report/
├── main.tex                 # Full thesis (LaTeX)
├── main.pdf                 # Compiled PDF
├── references.bib           # 20 verified references
└── figures/                 # 11 publication-quality plots
```

## Experimental design

**Environment:** Simple Spread (MPE/PettingZoo) — 3 agents cooperate to cover 3 landmarks.

**Algorithms compared:**
- **MAPPO** — policy gradient with centralized critic, shared parameters
- **QMIX** — value decomposition with monotonic mixing network
- **CommNet** — MAPPO with learned inter-agent communication (2 rounds)

**Attacks:**
- Observation perturbation: FGSM, PGD (with random start + ablation), random noise
- Behavioral: adversarial agent trained with negated team reward
- Stealth: KL-divergence-constrained adversary (Lagrangian relaxation)

**Defenses:**
- FGSM adversarial training, noise augmentation, temporal observation smoothing
- Anomaly detection (action divergence monitoring)
- Adaptive attack evaluation (attacker compensates for defense)

**Statistical rigor:** 5 independent training seeds, 95% confidence intervals, paired t-tests.

## Quick start

```bash
# Install dependencies
pip install pettingzoo[mpe] mpe2 torch matplotlib tensorboard

# Train baseline MAPPO
python scripts/train_mappo.py

# Evaluate attacks
python scripts/evaluate_attacks.py

# Full multi-seed pipeline
python scripts/train_multiseed.py
python scripts/eval_multiseed.py

# Deep analysis (trajectories, gradient heatmaps, FGSM vs PGD)
python scripts/deep_analysis.py

# Compile thesis
cd report && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

## Selected results

**FGSM is effective; PGD is not (in discrete action spaces):**

| Attack | ε | Reward [95% CI] | p-value |
|--------|---|-----------------|---------|
| Clean | — | −65.5 [−69.9, −61.1] | — |
| FGSM | 0.5 | −75.0 [−79.6, −70.3] | 0.012* |
| FGSM | 1.0 | −82.2 [−88.0, −76.4] | 0.005** |
| PGD | 0.5 | −67.7 [−71.1, −64.4] | n.s. |
| PGD | 1.0 | −68.7 [−71.6, −65.9] | n.s. |

**Stealth trade-off:**

| Agent | Reward | Δ vs Clean | Detection (TPR) |
|-------|--------|------------|-----------------|
| Clean | −65.5 | — | 0% |
| Unconstrained | −97.2 | −42% | 98% |
| Stealth (δ=2.0) | −74.9 | −10% | 38% |
| Stealth (δ≤0.5) | −69.7 | −2% | 27% |

## References

The thesis builds on:

- Lin et al. (2020) — *On the Robustness of Cooperative Multi-Agent Reinforcement Learning*
- Pinto et al. (2017) — *Robust Adversarial Reinforcement Learning*
- Li et al. (2025) — *Attacking Cooperative MARL by Adversarial Minority Influence*
- Pham et al. (2022) — *c-MBA: Adversarial Attack for Cooperative MARL*
- Standen et al. (2025) — *Adversarial ML Attacks and Defences in MARL* (ACM Computing Surveys)

Full bibliography with 20 references in [`report/references.bib`](report/references.bib).

## License

This project is part of a KTH bachelor thesis. Code is available for academic use.
