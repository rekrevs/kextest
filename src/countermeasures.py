"""Countermeasures against adversarial attacks on MARL.

1. Adversarial Training: retrain with random perturbations during training
2. Observation Smoothing: temporal exponential moving average of observations
3. Anomaly Detection: detect adversarial agents by monitoring action divergence
"""

import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Categorical


class ObservationSmoother:
    """Temporal exponential moving average of observations.

    Smooths out sudden perturbations by maintaining a running average.
    """

    def __init__(self, n_agents, obs_dim, alpha=0.3):
        """alpha: smoothing factor. Higher = more weight on current obs."""
        self.alpha = alpha
        self.ema = None

    def smooth(self, obs):
        """obs: (n_agents, obs_dim) tensor"""
        if self.ema is None:
            self.ema = obs.clone()
            return obs.clone()

        self.ema = self.alpha * obs + (1 - self.alpha) * self.ema
        return self.ema.clone()

    def reset(self):
        self.ema = None


class ActionDivergenceDetector:
    """Detect adversarial agents by monitoring KL divergence from expected behavior.

    Maintains a reference policy (the cooperative policy) and computes
    KL divergence of each agent's actions against the expected distribution.
    Flags agents whose behavior deviates too much.
    """

    def __init__(self, reference_actor, threshold=2.0, window_size=10):
        self.reference_actor = reference_actor
        self.threshold = threshold
        self.window_size = window_size
        self.kl_history = {}  # agent_idx -> list of KL values

    @torch.no_grad()
    def check(self, obs, actions):
        """Check each agent for anomalous behavior.

        obs: (n_agents, obs_dim)
        actions: (n_agents,) taken actions

        Returns: dict of agent_idx -> (is_anomalous, kl_value)
        """
        ref_logits = self.reference_actor(obs)
        ref_dist = Categorical(logits=ref_logits)

        results = {}
        for i in range(obs.shape[0]):
            # KL between uniform-on-taken-action and reference
            # Simpler: use negative log prob of taken action under reference
            neg_log_prob = -ref_dist.log_prob(actions)[i].item()

            if i not in self.kl_history:
                self.kl_history[i] = []
            self.kl_history[i].append(neg_log_prob)

            # Keep window
            if len(self.kl_history[i]) > self.window_size:
                self.kl_history[i] = self.kl_history[i][-self.window_size:]

            avg_nll = np.mean(self.kl_history[i])
            is_anomalous = avg_nll > self.threshold
            results[i] = (is_anomalous, avg_nll)

        return results

    def reset(self):
        self.kl_history = {}


def noise_augmentation(obs, epsilon=0.1):
    """Add random uniform noise to observations during training.

    This is NOT adversarial training — it is noise augmentation.
    """
    noise = torch.empty_like(obs).uniform_(-epsilon, epsilon)
    return obs + noise


def fgsm_adversarial_training_step(actor, obs, epsilon=0.1):
    """True FGSM-based adversarial training.

    Computes the gradient of the policy loss w.r.t. observations and
    perturbs in the adversarial direction (decreasing best-action probability).
    """
    obs_adv = obs.clone().detach().requires_grad_(True)
    logits = actor(obs_adv)
    dist = Categorical(logits=logits)
    best_actions = logits.argmax(dim=-1).detach()
    loss = dist.log_prob(best_actions).sum()
    loss.backward()
    grad_sign = obs_adv.grad.sign()
    perturbed = obs.detach() - epsilon * grad_sign.detach()
    return perturbed
