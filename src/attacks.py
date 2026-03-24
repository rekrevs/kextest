"""Adversarial attack implementations for c-MARL systems."""

import torch
import torch.nn as nn
from torch.distributions import Categorical


class FGSMAttack:
    """Fast Gradient Sign Method attack on agent observations.

    White-box attack: perturbs observations to maximize policy loss,
    causing agents to take suboptimal actions.
    """

    def __init__(self, actor, epsilon=0.1):
        self.actor = actor
        self.epsilon = epsilon

    def perturb(self, obs, target_agents=None):
        """Perturb observations using FGSM.

        Args:
            obs: (n_agents, obs_dim) tensor of observations
            target_agents: list of agent indices to attack. None = attack all.

        Returns:
            perturbed_obs: (n_agents, obs_dim) tensor
        """
        obs_adv = obs.clone().detach().requires_grad_(True)

        # Forward pass to get action logits
        logits = self.actor(obs_adv)
        dist = Categorical(logits=logits)

        # Get the best action under clean policy
        best_actions = logits.argmax(dim=-1).detach()

        # Loss: maximize probability of best action -> minimize negative log prob
        # To attack: we want to MINIMIZE the probability of the best action
        # So we compute log_prob of best action and do gradient ASCENT on -log_prob
        log_prob_best = dist.log_prob(best_actions)
        loss = log_prob_best.sum()  # We want to decrease this

        loss.backward()

        # FGSM: perturb in direction that DECREASES log prob of best action
        grad_sign = obs_adv.grad.sign()
        perturbation = -self.epsilon * grad_sign  # negative because we want to decrease

        perturbed = obs.clone().detach()
        if target_agents is None:
            perturbed = perturbed + perturbation.detach()
        else:
            for idx in target_agents:
                perturbed[idx] = perturbed[idx] + perturbation[idx].detach()

        return perturbed


class IterativeFGSMAttack:
    """Iterative FGSM (PGD-like) attack - stronger than single-step FGSM.

    Applies FGSM iteratively with smaller step size, projecting back
    into the epsilon-ball after each step.
    """

    def __init__(self, actor, epsilon=0.1, n_steps=10, step_size=None, random_start=True):
        self.actor = actor
        self.epsilon = epsilon
        self.n_steps = n_steps
        self.step_size = step_size or (epsilon / n_steps * 2)
        self.random_start = random_start

    def perturb(self, obs, target_agents=None):
        """Iterative FGSM (PGD) perturbation with optional random start."""
        original = obs.clone().detach()
        if self.random_start:
            perturbed = original + torch.empty_like(original).uniform_(-self.epsilon, self.epsilon)
        else:
            perturbed = original.clone()

        for _ in range(self.n_steps):
            perturbed = perturbed.requires_grad_(True)

            logits = self.actor(perturbed)
            dist = Categorical(logits=logits)
            best_actions = logits.argmax(dim=-1).detach()
            log_prob_best = dist.log_prob(best_actions)
            loss = log_prob_best.sum()
            loss.backward()

            grad_sign = perturbed.grad.sign()
            step = -self.step_size * grad_sign  # decrease log prob

            perturbed = perturbed.detach() + step

            # Apply only to target agents
            if target_agents is not None:
                mask = torch.zeros_like(perturbed, dtype=torch.bool)
                for idx in target_agents:
                    mask[idx] = True
                perturbed = torch.where(mask, perturbed, original)

            # Project back into epsilon ball
            delta = perturbed - original
            delta = torch.clamp(delta, -self.epsilon, self.epsilon)
            perturbed = original + delta

        return perturbed.detach()


class RandomNoiseAttack:
    """Baseline: random uniform noise added to observations."""

    def __init__(self, epsilon=0.1):
        self.epsilon = epsilon

    def perturb(self, obs, target_agents=None):
        noise = torch.empty_like(obs).uniform_(-self.epsilon, self.epsilon)
        perturbed = obs.clone()
        if target_agents is None:
            perturbed = perturbed + noise
        else:
            for idx in target_agents:
                perturbed[idx] = perturbed[idx] + noise[idx]
        return perturbed


class AdversarialAgent:
    """A malicious agent that tries to minimize team reward.

    Trained with negated team reward to learn adversarial behavior.
    Can replace one of the cooperative agents at inference time.
    """

    def __init__(self, actor):
        """actor: a trained Actor network (trained with -reward)."""
        self.actor = actor

    @torch.no_grad()
    def act(self, obs_single, deterministic=False):
        """Select action for single agent.

        Args:
            obs_single: (obs_dim,) observation for this agent

        Returns:
            action: scalar tensor
        """
        logits = self.actor(obs_single.unsqueeze(0)).squeeze(0)
        if deterministic:
            return logits.argmax()
        dist = Categorical(logits=logits)
        return dist.sample()
