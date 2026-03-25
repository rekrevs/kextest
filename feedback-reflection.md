# Reflection on Reviewer Feedback

This document reflects on feedback received on the AI-generated KEX report, analyzing the validity of each criticism, what it reveals about the limitations of AI-generated research, and how the prompting process could have been designed to avoid these shortcomings.

## The Feedback

The reviewer noted that while the report "looks OK for a KEX project on the surface," it is "not very strong in terms of methodology," with three specific criticisms:

1. **Section 4.4.2 (FGSM Adversarial Training):** It is unclear whose observations are perturbed and how many victims there are. The approach does not look like real adversarial training.
2. **Section 4.4.4 (Anomaly Detection):** The report did not cite or use Kazari's IJCAI'23 paper on detecting attacks.
3. **Missing evaluation of recent defenses:** The report did not evaluate the robustness of BATPAL (ICLR 2026, by Kiarash) or earlier robustification approaches such as EIR-MAPPO and RAP.

The overall assessment: the report is "not very good at citing or using recent related work."

## Analysis of Each Criticism

### 1. Adversarial Training Methodology is Underspecified

This is a valid criticism. The report's FGSM adversarial training (Section 4.4.2) describes applying perturbations to "the agent's observation" with probability 0.5 during PPO data collection, but:

- It is ambiguous whether all agents or just one are perturbed during training.
- There is no specification of how many victim agents are in the setup.
- The approach — injecting FGSM noise during on-policy rollouts — is not standard adversarial training in the RARL (Robust Adversarial RL) sense. Real adversarial training in MARL typically involves training against an adversary in a min-max framework (as in Pinto et al.'s RARL), not merely augmenting observations with gradient-based noise. The report itself notes that this "destabilizes learning," which is unsurprising: corrupting advantage estimates during PPO training is fundamentally different from proper min-max adversarial training.

This vagueness is a characteristic AI failure mode — generating text that sounds rigorous without actually pinning down every operational detail that a practitioner or reviewer would expect.

### 2. Insufficient Use of Kazari et al. (IJCAI'23)

This criticism is partially valid. The paper is cited in the bibliography as `kazari2023decentralized` and appears in the Related Work section (Section 3.4), but the anomaly detection method description in Section 4.4.4 does not cite it where the method is actually described. More importantly, the report does not engage deeply with Kazari et al.'s detection approach — it does not compare its simpler method against theirs, discuss the differences, or explain why the simpler version was chosen. For a thesis whose supervisors are closely connected to this work, this is a significant oversight.

### 3. Missing Evaluation of Recent Defense Methods (BATPAL, EIR-MAPPO, RAP)

This is the most damaging criticism. The report evaluates only basic defenses: noise augmentation, naive FGSM adversarial training, and observation smoothing. It entirely misses the line of robustification methods that the research community — and the supervisors' own group — has published:

- **BATPAL** (ICLR 2026) — a recent defense method by the supervisor's collaborator
- **EIR-MAPPO** — a robustness method for MAPPO
- **RAP** (Robust Adversarial Policy) — earlier work on robust policies

The related work section covering defenses is thin (~20 lines, with a single sentence on anomaly detection and one on adversarial training). For a thesis whose topic is explicitly about attacking cooperative MARL and reasoning about countermeasures, this gap undermines the contribution.

## What This Reveals About AI-Generated Research

The three failures share a common root cause: **the AI does not know what it does not know.**

The report confidently evaluates toy defenses without realizing the field has moved past them. It generates plausible-sounding methodology without the precision that comes from deeply understanding the community's conventions. It cites a relevant paper but does not engage with it the way someone who had actually read it and discussed it in a seminar would.

A human student embedded in the research group would absorb much of this context osmotically — from supervisor meetings, reading groups, hallway conversations, and the social knowledge of which papers matter most to the people evaluating the work. The AI gets none of that ambient context unless it is explicitly injected into the prompt.

This is a concrete demonstration of the gap between "surface-level competent" and "genuinely situated in a research community" — exactly the kind of gap the meta-thesis aims to examine.

## How the AI Should Have Been Prompted

Each failure mode is addressable with specific prompting strategies. These represent lessons for anyone attempting to use AI for research writing.

### Strategy 1: Feed the Supervisor Context Explicitly

The AI had no way to discover the supervisor group's research focus from the prompt alone. The single highest-leverage fix would have been:

> "The supervisors are Axel Andersson and Gyorgy Dan at KTH. A key collaborator is Kiarash Kazari. Before writing the related work or designing defenses, search for all recent publications by Kazari and Dan on adversarial MARL, anomaly detection in MARL, and robust cooperative RL. Specifically look for their IJCAI'23 paper on decentralized anomaly detection and any ICLR 2026 submissions. Build your defense evaluation around methods from this group's work, not just textbook baselines."

The AI did find and cite `kazari2023decentralized`, but only superficially — because nothing in the prompt signaled that this is the supervisor's collaborator and their work is central to the thesis.

### Strategy 2: Force a Literature Grounding Phase Before Method Design

The AI jumped from general knowledge of FGSM and PGD to implementing them, skipping the step a real student performs: reading 10-20 papers and adopting their experimental conventions. A prompt structure like:

> "Before designing any experiments or writing any code:
> 1. Search for the 15 most-cited papers on adversarial attacks AND defenses in cooperative MARL from 2022-2026.
> 2. Search specifically for: EIR-MAPPO, RAP, BATPAL, and any robust training methods for MAPPO/QMIX.
> 3. Compile a table of: paper, attack/defense type, environment, key metrics, and what they compare against.
> 4. Design our experiments to evaluate at least 2 existing defense methods from this table, not just naive baselines."

This forces the AI to ground its method design in the actual state of the field rather than generating plausible-sounding but outdated approaches.

### Strategy 3: Demand Methodological Precision via an Adversarial Review Step

The vagueness in Section 4.4.2 (whose observations? how many victims?) is a classic AI failure mode — generating text that sounds rigorous without actually pinning down every detail. A built-in review step would catch this:

> "After drafting each method section, critique it as a hostile reviewer would. For every experimental setup, verify that ALL of the following are unambiguously specified: which agents are attacked, how many, whether the attack is white-box or black-box, what the attacker has access to, the exact threat model, and how this relates to prior definitions in the literature (cite the source of the threat model). If 'adversarial training' is used, specify whether it is min-max RARL-style training or just data augmentation — these are fundamentally different."

### Strategy 4: Separate "What AI Knows" from "What the Field Knows"

The deepest issue is that the AI treated its parametric knowledge as a sufficient literature review. A prompt guardrail:

> "Your training data is not a substitute for a literature search. For every claim of the form 'X has not been studied' or 'we are the first to Y,' perform a web search to verify. For every defense you evaluate, search for whether stronger defenses exist in the literature that we should compare against. If you cannot find a paper, say so explicitly rather than omitting the comparison."

## The Meta-Lesson

The single most impactful change would have been a two-sentence addition to the original prompt:

> "The supervisors' group has published on anomaly detection for MARL (IJCAI'23) and robust MARL training (BATPAL, ICLR 2026). Our defense evaluation must engage seriously with their methods, not just textbook adversarial training."

That is the kind of context that takes 30 seconds to write but fundamentally changes the quality of the output. It is also exactly the kind of judgment that the meta-thesis argues humans still need to provide: not the writing itself, not the implementation, not the statistical analysis — but the situated knowledge of what matters, to whom, and why. The AI can do the labor, but it cannot yet supply the taste.
