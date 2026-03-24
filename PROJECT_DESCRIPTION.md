# Project Description

> Extracted from the KTH NSE Bachelor Projects 2025 catalog (cybersecurity context).

## Adversarial Attacks on Multi-Agent Reinforcement Learning Systems

In this project, the goal is to find effective attacks against multi-agent reinforcement learning (MARL) systems. A MARL system is a system where multiple autonomous agents operate in the same environment and wants to accomplish some goal. If the agents work together, as a team, it is called cooperative MARL (c-MARL). An example of this is a search-and-rescue task where multiple autonomous agents (e.g., drones) are searching for a target.

In the project we will study how robust MARL algorithms are by trying to construct attacks that causes the team of autonomous agent to fail their mission. A possible avenue could be to perturb the agents observations in some clever way, causing it to take bad decisions. One could consider both training and inference time attacks, that is, is the attacker present during the training of these algorithms or when the agent is actually used?

### Main Tasks of this Project

1. Choose an interesting and not too complex c-MARL task.
2. Design a simple and effective attack strategy against the MARL system that causes the system to fail the mission or decrease the general performance. This could be:
   - **Perturbation Attacks:** Introduce small changes in one or several of the agents observations, affecting decision making at inference-time or during training.
   - **Behavioral Attacks:** Introduce a malicious agent whose goal is to minimize the overall team reward.
3. Interact with open-source environments commonly used in RL and MARL research such as OpenAI's Gym or PettingZoo.
4. Train a MARL system to understand the baseline performance.
5. Implement the Attack Strategy.
6. Evaluation. Measure how the attacks affect the overall performance of the MARL system.
7. Report and Presentation: Document your findings and the process. Reason about potential countermeasures.

### Tools we will use

Python and some automatic differentiation library like PyTorch, TensorFlow or Jax. OpenAI's Gym or PettingZoo for simulating an environment.

### Related Work

1. Lin, Jieyu & Dzeparoska, Kristina & Zhang, Sai & Leon-Garcia, A. & Papernot, Nicolas. *On the Robustness of Cooperative Multi-Agent Reinforcement Learning.* SPW 2020, 62-68.
2. Pinto, Lerrel & Davidson, James & Sukthankar, Rahul & Gupta, Abhinav. *Robust Adversarial Reinforcement Learning.* ICML 2017.
