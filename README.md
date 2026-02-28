# Scrabble RL

A fully playable, rules-complete Scrabble engine with reinforcement learning agents trained via Tabular Q-Learning, DQN, PPO, and Self-Play PPO. Built as an open-source OpenAI Gym environment.

## Overview

This project implements Scrabble from scratch — including legal move generation, scoring, cross-word validation, and tile bag management — and wraps it in a Gym-compatible environment for RL training.

The core move generator uses a **GADDAG data structure** for efficient legal move enumeration from any board state, which is the standard approach used in competitive Scrabble AI.

## Project Structure
```
├── Board.py              # 15×15 board, move generation, scoring, cross-set updates
├── Game.py               # Game loop, player management, tile drawing
├── Gaddag.py             # GADDAG lexicon construction and traversal
├── Tile.py               # Tile representation with multipliers and cross-sets
├── utils.py              # Board initialization, state encoding, scoring tables
├── scrabble_rl/
│   ├── env.py            # OpenAI Gym environment (single-agent)
│   ├── selfplay_env_dynamic.py  # Self-play environment with relaxation scheduling
│   ├── model_cnnppo.py   # CNN-based Actor-Critic (PPO)
│   ├── model.py          # MLP-based Actor-Critic (baseline)
│   ├── ppo.py            # PPO trainer
│   ├── deep_q_agent.py   # DQN agent with replay buffer
│   ├── q_agent.py        # Tabular Q-learning agent
│   └── ppo_utils.py      # GAE computation
├── evaluate_best_agent.py  # Evaluation vs Random/Greedy/Self-Play opponents
└── evaluate_greedy_vs_greedy.py  # Greedy baseline benchmark
```

## Agents Implemented

| Agent | Description |
|-------|-------------|
| Tabular Q-Learning | State-space approximation via bag/rack/score tuple |
| DQN | Neural Q-network with experience replay and target network |
| PPO (MLP) | Actor-Critic with GAE, flat board encoding |
| PPO (CNN) | Actor-Critic with 2-layer CNN board encoder + MLP for rack/bag/score |
| Self-Play PPO | Two CNN agents trained adversarially with relaxation scheduling |

## State Representation

The observation space encodes:
- **Board** `(2, 15, 15)` — letter indices and multiplier types per tile
- **Rack** `(27,)` — letter counts including blanks
- **Bag** `(27,)` — remaining tile counts
- **Score** `(2,)` — current and opponent score

## Move Selection

At each step, the agent selects from the **top-N legal moves** ranked by the GADDAG generator. The action is an index into this candidate list, decoupling the policy from raw board coordinates.

## Results

| Agent | Opponent | Win Rate |
|-------|----------|----------|
| Self-Play PPO (CNN) | Greedy | 95% |
| Self-Play PPO (CNN) | Random | 95%+ |
| DQN | Random | — |

## Getting Started
```bash
pip install -r requirements.txt

# Build the GADDAG dictionary (first run only)
python Game.py

# Train PPO-CNN agent
python scrabble_rl/train_ppo_cnn.py

# Evaluate against greedy opponent
python evaluate_best_agent.py
```

Requires a `dictionary.txt` wordlist (TWL06 or SOWPODS). The GADDAG is serialized to `dictionary.p` after first construction.

## References

- [GADDAG data structure](https://www.cs.cmu.edu/afs/cs/academic/class/15451-s06/www/lectures/scrabble.pdf) — Gordon (1994)
- [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347) — Schulman et al.
