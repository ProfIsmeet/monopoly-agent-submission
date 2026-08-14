"""
monopoly_game_engine – Shared ppo-plus-v2 Monopoly simulator
=============================================================

Based on:
  "Decision Making in Monopoly Using a Hybrid Deep Reinforcement
   Learning Approach"
  Bonjour et al., IEEE TETCI, Vol. 6, No. 6, December 2022.

Quick start
-----------
>>> from monopoly_game_engine import train_ppo, train_ddqn, evaluate_agent
>>> agent, history = train_ppo(hybrid=True, n_games=2000)
>>> results = evaluate_agent(agent, is_ppo=True, n_games=2000)
"""

import random

import numpy as np
import torch

from .env          import MonopolyEnv
from .agent_ppo    import PPOAgent
from .agent_ddqn   import DDQNAgent
from .agents_fixed import FPAgentA, FPAgentB, FPAgentC
from .train        import train, evaluate
from .state        import build_state_vector
from .actions      import ACTION_SPACE_SIZE, action_to_description


def train_ppo(
    hybrid: bool = True,
    player_id: int = 0,
    n_games: int = 2000,
    log_every: int = 100,
    checkpoint_every: int = 0,
    checkpoint_path: str | None = None,
    watchdog=None,
    seed: int = 42,
    resume_path: str | None = None,
    game_log_path: str | None = None,
    action_log_path: str | None = None,
    stats_log_path: str | None = None,
    opponent_pool: str = "diverse",
    entropy_coef_override: float | None = None,
    fresh_optimizer: bool = False,
    **kwargs,
):
    """Train a PPO agent. Set hybrid=True for the hybrid approach."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    agent = PPOAgent(player_id=player_id, hybrid=hybrid, **kwargs)
    resumed = resume_path is not None
    if resumed:
        agent.load(resume_path)
        n_games = max(0, n_games - agent.games_trained)
        # A resumed checkpoint's training_config (entropy_coef) and Adam
        # optimizer state were tuned against whatever opponent pool produced
        # it. Switching to a harder/narrower pool (e.g. rival_mix) with that
        # same low entropy_coef and stale Adam moments gives the policy no
        # fresh exploration budget and collapses it into a low-entropy local
        # optimum fast (observed: ppo_rivals_v1, entropy 0.35->0.24 over 14
        # windows with win% drifting 23.8%->19.1%). These overrides exist to
        # break that inheritance when deliberately changing opponent regime.
        if entropy_coef_override is not None:
            agent.entropy_coef = entropy_coef_override
        if fresh_optimizer:
            import torch.optim as optim

            agent.opt = optim.Adam(
                list(agent.actor.parameters()) + list(agent.critic.parameters()),
                lr=1e-4,
            )
    history = train(
        agent,
        is_ppo=True,
        hybrid=hybrid,
        n_games=n_games,
        log_every=log_every,
        checkpoint_every=checkpoint_every,
        checkpoint_path=checkpoint_path,
        watchdog=watchdog,
        seed=seed,
        game_log_path=game_log_path,
        action_log_path=action_log_path,
        stats_log_path=stats_log_path,
        resume=resumed,
        opponent_pool=opponent_pool,
    )
    return agent, history


def train_ddqn(
    hybrid: bool = True,
    player_id: int = 0,
    n_games: int = 10_000,
    log_every: int = 100,
    checkpoint_every: int = 0,
    checkpoint_path: str | None = None,
    watchdog=None,
    seed: int = 42,
    resume_path: str | None = None,
    game_log_path: str | None = None,
    action_log_path: str | None = None,
    stats_log_path: str | None = None,
    **kwargs,
):
    """Train a DDQN agent. Set hybrid=True for the hybrid approach."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    agent = DDQNAgent(player_id=player_id, hybrid=hybrid, **kwargs)
    resumed = resume_path is not None
    if resumed:
        agent.load(resume_path)
        n_games = max(0, n_games - agent.games_trained)
    history = train(
        agent,
        is_ppo=False,
        hybrid=hybrid,
        n_games=n_games,
        log_every=log_every,
        checkpoint_every=checkpoint_every,
        checkpoint_path=checkpoint_path,
        watchdog=watchdog,
        seed=seed,
        game_log_path=game_log_path,
        action_log_path=action_log_path,
        stats_log_path=stats_log_path,
        resume=resumed,
    )
    return agent, history


def evaluate_agent(agent, is_ppo: bool, n_games: int = 2000, n_runs: int = 5):
    """Evaluate a trained agent against fixed-policy opponents."""
    return evaluate(agent, is_ppo=is_ppo, n_games=n_games, n_runs=n_runs)


__all__ = [
    "MonopolyEnv",
    "PPOAgent", "DDQNAgent",
    "FPAgentA", "FPAgentB", "FPAgentC",
    "train_ppo", "train_ddqn", "evaluate_agent",
    "build_state_vector", "ACTION_SPACE_SIZE", "action_to_description",
]
