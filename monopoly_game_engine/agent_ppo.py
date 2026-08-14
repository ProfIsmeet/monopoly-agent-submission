"""
PPO Agent (Section V-A-1 and V-B).

Implements actor-critic PPO with clipped surrogate objective and
truncated GAE advantage estimation, as described in the paper.

Hybrid mode: BUY_PROPERTY and ACCEPT_TRADE are handled by fixed rules,
             all other actions go through the actor network.

Fixes applied
─────────────
Bug 1 – Hybrid actions no longer stored in the PPO buffer.
    choose_action() now returns a sentinel log_prob=None to signal that
    the action was chosen by the fixed policy and must not be stored.
    train.py should check `if log_prob is not None` before calling store().

Bug 2 – Correct action mask used during update().
    Each transition now records the allowed_actions mask at collection
    time. The PPO update uses that per-step mask instead of an all-ones
    mask, so the actor is never trained on actions that were illegal in
    the state where the transition was collected.

Bug 3 – Proper GAE bootstrap for mid-game rollout boundaries.
    _compute_gae() now accepts an explicit `last_next_value` argument.
    When the buffer is flushed mid-game (done=False at the boundary),
    train.py should query the critic on the next state and pass that
    value here so the advantage estimate is not truncated to zero.
"""

from typing import List, Optional
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .actions import ACTION_SPACE_SIZE, OFFSETS, ActionType
from .constants import (
    COLOR_GROUPS,
    JAIL_BAIL,
    NUM_PLAYERS,
    PROPERTY_IDS,
    RULESET_VERSION,
    TRADE_CASH_LEVELS,
)
from .networks import ActorNetwork, CriticNetwork
from .state import STATE_DIM

# ── Hybrid fixed-policy decisions ─────────────────────────────────────────────


def fixed_buy_decision(env, pid: int) -> bool:
    player = env.players[pid]
    sq = player.position
    if sq not in env.properties:
        return False
    prop = env.properties[sq]
    if prop.owner is not None or not player.can_afford(prop.price):
        return False

    # Always buy if it completes a monopoly
    color = prop.color
    group = COLOR_GROUPS.get(color, [])
    if group:
        owned = sum(1 for s in group if env.properties[s].owner == pid)
        if owned + 1 == len(group):
            return True

    return player.cash >= prop.price + 100


def fixed_accept_trade_decision(env, pid: int) -> bool:
    offer = env._incoming_trade(pid)
    if offer is None:
        return False

    if offer.offered_prop:
        color = offer.offered_prop.color
        group = COLOR_GROUPS.get(color, [])
        if group:
            owned_after = sum(
                1
                for s in group
                if env.properties[s].owner == pid
                or env.properties[s] == offer.offered_prop
            )
            if owned_after == len(group):
                return True

    # Never hand the sender a completed real-estate monopoly via the
    # property they're requesting from us, no matter how cash-favorable
    # the offer looks — house/hotel rent escalation on a finished set
    # outweighs a one-off cash-neutral (or even cash-positive) gain.
    # Mirror of fixed_sell_trade_offer_decision's guard on our own
    # initiated sells; railroad/utility completions are exempt since their
    # rent is capped with no house-style escalation.
    if offer.requested_prop:
        color = offer.requested_prop.color
        group = COLOR_GROUPS.get(color, [])
        is_infra = color in ("railroad", "utility")
        if group and not is_infra:
            owned_by_sender = sum(
                1 for s in group if env.properties[s].owner == offer.from_player
            )
            if owned_by_sender + 1 >= len(group):
                return False

    nwo = offer.net_worth()
    return nwo >= 0


def fixed_trade_offer_decision(env, pid: int, allowed_actions: list) -> Optional[int]:
    """
    Rule-based trade initiation: only offer to buy a property when it would
    complete one of our own color groups, at the highest affordable price
    tier (most likely to be accepted by the responder — a low offer is a
    low offer for them too). Self-contained — uses only the
    already-computed legal-action list the caller has, filtered to the
    buy-trade range; avoids re-deriving trade legality from scratch.
    """
    if pid in env.pending_trades:
        return None

    stride = len(PROPERTY_IDS) * len(TRADE_CASH_LEVELS)
    best_price_idx: Optional[int] = None
    best_action: Optional[int] = None

    for action_id in allowed_actions:
        if not (OFFSETS["buy_trade"] <= action_id < OFFSETS["sell_trade"]):
            continue
        local = (action_id - OFFSETS["buy_trade"]) % stride
        prop_idx = local // len(TRADE_CASH_LEVELS)
        price_idx = local % len(TRADE_CASH_LEVELS)
        sq = PROPERTY_IDS[prop_idx]
        prop = env.properties[sq]

        group = COLOR_GROUPS.get(prop.color, [])
        if not group:
            continue
        owned_after = sum(1 for s in group if env.properties[s].owner == pid) + 1
        if owned_after != len(group):
            continue

        if best_price_idx is None or price_idx > best_price_idx:
            best_price_idx = price_idx
            best_action = action_id

    return best_action


def fixed_sell_trade_offer_decision(env, pid: int, allowed_actions: list) -> Optional[int]:
    """
    Rule-based sell-offer initiation: liquidate a spare property (one we
    hold no group-synergy on) for cash at fair market value (1.0x — always
    net_worth()==0 for the buyer, so any rational responder should accept
    it, not just personalities with a generous acceptance margin).

    Never offered to a target whose REAL-ESTATE color group it would
    complete: giving an opponent a free (or discounted) house-buildable
    monopoly is a large strategic loss (hotel rent escalation) that
    outweighs the cash gained, even though the trade would clearly be
    "accepted". This is the mirror-image guard of fixed_trade_offer_decision
    (which only *buys* to complete our own monopoly, never sells one away).

    Railroad/utility completions are exempt from that guard: their rent is
    capped (max $200 for 4 railroads, 10x-dice for 2 utilities) with no
    house-style escalation, so the downside of completing one for an
    opponent is bounded and small — and verified against every
    _should_accept_trade in agents_fixed.py, TheRailBaron (FPAgentF) is the
    *only* personality that ever accepts a spare-infrastructure sale in the
    first place (it only accepts railroads/utilities it doesn't already
    own), so refusing to ever complete an infra set would make this rule
    permanently unusable against that personality for no real safety gain.

    Fires whenever cash is on the lower side (< 300, matching the
    mortgage-trigger cash floors used by the fixed-policy opponents in
    agents_fixed.py) rather than only in acute distress — turns dead
    weight into liquidity before it's needed, instead of panic-selling to
    the bank at mortgage value once already cash-strapped.
    """
    if pid in env.pending_trades:
        return None
    if env.players[pid].cash >= 300:
        return None

    FAIR_PRICE_IDX = 1  # 1.0x — net_worth()==0 for the buyer
    others = [i for i in range(NUM_PLAYERS) if i != pid]
    stride = len(PROPERTY_IDS) * len(TRADE_CASH_LEVELS)
    best_price: Optional[int] = None
    best_action: Optional[int] = None

    for action_id in allowed_actions:
        if not (OFFSETS["sell_trade"] <= action_id < OFFSETS["exch_trade"]):
            continue
        offset = action_id - OFFSETS["sell_trade"]
        t_idx = offset // stride
        local = offset % stride
        prop_idx = local // len(TRADE_CASH_LEVELS)
        price_idx = local % len(TRADE_CASH_LEVELS)
        if price_idx != FAIR_PRICE_IDX or t_idx >= len(others):
            continue
        target_pid = others[t_idx]
        sq = PROPERTY_IDS[prop_idx]
        prop = env.properties[sq]

        group = COLOR_GROUPS.get(prop.color, [])
        if not group:
            continue
        owned_by_us = sum(1 for s in group if env.properties[s].owner == pid)
        if owned_by_us > 1:
            continue  # keep properties we're still building toward
        owned_by_target = sum(1 for s in group if env.properties[s].owner == target_pid)
        is_infra = prop.color in ("railroad", "utility")
        if not is_infra and owned_by_target + 1 >= len(group):
            continue  # never hand the buyer a completed real-estate monopoly

        # Prefer offloading the most valuable safe spare first (best cash return).
        if best_price is None or prop.price > best_price:
            best_price = prop.price
            best_action = action_id

    return best_action


def fixed_exchange_trade_offer_decision(env, pid: int, allowed_actions: list) -> Optional[int]:
    """
    Rule-based exchange initiation: swap a spare, no-synergy property we
    hold for an opponent's property that would complete one of our color
    groups — pure barter, no cash involved (see env._trade_offer_actions).

    Two guards keep this from being a net loss:
      - We only ever request a piece that completes OUR monopoly (mirrors
        fixed_trade_offer_decision's buy-side rule).
      - We never offer a piece that would complete THE OPPONENT's monopoly
        in its own color group (same anti-gift guard as the sell rule) —
        an exchange that fixes our board while breaking theirs is not a
        trade a rational opponent accepts anyway, but we don't want the
        neural policy's training signal muddied by offers that would be
        catastrophic if a future (differently-coded) opponent accepted them.
      - We only offer a piece worth at least as much as what we're asking
        for (offered.price >= requested.price), so the deal reads as fair
        or favorable to the responder and is more likely accepted.
    """
    if pid in env.pending_trades:
        return None

    others = [i for i in range(NUM_PLAYERS) if i != pid]
    n_props = len(PROPERTY_IDS)
    stride = n_props * (n_props - 1)

    for action_id in allowed_actions:
        if not (OFFSETS["exch_trade"] <= action_id < OFFSETS["auction"]):
            continue
        offset = action_id - OFFSETS["exch_trade"]
        t_idx = offset // stride
        local = offset % stride
        offer_idx = local // (n_props - 1)
        req_raw = local % (n_props - 1)
        req_idx = req_raw if req_raw < offer_idx else req_raw + 1
        if t_idx >= len(others):
            continue
        target_pid = others[t_idx]
        offer_sq = PROPERTY_IDS[offer_idx]
        req_sq = PROPERTY_IDS[req_idx]
        offer_prop = env.properties[offer_sq]
        req_prop = env.properties[req_sq]

        # Requested piece must complete OUR group.
        req_group = COLOR_GROUPS.get(req_prop.color, [])
        if not req_group:
            continue
        owned_after = sum(1 for s in req_group if env.properties[s].owner == pid) + 1
        if owned_after != len(req_group):
            continue

        # Offered piece must be a true spare (no synergy for us)...
        offer_group = COLOR_GROUPS.get(offer_prop.color, [])
        if offer_group:
            owned_by_us = sum(1 for s in offer_group if env.properties[s].owner == pid)
            if owned_by_us > 1:
                continue
            # ...and must not complete the opponent's REAL-ESTATE monopoly in
            # return (railroad/utility exempt — see fixed_sell_trade_offer_decision).
            owned_by_target = sum(
                1 for s in offer_group if env.properties[s].owner == target_pid
            )
            is_infra = offer_prop.color in ("railroad", "utility")
            if not is_infra and owned_by_target + 1 >= len(offer_group):
                continue

        if offer_prop.price < req_prop.price:
            continue  # lowball swaps are unlikely to be accepted

        return action_id

    return None


# ── Experience buffer ─────────────────────────────────────────────────────────


class PPOBuffer:
    """Stores a single rollout trajectory for PPO updates."""

    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        self.action_masks = []  # FIX 2: per-step allowed-action masks

    def store(self, state, action, log_prob, reward, value, done, action_mask):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
        self.action_masks.append(action_mask)  # FIX 2

    def clear(self):
        self.__init__()

    def __len__(self):
        return len(self.states)


# ── PPO Agent ─────────────────────────────────────────────────────────────────


class PPOAgent:
    """
    Actor-critic PPO agent, with optional hybrid mode.

    Parameters mirror Appendix B-A of the paper.
    """

    def __init__(
        self,
        player_id: int,
        hybrid: bool = False,
        lr: float = 1e-4,
        gamma: float = 0.9999,
        lam: float = 0.95,
        clip_eps: float = 0.2,
        entropy_coef: float = 0.05,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        n_steps: int = 1024,
        n_epochs: int = 4,
        batch_size: int = 64,
        hidden_dim: int = 256,
        win_loss_bonus: float = 0.0,
        device: str = "auto",
    ):
        self.player_id = player_id
        self.hybrid = hybrid
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.n_steps = n_steps
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.win_loss_bonus = win_loss_bonus
        self.hidden_dim = hidden_dim
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is not available")
        self.device = torch.device(
            "cuda" if device == "auto" and torch.cuda.is_available() else
            "cpu" if device == "auto" else device
        )

        self.actor = ActorNetwork(hidden_dim).to(self.device)
        self.critic = CriticNetwork(hidden_dim).to(self.device)
        self.opt = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()), lr=lr
        )

        # Private action-sampling RNG, isolated from torch's global default
        # generator. .seed() draws from OS entropy directly, not from the
        # global stream, so constructing this never perturbs it. Tournament
        # harnesses pair identical global-RNG seeds across all four seats;
        # sampling actions from the default generator would desync that.
        self._rng = torch.Generator(device=str(self.device))
        self._rng.seed()

        self.buffer = PPOBuffer()
        self.step_count = 0
        self.games_trained = 0

        # Mask actions permanently handled by fixed policy (hybrid only)
        self.fixed_action_mask = torch.zeros(ACTION_SPACE_SIZE, dtype=torch.bool)
        if hybrid:
            self.fixed_action_mask[int(ActionType.BUY_PROPERTY)] = True
            self.fixed_action_mask[int(ActionType.ACCEPT_TRADE)] = True
            self.fixed_action_mask[OFFSETS["buy_trade"] : OFFSETS["auction"]] = True
            # Selling a property back to the bank pays the exact same cash
            # as mortgaging it (both = prop.mortgage_v) under the identical
            # eligibility condition (owned, unmortgaged, house-free), but
            # permanently surrenders the property to the open market instead
            # of just freezing rent income while keeping ownership. Mortgage
            # strictly dominates sell-to-bank in every situation where both
            # are legal (they always are, together) — never worth letting
            # the network learn otherwise.
            self.fixed_action_mask[OFFSETS["sell_prop"] : OFFSETS["buy_trade"]] = True

    # ── Action selection ──────────────────────────────────────────────────────

    def choose_action(self, state: np.ndarray, env, allowed_actions: List[int]):
        """
        Choose an action. In hybrid mode, intercept BUY and ACCEPT_TRADE.

        Returns (action, log_prob, value, allowed_actions).

        FIX 1: When the hybrid fixed policy fires, log_prob is returned as
        None.  The caller (train.py) must NOT store these transitions in the
        PPO buffer — storing them with log_prob=0 corrupts the importance
        sampling ratio.
        """
        pid = self.player_id

        # Hybrid: handle buy property
        if self.hybrid and int(ActionType.BUY_PROPERTY) in allowed_actions:
            if fixed_buy_decision(env, pid):
                # FIX 1: return None sentinel — caller skips buffer storage
                return int(ActionType.BUY_PROPERTY), None, None, allowed_actions

        # Hybrid: handle trade acceptance
        if self.hybrid and int(ActionType.ACCEPT_TRADE) in allowed_actions:
            pending = next(
                (o for o in env.pending_trades.values() if o.to_player == pid),
                None,
            )
            if pending is not None:
                if fixed_accept_trade_decision(env, pid):
                    return int(ActionType.ACCEPT_TRADE), None, None, allowed_actions
                else:
                    return int(ActionType.DECLINE_TRADE), None, None, allowed_actions

        # Hybrid: handle rule-based trade initiation (monopoly completion only)
        if self.hybrid:
            trade_action = fixed_trade_offer_decision(env, pid, allowed_actions)
            if trade_action is not None:
                # trade_action is drawn directly from allowed_actions, so it
                # is always legal here; no need to recheck membership.
                # Trade-family actions must never appear in the mask handed
                # back to the caller (the whole family is fixed-policy-only
                # in hybrid mode), unlike BUY_PROPERTY/ACCEPT_TRADE above
                # which are single actions safely covered by allowed_actions.
                filtered = [
                    a for a in allowed_actions if not self.fixed_action_mask[a]
                ]
                if not filtered:
                    filtered = [int(ActionType.DO_NOTHING)]
                return trade_action, None, None, filtered

            exch_action = fixed_exchange_trade_offer_decision(env, pid, allowed_actions)
            if exch_action is not None:
                filtered = [
                    a for a in allowed_actions if not self.fixed_action_mask[a]
                ]
                if not filtered:
                    filtered = [int(ActionType.DO_NOTHING)]
                return exch_action, None, None, filtered

            sell_action = fixed_sell_trade_offer_decision(env, pid, allowed_actions)
            if sell_action is not None:
                filtered = [
                    a for a in allowed_actions if not self.fixed_action_mask[a]
                ]
                if not filtered:
                    filtered = [int(ActionType.DO_NOTHING)]
                return sell_action, None, None, filtered

        # Filter out fixed-policy actions from neural net consideration
        nn_allowed = [a for a in allowed_actions if not self.fixed_action_mask[a]]
        if not nn_allowed:
            nn_allowed = [int(ActionType.DO_NOTHING)]

        state_t = torch.as_tensor(
            state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        with torch.inference_mode():
            value = self.critic(state_t).item()
        action, log_prob = self.actor.get_action(state, nn_allowed, generator=self._rng)

        return action, log_prob, value, nn_allowed  # FIX 2: return nn_allowed

    # ── Store experience ──────────────────────────────────────────────────────

    def store(
        self, state, action, log_prob, reward, value, done, allowed_actions: List[int]
    ):
        """
        Store one NN-chosen transition.
        FIX 1: Callers must only call this when log_prob is not None (i.e.
               the action was chosen by the neural network, not the fixed
               hybrid policy).
        FIX 2: allowed_actions is stored so update() can reconstruct the
               per-step action mask.
        """
        # Build boolean mask for the allowed actions at this step (FIX 2)
        mask = torch.zeros(ACTION_SPACE_SIZE, dtype=torch.bool)
        mask[allowed_actions] = True

        self.buffer.store(state, action, log_prob, reward, value, done, mask)
        self.step_count += 1

    def add_win_loss(self, won: bool):
        """Add terminal win/loss bonus to the last stored reward."""
        if self.win_loss_bonus != 0 and len(self.buffer.rewards) > 0:
            self.buffer.rewards[-1] += self.win_loss_bonus * (1.0 if won else -1.0)

    # ── PPO update ────────────────────────────────────────────────────────────

    def update(
        self, last_next_state: Optional[np.ndarray] = None, last_done: bool = False
    ):
        """
        Run PPO update over the current buffer.

        FIX 3: last_next_state / last_done support correct GAE bootstrap.
            Pass the state that followed the last stored transition and
            whether that transition was terminal.  When the rollout ends
            mid-game (last_done=False), the critic evaluates last_next_state
            to get the bootstrap value instead of using 0.
        """
        if len(self.buffer) == 0:
            return {}

        # FIX 3: bootstrap value for the end of the rollout
        if last_next_state is not None and not last_done:
            with torch.inference_mode():
                st = torch.as_tensor(
                    last_next_state, dtype=torch.float32, device=self.device
                ).unsqueeze(0)
                bootstrap_value = self.critic(st).item()
        else:
            bootstrap_value = 0.0

        # Convert to tensors
        states = torch.as_tensor(
            np.array(self.buffer.states), dtype=torch.float32, device=self.device
        )
        actions = torch.as_tensor(
            self.buffer.actions, dtype=torch.long, device=self.device
        )
        old_lps = torch.as_tensor(
            self.buffer.log_probs, dtype=torch.float32, device=self.device
        )
        rewards = self.buffer.rewards
        values = self.buffer.values
        dones = self.buffer.dones
        # FIX 2: per-step masks stacked into (N, ACTION_SPACE_SIZE)
        step_masks = torch.stack(self.buffer.action_masks).to(self.device)

        # FIX 3: pass bootstrap value into GAE
        advantages = self._compute_gae(rewards, values, dones, bootstrap_value)
        returns = advantages + torch.as_tensor(
            values, dtype=torch.float32, device=self.device
        )
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        else:
            advantages = advantages - advantages.mean()

        stats = {"actor_loss": 0.0, "critic_loss": 0.0, "entropy": 0.0}
        n_batches = 0

        for _ in range(self.n_epochs):
            indices = torch.randperm(len(states), device=self.device)
            for start in range(0, len(states), self.batch_size):
                idx = indices[start : start + self.batch_size]
                if len(idx) < 2:
                    continue

                sb = states[idx]
                ab = actions[idx]
                olp = old_lps[idx]
                adv = advantages[idx]
                ret = returns[idx]
                mask = step_masks[idx]  # FIX 2: per-step masks for this batch

                # FIX 2: use the recorded per-step mask, not all-ones
                log_probs_all = self.actor(sb, mask)
                new_lps = log_probs_all.gather(1, ab.unsqueeze(1)).squeeze(1)
                probs_all = log_probs_all.exp()
                # Masked actions have log_prob=-inf and prob=0. Multiplying
                # them directly yields 0 * -inf = nan, which can poison the
                # PPO loss and corrupt the actor weights.
                safe_log_probs = torch.where(
                    mask, log_probs_all, torch.zeros_like(log_probs_all)
                )
                entropy = -(probs_all * safe_log_probs).sum(dim=-1).mean()

                ratio = (new_lps - olp).exp()
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv
                actor_loss = -torch.min(surr1, surr2).mean()

                values_pred = self.critic(sb)
                critic_loss = nn.MSELoss()(values_pred, ret)

                loss = (
                    actor_loss
                    + self.value_coef * critic_loss
                    - self.entropy_coef * entropy
                )

                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()),
                    self.max_grad_norm,
                )
                self.opt.step()

                stats["actor_loss"] += actor_loss.item()
                stats["critic_loss"] += critic_loss.item()
                stats["entropy"] += entropy.item()
                n_batches += 1

        self.buffer.clear()
        if n_batches > 0:
            return {k: v / n_batches for k, v in stats.items()}
        return stats

    def _compute_gae(
        self, rewards, values, dones, last_next_value: float = 0.0
    ) -> torch.Tensor:
        """
        Generalised Advantage Estimation.

        FIX 3: last_next_value is the critic's estimate of V(s_{T+1}).
        For terminal transitions it is 0; for mid-game buffer flushes it is
        the critic's evaluation of the state that followed the last stored
        step, preventing the advantage from being truncated to zero.
        """
        advantages = torch.zeros(len(rewards), device=self.device)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            if t + 1 < len(values):
                next_val = values[t + 1]
            else:
                # FIX 3: use bootstrapped value, not hard-coded 0
                next_val = last_next_value
            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.lam * (1 - dones[t]) * gae
            advantages[t] = gae
        return advantages

    def save(self, path: str):
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".tmp")
        torch.save(
            {
                "format_version": 3,
                "ruleset": RULESET_VERSION,
                "state_dim": STATE_DIM,
                "action_dim": ACTION_SPACE_SIZE,
                "player_id": self.player_id,
                "hybrid": self.hybrid,
                "hidden_dim": self.hidden_dim,
                "step_count": self.step_count,
                "games_trained": self.games_trained,
                "training_config": {
                    "gamma": self.gamma,
                    "lam": self.lam,
                    "clip_eps": self.clip_eps,
                    "entropy_coef": self.entropy_coef,
                    "value_coef": self.value_coef,
                    "max_grad_norm": self.max_grad_norm,
                    "n_steps": self.n_steps,
                    "n_epochs": self.n_epochs,
                    "batch_size": self.batch_size,
                    "win_loss_bonus": self.win_loss_bonus,
                },
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "optimizer": self.opt.state_dict(),
            },
            temporary,
        )
        os.replace(temporary, destination)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        if ckpt.get("format_version") is None:
            raise ValueError(
                "Legacy PPO checkpoint is incompatible with "
                f"{RULESET_VERSION}; train a new checkpoint."
            )
        expected = {
            "format_version": 3,
            "ruleset": RULESET_VERSION,
            "state_dim": STATE_DIM,
            "action_dim": ACTION_SPACE_SIZE,
            "player_id": self.player_id,
            "hybrid": self.hybrid,
            "hidden_dim": self.hidden_dim,
        }
        actual = {key: ckpt.get(key) for key in expected}
        if actual != expected:
            raise ValueError(
                f"Incompatible PPO checkpoint metadata: {actual}; "
                f"expected {expected}."
            )
        try:
            self.actor.load_state_dict(ckpt["actor"])
            self.critic.load_state_dict(ckpt["critic"])
        except RuntimeError as exc:
            raise ValueError(
                f"PPO checkpoint network is incompatible with {RULESET_VERSION}."
            ) from exc
        if "optimizer" in ckpt:
            self.opt.load_state_dict(ckpt["optimizer"])
            for state in self.opt.state.values():
                for key, value in state.items():
                    if torch.is_tensor(value):
                        state[key] = value.to(self.device)
        for key, value in ckpt.get("training_config", {}).items():
            setattr(self, key, value)
        self.step_count = int(ckpt.get("step_count", 0))
        self.games_trained = int(ckpt.get("games_trained", 0))
