"""
Tournament entrypoint.

The organizer's actual runner (/opt/exposure-agent-runner.py) requires a
module-level function:

    def choose_action(state, player_id, allowed_actions): ...

No `env` object is passed — the previous version of this file assumed a
class-based `Agent(player_id)` with `choose_action(self, state,
allowed_actions, env)`, based on a stale/incorrect README elsewhere. That
crashed the live harness with:
    RuntimeError: agent.py must define choose_action(state, player_id, allowed_actions)

PPOAgent's hybrid rules (fixed_buy_decision, fixed_accept_trade_decision in
monopoly_game_engine/agent_ppo.py) need real env.players[pid]/env.properties/
env.pending_trades lookups, which don't exist here. Since the 300-dim state
vector is self-perspective-normalized and already encodes full property
ownership, cash, and incoming-trade details (see
monopoly_game_engine/state.py build_state_vector), the two single-action
hybrid rules (buy / accept-trade) are reimplemented below by decoding those
fields directly from `state`, using the static PROPERTIES/COLOR_GROUPS
tables for price/color lookups instead of a live env. This reproduces
agent_ppo.py's fixed_buy_decision/fixed_accept_trade_decision exactly.

The three trade-*initiation* rules (offer/exchange/sell) are NOT
reconstructed — they need broader env context than the state vector
carries. They're simply never fired, so those action indices stay filtered
out of `nn_allowed` via the same `fixed_action_mask` used during training
(the actor network was never trained to select them anyway, since hybrid
mode always intercepted them), and the agent falls through to normal
network inference for everything else.
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from monopoly_game_engine.agent_ppo import PPOAgent
from monopoly_game_engine.actions import ActionType
from monopoly_game_engine.constants import COLOR_GROUPS, NUM_PLAYERS, PROPERTIES, PROPERTY_IDS
from monopoly_game_engine.state import STATE_DIM

MODEL_PATH = str(ROOT / "ppo_v4_hybridfix.pt")

# ── State-vector layout (must mirror monopoly_game_engine/state.py) ────────────
# Player block: 16 dims (4 players x 4 feats), own player first.
# Property block: 224 dims starting at 16 (28 properties x 8 feats).
_PROPERTY_BLOCK_START = 16
_INCOMING_SENDER_START = 289  # 5 slots: [no-sender, order[0]..order[3]]
_INCOMING_OFFERED_PROP = 294
_INCOMING_REQUESTED_PROP = 295
_INCOMING_CASH_OFFERED = 296
_INCOMING_CASH_REQUESTED = 297

# Loaded once at import time (during the runner's 60s startup window), not
# lazily on the first decision call — generation 2 failed because loading
# the checkpoint inside the first 2-second decision budget timed out.
_agent = PPOAgent(player_id=0, hybrid=True, device="cpu")
if Path(MODEL_PATH).exists():
    _agent.load(MODEL_PATH)

_agent.actor.eval()

# Warm up during the allowed 60-second startup period.
_agent.actor.get_action(
    np.zeros(STATE_DIM, dtype=np.float32),
    [int(ActionType.DO_NOTHING)],
    generator=_agent._rng,
)


def _get_agent(player_id):
    _agent.player_id = int(player_id)
    return _agent


def _own_cash(state):
    return float(state[1]) * 5000.0


def _own_position(state):
    return int(round(float(state[0]) * 39.0))


def _prop_owner(state, square_id):
    offset = _PROPERTY_BLOCK_START + PROPERTY_IDS.index(square_id) * 8
    for pid in range(NUM_PLAYERS):
        if state[offset + pid] >= 0.5:
            return pid
    return None


def _group_owned_count(state, color, by_pid):
    group = COLOR_GROUPS.get(color, [])
    return sum(1 for sq in group if _prop_owner(state, sq) == by_pid)


def _buy_decision(state, player_id, square_id):
    prop = PROPERTIES[square_id]
    if _prop_owner(state, square_id) is not None:
        return False
    cash = _own_cash(state)
    if cash < prop["price"]:
        return False
    color = prop["color"]
    group = COLOR_GROUPS.get(color, [])
    if group and _group_owned_count(state, color, player_id) + 1 == len(group):
        return True
    return cash >= prop["price"] + 100


def _decode_prop_index(value):
    if value <= 0.0:
        return None
    idx = int(round(value * (len(PROPERTY_IDS) + 1))) - 1
    if 0 <= idx < len(PROPERTY_IDS):
        return PROPERTY_IDS[idx]
    return None


def _decode_incoming_trade(state, player_id):
    if state[_INCOMING_SENDER_START] >= 0.5:
        return None  # explicit no-sender flag
    order = [player_id] + [i for i in range(NUM_PLAYERS) if i != player_id]
    sender = None
    for k in range(NUM_PLAYERS):
        if state[_INCOMING_SENDER_START + 1 + k] >= 0.5:
            sender = order[k]
            break
    if sender is None:
        return None
    return {
        "from_player": sender,
        "offered_sq": _decode_prop_index(float(state[_INCOMING_OFFERED_PROP])),
        "requested_sq": _decode_prop_index(float(state[_INCOMING_REQUESTED_PROP])),
        "cash_offered": float(state[_INCOMING_CASH_OFFERED]) * 2000.0,
        "cash_requested": float(state[_INCOMING_CASH_REQUESTED]) * 2000.0,
    }


def _accept_trade_decision(state, player_id):
    offer = _decode_incoming_trade(state, player_id)
    if offer is None:
        return False

    if offer["offered_sq"] is not None:
        color = PROPERTIES[offer["offered_sq"]]["color"]
        group = COLOR_GROUPS.get(color, [])
        if group:
            owned_after = sum(
                1
                for sq in group
                if _prop_owner(state, sq) == player_id or sq == offer["offered_sq"]
            )
            if owned_after == len(group):
                return True

    if offer["requested_sq"] is not None:
        color = PROPERTIES[offer["requested_sq"]]["color"]
        group = COLOR_GROUPS.get(color, [])
        is_infra = color in ("railroad", "utility")
        if group and not is_infra:
            owned_by_sender = _group_owned_count(state, color, offer["from_player"])
            if owned_by_sender + 1 >= len(group):
                return False

    po = PROPERTIES[offer["offered_sq"]]["price"] if offer["offered_sq"] is not None else 0
    pr = PROPERTIES[offer["requested_sq"]]["price"] if offer["requested_sq"] is not None else 0
    net_worth = (po + offer["cash_offered"]) - (pr + offer["cash_requested"])
    return net_worth >= 0


def choose_action(state, player_id, allowed_actions):
    agent = _get_agent(player_id)
    allowed_actions = list(allowed_actions)
    state = np.asarray(state, dtype=np.float32)

    if int(ActionType.BUY_PROPERTY) in allowed_actions:
        sq = _own_position(state)
        if sq in PROPERTIES and _buy_decision(state, player_id, sq):
            return int(ActionType.BUY_PROPERTY)

    if int(ActionType.ACCEPT_TRADE) in allowed_actions:
        if _accept_trade_decision(state, player_id):
            return int(ActionType.ACCEPT_TRADE)
        if int(ActionType.DECLINE_TRADE) in allowed_actions:
            return int(ActionType.DECLINE_TRADE)

    nn_allowed = [a for a in allowed_actions if not agent.fixed_action_mask[a]]
    if not nn_allowed:
        nn_allowed = (
            [int(ActionType.DO_NOTHING)]
            if int(ActionType.DO_NOTHING) in allowed_actions
            else allowed_actions
        )
    action, _log_prob = agent.actor.get_action(state, nn_allowed, generator=agent._rng)
    return int(action)
