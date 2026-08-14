# RL Φ Genişletmesi + Aksiyon Uzayı Daraltma Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bu gece Colab'da bir PPO-Plus hybrid eğitim koşusu başlatmak için: (1) `env._compute_reward`'ı ASU'dan tamamen bağımsız iki yeni terimle genişletmek, (2) hybrid PPO'nun öğrenilen aksiyon uzayından tüm trade ailesini çıkarıp kendi kural katmanımıza vermek, (3) ASU'yu (sadece rakip/benchmark olarak) değerlendirme aracına eklemek.

**Architecture:** Mevcut `MonopolyEnv._compute_reward` (relative net-worth, `[-1,1]`) korunuyor; üstüne monopoly-yakınlık ve nakit-güvenlik terimleri ağırlıklı toplanıyor. `PPOAgent`'in hybrid modu, mevcut BUY_PROPERTY/ACCEPT_TRADE müdahalesine ek olarak tüm buy/sell/exchange-trade aksiyon ailesini (2772 aksiyon) sinir ağının seçebileceği kümeden kalıcı olarak maskeliyor ve tekel-tamamlayan ucuz teklifleri kendi deterministik kuralımızla veriyor.

**Tech Stack:** Python, PyTorch, unittest (mevcut repo konvansiyonu — pytest değil, `unittest.TestCase`).

## Global Constraints

- ASU'nun kodu/çıktısı hiçbir reward-shaping veya davranış-klonlama sinyaline giremez (yarışma kuralı). ASU sadece bir rakip/benchmark olarak çağrılabilir.
- `ppo-plus-v2` ruleset kontratı (state_dim=300, action_dim=2958) değişmiyor — sadece reward hesaplama ve hybrid aksiyon maskeleme değişiyor.
- Testler `unittest` ile yazılır, `ROOT = Path(__file__).resolve().parents[1]` + `sys.path.insert` deseniyle (bkz. `tests/test_ppo_agent.py`).
- Mevcut `_compute_reward` davranışı (bankrupt→-1.0, tek oyuncu kalınca→1.0) korunmalı; sadece devam eden oyun adımlarındaki büyüklük değişir.

---

### Task 1: Φ genişletmesi — monopoly-yakınlık ve nakit-güvenlik terimleri

**Files:**
- Modify: `monopoly_game_engine/env.py:1047` (`_compute_reward` metodu ve çevresi)
- Test: Create `tests/test_reward_shaping.py`

**Interfaces:**
- Consumes: `MonopolyEnv.properties` (dict `square_id -> Property`), `MonopolyEnv.players` (list `Player`), `constants.COLOR_GROUPS` (dict `color -> List[int]`), `Player.net_worth()`, `Player.cash`, `Player.bankrupt`, `env.debt_player`, `env.debt_amount`.
- Produces: `MonopolyEnv._compute_reward(pid: int) -> float` (imzası ve dönüş aralığı `[-1,1]` değişmiyor — `train.py:potential_delta()` bunu değiştirmeden çağırmaya devam eder). Yeni yardımcı metodlar: `MonopolyEnv._color_group_progress(pid: int) -> float`, `MonopolyEnv._cash_safety_score(pid: int) -> float`, `MonopolyEnv._relative_score(own: float, others: List[float]) -> float` (staticmethod).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_reward_shaping.py`:

```python
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from monopoly_game_engine.env import MonopolyEnv  # noqa: E402


class RewardShapingTests(unittest.TestCase):
    def test_relative_score_clips_and_handles_empty_others(self) -> None:
        self.assertEqual(MonopolyEnv._relative_score(0.9, []), 0.0)
        self.assertEqual(MonopolyEnv._relative_score(5.0, [0.0]), 1.0)
        self.assertEqual(MonopolyEnv._relative_score(-5.0, [0.0]), -1.0)
        self.assertAlmostEqual(MonopolyEnv._relative_score(0.3, [0.1]), 0.2)

    def test_color_group_progress_rewards_completing_a_group(self) -> None:
        env = MonopolyEnv(agent_ids=[0], max_rounds=2)
        # Brown group is squares 1 and 3 (see constants.py).
        env.properties[1].owner = 0
        partial = env._color_group_progress(0)

        env.properties[3].owner = 0
        complete = env._color_group_progress(0)

        self.assertGreater(complete, partial)
        self.assertGreater(partial, 0.0)

    def test_color_group_progress_zero_when_nothing_owned(self) -> None:
        env = MonopolyEnv(agent_ids=[0], max_rounds=2)
        self.assertEqual(env._color_group_progress(0), 0.0)

    def test_cash_safety_score_penalizes_debt_regardless_of_cash(self) -> None:
        env = MonopolyEnv(agent_ids=[0], max_rounds=2)
        env.players[0].cash = 1000
        env.debt_player = 0
        env.debt_amount = 50
        self.assertEqual(env._cash_safety_score(0), -1.0)

    def test_cash_safety_score_scales_with_cash_when_no_debt(self) -> None:
        env = MonopolyEnv(agent_ids=[0], max_rounds=2)
        env.players[0].cash = 350
        self.assertEqual(env._cash_safety_score(0), 1.0)
        env.players[0].cash = 50
        self.assertEqual(env._cash_safety_score(0), 0.0)
        env.players[0].cash = 0
        self.assertLess(env._cash_safety_score(0), 0.0)

    def test_compute_reward_bounded_and_terminal_cases_unchanged(self) -> None:
        env = MonopolyEnv(agent_ids=[0], max_rounds=2)
        reward = env._compute_reward(0)
        self.assertGreaterEqual(reward, -1.0)
        self.assertLessEqual(reward, 1.0)

        env.players[0].bankrupt = True
        self.assertEqual(env._compute_reward(0), -1.0)

        env.players[0].bankrupt = False
        for p in env.players[1:]:
            p.bankrupt = True
        self.assertEqual(env._compute_reward(0), 1.0)

    def test_compute_reward_increases_when_owning_a_full_monopoly(self) -> None:
        env = MonopolyEnv(agent_ids=[0], max_rounds=2)
        baseline = env._compute_reward(0)

        env.properties[1].owner = 0
        env.properties[3].owner = 0
        with_monopoly = env._compute_reward(0)

        self.assertGreater(with_monopoly, baseline)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m unittest tests.test_reward_shaping -v`
Expected: FAIL — `AttributeError: 'MonopolyEnv' object has no attribute '_color_group_progress'` (and similar for the other new methods).

- [ ] **Step 3: Implement the extended `_compute_reward`**

In `monopoly_game_engine/env.py`, replace the existing `_compute_reward` method (currently at line 1047) with:

```python
    @staticmethod
    def _relative_score(own: float, others: List[float]) -> float:
        """Own score minus the mean of others, clipped to [-1, 1]."""
        if not others:
            return 0.0
        mean_other = float(np.mean(others))
        return float(np.clip(own - mean_other, -1.0, 1.0))

    def _color_group_progress(self, pid: int) -> float:
        """
        Mean squared ownership-fraction across every color group (including
        railroads/utilities). Squaring makes near-complete groups worth much
        more than scattered partial ownership. Independent formula from the
        ASU teacher's discounted-monopoly-value calculation — deliberately
        simpler (no dice projection, no development search).
        """
        total = 0.0
        for squares in COLOR_GROUPS.values():
            owned = sum(1 for s in squares if self.properties[s].owner == pid)
            total += (owned / len(squares)) ** 2
        return total / len(COLOR_GROUPS)

    def _cash_safety_score(self, pid: int) -> float:
        """
        -1.0 while in explicit debt; otherwise a linear cushion score capped
        at $350 of cash. Independent of ASU's two-gate safety formula.
        """
        if self.debt_player == pid and self.debt_amount > 0:
            return -1.0
        cash = self.players[pid].cash
        return float(np.clip((cash - 50) / 300.0, -1.0, 1.0))

    def _compute_reward(self, pid: int) -> float:
        """Bounded net-worth potential used for decision-to-decision shaping."""
        if self.players[pid].bankrupt:
            return -1.0
        active = [p for p in self.players if not p.bankrupt]
        if len(active) <= 1:
            return 1.0

        own_nw = self.players[pid].net_worth()
        others_nw = [
            player.net_worth() for player in active if player.player_id != pid
        ]
        mean_other_nw = float(np.mean(others_nw))
        net_worth_term = float(
            np.clip(
                (own_nw - mean_other_nw) / (abs(own_nw) + abs(mean_other_nw) + 1e-9),
                -1.0,
                1.0,
            )
        )

        monopoly_term = self._relative_score(
            self._color_group_progress(pid),
            [
                self._color_group_progress(p.player_id)
                for p in active
                if p.player_id != pid
            ],
        )
        cash_term = self._relative_score(
            self._cash_safety_score(pid),
            [
                self._cash_safety_score(p.player_id)
                for p in active
                if p.player_id != pid
            ],
        )

        return float(
            np.clip(
                0.6 * net_worth_term + 0.25 * monopoly_term + 0.15 * cash_term,
                -1.0,
                1.0,
            )
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m unittest tests.test_reward_shaping -v`
Expected: all 7 tests PASS.

- [ ] **Step 5: Run the existing PPO/DDQN/regression suites to check for regressions**

Run: `python -m unittest tests.test_ppo_agent tests.test_ddqn_agent tests.test_monopoly_regression -v`
Expected: all PASS (these tests use fakes for `_compute_reward` or don't assert exact reward values, so they should be unaffected — see prior audit of these files).

- [ ] **Step 6: Commit**

```bash
git add monopoly_game_engine/env.py tests/test_reward_shaping.py
git commit -m "feat: extend PBRS potential with monopoly-progress and cash-safety terms"
```

(Not a git repo yet in this workspace — if `git status` errors with "not a git repository", skip this step and note it to the user instead of running `git init` unprompted.)

---

### Task 2: Trade family excluded from the learned policy, own rule-based trade offers

**Files:**
- Modify: `monopoly_game_engine/agent_ppo.py` (top-level helper functions + `PPOAgent.__init__` + `PPOAgent.choose_action`)
- Test: Modify `tests/test_ppo_agent.py` (append new test methods to `PPOAgentTests`)

**Interfaces:**
- Consumes: `env._trade_offer_actions(pid: int) -> List[int]` (existing, already legality/affordability-filtered), `OFFSETS` dict (`monopoly_game_engine.actions`), `PROPERTY_IDS`, `TRADE_CASH_LEVELS` (`monopoly_game_engine.constants`, already imported in `agent_ppo.py`), `COLOR_GROUPS` (already imported).
- Produces: `fixed_trade_offer_decision(env, pid: int) -> Optional[int]` (new top-level function in `agent_ppo.py`, same module as `fixed_buy_decision`/`fixed_accept_trade_decision`). `PPOAgent.choose_action` behavior: when hybrid, `nn_allowed` never contains an action in `[OFFSETS["buy_trade"], OFFSETS["auction"])`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_ppo_agent.py` (inside `PPOAgentTests`, and add the new import at the top alongside the existing `agent_ppo` import):

```python
from monopoly_game_engine.agent_ppo import (  # noqa: E402
    PPOAgent,
    fixed_accept_trade_decision,
    fixed_trade_offer_decision,
)
from monopoly_game_engine.actions import OFFSETS  # noqa: E402
```

```python
    def test_fixed_trade_offer_decision_targets_monopoly_completion(self) -> None:
        env = MonopolyEnv(agent_ids=[0], max_rounds=2)
        # Brown group: squares 1 and 3. Player 0 owns square 1, player 1 owns 3.
        env.properties[1].owner = 0
        env.properties[3].owner = 1
        env.turn_order = [0, 1, 2, 3]

        action = fixed_trade_offer_decision(env, 0)

        self.assertIsNotNone(action)
        self.assertTrue(OFFSETS["buy_trade"] <= action < OFFSETS["sell_trade"])

    def test_fixed_trade_offer_decision_none_when_no_completion_available(self) -> None:
        env = MonopolyEnv(agent_ids=[0], max_rounds=2)
        env.turn_order = [0, 1, 2, 3]
        self.assertIsNone(fixed_trade_offer_decision(env, 0))

    def test_hybrid_agent_never_exposes_trade_family_to_the_actor(self) -> None:
        env = MonopolyEnv(agent_ids=[0], max_rounds=2)
        env.properties[1].owner = 0
        env.properties[3].owner = 1
        env.turn_order = [0, 1, 2, 3]
        env.current_turn_idx = 0
        env.phase = PHASE_POST_ROLL
        env.has_rolled = True
        agent = PPOAgent(player_id=0, hybrid=True, device="cpu")

        allowed = env.get_allowed_actions(0)
        action, log_prob, _, nn_allowed = agent.choose_action(
            env._get_state(0), env, allowed
        )

        # The completing buy-trade offer must fire through the rule layer,
        # not the actor (log_prob is None for fixed-policy decisions).
        self.assertTrue(OFFSETS["buy_trade"] <= action < OFFSETS["sell_trade"])
        self.assertIsNone(log_prob)
        for a in nn_allowed:
            self.assertFalse(OFFSETS["buy_trade"] <= a < OFFSETS["auction"])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m unittest tests.test_ppo_agent -v`
Expected: FAIL — `ImportError: cannot import name 'fixed_trade_offer_decision'`.

- [ ] **Step 3: Implement `fixed_trade_offer_decision` and wire it into the hybrid mask**

In `monopoly_game_engine/agent_ppo.py`, add this function right after `fixed_accept_trade_decision` (near line 93):

```python
def fixed_trade_offer_decision(env, pid: int) -> Optional[int]:
    """
    Rule-based trade initiation: only offer to buy a property when it would
    complete one of our own color groups, at the cheapest affordable price
    tier. Independent of ASU — uses only ownership counts already exposed by
    `env._trade_offer_actions`, which is the same legality/affordability
    filter the environment uses to build the legal-action mask.
    """
    if pid in env.pending_trades:
        return None

    stride = len(PROPERTY_IDS) * len(TRADE_CASH_LEVELS)
    best_price_idx: Optional[int] = None
    best_action: Optional[int] = None

    for action_id in env._trade_offer_actions(pid):
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

        if best_price_idx is None or price_idx < best_price_idx:
            best_price_idx = price_idx
            best_action = action_id

    return best_action
```

This needs `Optional` (already imported at the top of `agent_ppo.py`) and `PROPERTY_IDS`/`TRADE_CASH_LEVELS` (already imported from `.constants`) and `OFFSETS` (already imported from `.actions`) — no new imports required.

Now update `PPOAgent.__init__` (the `fixed_action_mask` block, near line 186) to mask the whole trade family whenever hybrid mode is on:

```python
        # Mask actions permanently handled by fixed policy (hybrid only)
        self.fixed_action_mask = torch.zeros(ACTION_SPACE_SIZE, dtype=torch.bool)
        if hybrid:
            self.fixed_action_mask[int(ActionType.BUY_PROPERTY)] = True
            self.fixed_action_mask[int(ActionType.ACCEPT_TRADE)] = True
            self.fixed_action_mask[OFFSETS["buy_trade"] : OFFSETS["auction"]] = True
```

Finally, update `PPOAgent.choose_action` to try the trade-offer rule right after the accept-trade block (near line 222, before the `nn_allowed` filtering):

```python
        # Hybrid: handle rule-based trade initiation (monopoly completion only)
        if self.hybrid:
            trade_action = fixed_trade_offer_decision(env, pid)
            if trade_action is not None and trade_action in allowed_actions:
                return trade_action, None, None, allowed_actions
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m unittest tests.test_ppo_agent -v`
Expected: all tests PASS, including the 3 new ones.

- [ ] **Step 5: Run the full test suite for regressions**

Run: `python -m unittest discover -s tests -v`
Expected: all PASS. Pay particular attention to `test_ddqn_agent.py` (DDQN is untouched by this task, should be unaffected) and `test_monopoly_regression.py`.

- [ ] **Step 6: Commit**

```bash
git add monopoly_game_engine/agent_ppo.py tests/test_ppo_agent.py
git commit -m "feat: exclude trade family from hybrid PPO actor, add rule-based monopoly-completion offers"
```

---

### Task 3: ASU as an evaluation-only opponent in `generate_stats.py`

**Files:**
- Modify: `tools/generate_stats.py` (opponent-construction section)
- Test: Create `tests/test_generate_stats_asu_opponent.py`

**Interfaces:**
- Consumes: `ASU_FROZEN_TEACHER.core.ASUValueV1(player_id: int)` — exposes `.player_id` and `.choose_action(env) -> int`, same shape as `FPAgentA`/`FPAgentB`/`FPAgentC`.
- Produces: `generate_stats.py --p1 asu` (and `--p2 asu`, `--p3 asu`) becomes a valid opponent selector, resolved by the same code path that already builds `FPAgentA(pid)` etc. This is eval-only wiring — ASU never touches training code (Tasks 1-2 are untouched by this task).

- [ ] **Step 1: Write the failing test**

Create `tests/test_generate_stats_asu_opponent.py`:

```python
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import generate_stats  # noqa: E402


class AsuOpponentTests(unittest.TestCase):
    def test_build_opponent_asu_returns_asu_value_v1(self) -> None:
        from ASU_FROZEN_TEACHER.core import ASUValueV1

        opponent = generate_stats.build_opponent("asu", player_id=2)

        self.assertIsInstance(opponent, ASUValueV1)
        self.assertEqual(opponent.player_id, 2)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m unittest tests.test_generate_stats_asu_opponent -v`
Expected: FAIL — either `AttributeError: module 'generate_stats' has no attribute 'build_opponent'` (if opponent construction is inline rather than a function — check the actual current code first) or a similar error confirming `"asu"` is not yet a recognized opponent type.

> **Before writing Step 3**, read `tools/generate_stats.py` in full to find the exact function/branch that turns a `--p1`/`--p2`/`--p3` string like `"fixed-a"` into an agent instance (it will look like a chain of `if opponent_type == "fixed-a": ... elif ...`). If it is not already a standalone function, extract it into one named `build_opponent(opponent_type: str, player_id: int)` so the test above can call it directly — keep every existing branch's behavior byte-for-byte identical, this is a pure refactor plus one new branch.

- [ ] **Step 3: Add the `"asu"` branch**

Add this branch to the opponent-construction logic (alongside the existing `"fixed-a"`, `"fixed-b"`, etc. branches), and add the import at the top of `tools/generate_stats.py` next to the other `monopoly_game_engine` imports:

```python
from ASU_FROZEN_TEACHER.core import ASUValueV1
```

```python
    if opponent_type == "asu":
        return ASUValueV1(player_id)
```

Also add one line to the module docstring's "Opponent types" list (near line 17-24):

```
  asu                            ASU_FROZEN_TEACHER value-based teacher (eval-only; never trains)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m unittest tests.test_generate_stats_asu_opponent -v`
Expected: PASS.

- [ ] **Step 5: Smoke-test a tiny real run**

Run: `python tools/generate_stats.py --model artifacts/ppo_plus/ppo_hybrid_model.pt --p1 asu --p2 fixed-b --p3 fixed-c --games 2`

Expected: completes without a stack trace and writes a stats JSON (if no checkpoint exists yet at that path, use `--model` pointing at any existing `.pt` under `artifacts/`, or skip this step until after Task 4 produces one — note which you did).

- [ ] **Step 6: Commit**

```bash
git add tools/generate_stats.py tests/test_generate_stats_asu_opponent.py
git commit -m "feat: add ASU as an eval-only opponent type in generate_stats.py"
```

---

### Task 4: Kick off the Colab training run

**Files:**
- None created/modified — this task runs the existing `tools/train_and_save.py` (already supports `--device cuda`, `--checkpoint-every`, `--resume`, and a memory watchdog per `REPO_STUDY_NOTES.md` §11).

**Interfaces:**
- Consumes: `tools/train_and_save.py` CLI (Tasks 1-2 changes take effect automatically — `train_ppo` calls into `env._compute_reward` and `PPOAgent` with no signature changes).
- Produces: a checkpoint at `artifacts/ppo_plus/ppo_hybrid_model.pt` and a history file `artifacts/ppo_plus/ppo_hybrid_model_history.json`.

- [ ] **Step 1: Local dry run (fast sanity check before spending Colab time)**

Run: `python tools/train_and_save.py --algo ppo --games 10 --checkpoint-every 5 --device cpu`
Expected: completes in well under a minute, prints a win-rate line, saves a checkpoint. This exists purely to catch an import/runtime error from Tasks 1-2 before burning Colab minutes on it.

- [ ] **Step 2: Upload/clone the repo to Colab and install dependencies**

In a Colab cell:

```python
!git clone <this-repo-url> /content/monopoly
%cd /content/monopoly/DeepRL_Monopoly
!pip install -q torch numpy
```

(If the repo isn't pushed anywhere yet, `!git init` is not appropriate to run unprompted — instead upload the `DeepRL_Monopoly` folder as a zip via the Colab file browser and `!unzip`.)

- [ ] **Step 3: Confirm GPU and start the real run**

```python
import torch
assert torch.cuda.is_available(), "Runtime > Change runtime type > GPU"
```

```python
!python tools/train_and_save.py --algo ppo --games 5000 --checkpoint-every 100 --device cuda --out artifacts/ppo_plus/ppo_hybrid_model.pt
```

- [ ] **Step 4: Verify checkpoint integrity after the run (or after a Colab disconnect)**

```python
!python -c "import torch; ckpt = torch.load('artifacts/ppo_plus/ppo_hybrid_model.pt', weights_only=True); print(ckpt['games_trained'], ckpt['format_version'])"
```

Expected: prints the games-trained count and `format_version` 3, confirming the checkpoint is loadable. If the session disconnected mid-run, re-run the Step 3 command with `--resume` appended — `train_and_save.py` already merges training history across resumes (see `merge_training_history` in that file).

- [ ] **Step 5: Evaluate**

```python
!python tools/generate_stats.py --model artifacts/ppo_plus/ppo_hybrid_model.pt --p1 fixed-a --p2 fixed-b --p3 fixed-c --games 500 --out results/tonight_vs_fixed
!python tools/generate_stats.py --model artifacts/ppo_plus/ppo_hybrid_model.pt --p1 asu --p2 fixed-b --p3 fixed-c --games 500 --out results/tonight_vs_asu
```

- [ ] **Step 6: Download results and note follow-ups**

Download `artifacts/ppo_plus/ppo_hybrid_model.pt`, its `_history.json`, and both `results/tonight_*` JSON files back to the local machine. Read the win-rate trend in the history file and the final stats JSONs; use them to decide which `ARCHITECTURE.md` P1/P2 item (Φ weight tuning, board-landing calibration, opponent-pool diversity) to tackle next — this decision is out of scope for this plan.

---

## Self-Review Notes

- **Spec coverage:** Design doc §1 → Task 1. §2 → Task 2. §3 (Colab training) → Task 4. §4 (evaluation incl. ASU) → Tasks 3-4.
- **Placeholder scan:** no TBD/TODO; every step has literal runnable code or an exact command.
- **Type consistency:** `fixed_trade_offer_decision(env, pid: int) -> Optional[int]` matches its one call site in `choose_action`; `_relative_score`, `_color_group_progress`, `_cash_safety_score` signatures match between definition (Task 1) and test usage.
- **Scope:** Task 3 depends on reading the real current branching structure of `generate_stats.py` before writing the exact diff (flagged explicitly in Task 3 Step 2) since its precise current shape wasn't fully read during planning — this is the one task where the implementer must look before leaping; every other task's code was written against file contents read during planning.
