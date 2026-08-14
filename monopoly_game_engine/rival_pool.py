"""Registry of verified competitor-team agents used as training opponents.

Only EXPO is registered here. A second candidate (a closed-form net-worth
heuristic) was dropped after review: its original repo named the class
``ASU_SLAYER``, and even though the algorithm itself had no ASU import or
ASU-derived logic, a rename-and-scrub was judged too close to the banned
ASU association to risk — see this session's decision. EXPO's action-ID
space (``rival_agents``' vendored ``actions.py`` semantics) is byte-identical
to ours since both trace back to the same shared course template — confirmed
by diff before wiring this in.

Training-only: never imported by ``DeepRL_Monopoly/agent.py`` or anything
under ``tournament_submission/``.
"""

from rival_agents.expo_agent import ExpoHeuristicAgent

RIVAL_AGENT_CLASSES = [ExpoHeuristicAgent]
