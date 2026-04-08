"""
environment.py — Core OpenEnv environment class
Implements reset() / step() / state() per the OpenEnv interface spec.
Internal state is mutable Python; all outputs are frozen Pydantic models.
"""

from __future__ import annotations
from typing import Any
from models import (
    Observation, Action, Reward, EpisodeResult,
    ActionType, AgentID, ConsensusState, TaskDifficulty,
    Message, RewardBreakdown,
)


# ---------------------------------------------------------------------------
# Legal action map — what actions are allowed in each consensus state
# ---------------------------------------------------------------------------

LEGAL_ACTIONS: dict[ConsensusState, list[ActionType]] = {
    ConsensusState.NONE: [
        ActionType.SHARE_INFORMATION,
        ActionType.PROPOSE_CONSENSUS,
        ActionType.REQUEST_CLARIFICATION,
        ActionType.CHALLENGE_PROPOSAL,
        ActionType.FLAG_BIAS,
    ],
    ConsensusState.PARTIAL: [
        ActionType.ACCEPT_CONSENSUS,
        ActionType.REJECT_CONSENSUS,
        ActionType.CHALLENGE_PROPOSAL,
        ActionType.SHARE_INFORMATION,
        ActionType.FLAG_BIAS,
    ],
    ConsensusState.REACHED: [],   # Episode over — no actions legal
    ConsensusState.FAILED:  [],   # Episode over — no actions legal
}


# ---------------------------------------------------------------------------
# NegotiationEnvironment
# ---------------------------------------------------------------------------

class NegotiationEnvironment:
    """
    Two-agent negotiation environment.

    Internal mutable state lives as plain Python attributes.
    All outputs (Observation, Reward, EpisodeResult) are frozen Pydantic models.
    """

    def __init__(self, task: dict[str, Any]) -> None:
        """
        Args:
            task: A task dict loaded from tasks.py. Must contain:
                  id, difficulty, description, max_turns,
                  private_information_a, private_information_b,
                  correct_answer (used by graders, not exposed to agents)
        """
        self._task = task
        self._validate_task(task)

        # Mutable episode state — reset on every reset() call
        self._current_turn:       int             = 0
        self._consensus_state:    ConsensusState  = ConsensusState.NONE
        self._pending_proposal:   str | None      = None
        self._conversation:       list[Message]   = []
        self._done:               bool            = False
        self._truncated:          bool            = False
        self._cumulative_reward:  float           = 0.0
        self._whose_turn:         AgentID         = AgentID.AGENT_A
        self._bias_flagged:       bool            = False
        self._bias_flag_action:   Action | None   = None
        self._curveball_injected: bool            = False

    # ------------------------------------------------------------------
    # OpenEnv interface — the three required methods
    # ------------------------------------------------------------------

    def reset(self) -> tuple[Observation, Observation]:
        """
        Start a fresh episode.
        Returns a tuple of (obs_for_agent_a, obs_for_agent_b).
        Each observation contains only that agent's private information.
        """
        self._current_turn      = 0
        self._consensus_state   = ConsensusState.NONE
        self._pending_proposal  = None
        self._conversation      = []
        self._done              = False
        self._truncated         = False
        self._cumulative_reward = 0.0
        self._whose_turn        = AgentID.AGENT_A
        self._bias_flagged      = False
        self._bias_flag_action  = None
        self._curveball_injected = False

        obs_a = self._build_observation(AgentID.AGENT_A)
        obs_b = self._build_observation(AgentID.AGENT_B)
        return obs_a, obs_b

    def step(self, action: Action) -> tuple[Observation, Observation, Reward]:
        """
        Agent submits an action. Environment advances one turn.

        Returns:
            obs_a      — updated observation for Agent A
            obs_b      — updated observation for Agent B
            reward     — step reward + cumulative + done signal

        Raises:
            ValueError — if action is illegal (wrong agent, invalid type, episode over)
        """
        self._validate_action(action)

        # Build the message and append to shared history
        message = Message(
            turn=self._current_turn,
            agent_id=action.agent_id,
            action_type=action.action_type,
            content=action.content,
            reasoning=action.reasoning,
        )
        # Snapshot state before action so rewards.py can compare before/after
        self._prev_state_snapshot = self.state()
        self._last_breakdown = None
        self._conversation.append(message)

        # Update consensus state machine
        self._advance_consensus(action)

        # Track bias flagging for Task 3 grader
        if action.action_type == ActionType.FLAG_BIAS and not self._bias_flagged:
            self._bias_flagged     = True
            self._bias_flag_action = action

        # Advance turn counter and rotate whose turn it is
        self._current_turn += 1
        self._whose_turn = (
            AgentID.AGENT_B
            if self._whose_turn == AgentID.AGENT_A
            else AgentID.AGENT_A
        )

        # Check hard turn limit AFTER advancing the counter
        if self._current_turn >= self._task["max_turns"]:
            if self._consensus_state not in (
                ConsensusState.REACHED, ConsensusState.FAILED
            ):
                self._consensus_state = ConsensusState.FAILED
                self._truncated       = True
                self._done            = True

        # Check natural episode termination
        if self._consensus_state in (ConsensusState.REACHED, ConsensusState.FAILED):
            self._done = True

        # Compute reward (rewards.py will replace this stub in Phase 2)
        step_reward = self._compute_step_reward(action)
        self._cumulative_reward += step_reward
        reward = self._build_reward(step_reward)

        obs_a = self._build_observation(AgentID.AGENT_A)
        obs_b = self._build_observation(AgentID.AGENT_B)
        return obs_a, obs_b, reward

    def state(self) -> dict[str, Any]:
        """
        Returns the full internal environment state at any point.
        Used for debugging, logging, and grader access.
        Note: includes correct_answer and private info for both agents —
        this is the god-view, not shown to agents.
        """
        return {
            "task_id":            self._task["id"],
            "task_difficulty":    self._task["difficulty"],
            "current_turn":       self._current_turn,
            "max_turns":          self._task["max_turns"],
            "whose_turn":         self._whose_turn.value,
            "consensus_state":    self._consensus_state.value,
            "pending_proposal":   self._pending_proposal,
            "conversation":       [m.model_dump() for m in self._conversation],
            "done":               self._done,
            "truncated":          self._truncated,
            "cumulative_reward":  self._cumulative_reward,
            "bias_flagged":       self._bias_flagged,
            "bias_flag_action":   self._bias_flag_action.model_dump()
                                  if self._bias_flag_action else None,
            "correct_answer":     self._task.get("correct_answer"),
            "private_info_a":     self._task["private_information_a"],
            "private_info_b":     self._task["private_information_b"],
            "curveball_injected": self._curveball_injected,
        }

    # ------------------------------------------------------------------
    # Internal helpers — not part of the public OpenEnv interface
    # ------------------------------------------------------------------

    def _build_observation(self, agent_id: AgentID) -> Observation:
        """Build a frozen Observation for one agent — only their private info included."""
        private_info = (
            self._task["private_information_a"]
            if agent_id == AgentID.AGENT_A
            else self._task["private_information_b"]
        )
        max_turns    = self._task["max_turns"]
        turn_warning = self._current_turn >= int(max_turns * 0.8)

        # --- Curveball injection ---
        # At the trigger turn, if consensus hasn't been reached yet, inject
        # new evidence into the task description for both agents to see.
        task_desc = self._task["description"]
        curveball = self._task.get("curveball")
        if (
            curveball
            and self._current_turn >= curveball.get("trigger_turn", 999)
            and self._consensus_state in (ConsensusState.NONE, ConsensusState.PARTIAL)
            and len(self._conversation) >= curveball.get("trigger_turn", 999)
        ):
            if not self._curveball_injected:
                self._curveball_injected = True
            task_desc = task_desc + "\n\n" + curveball["content"]

        return Observation(
            current_turn             = self._current_turn,
            max_turns                = max_turns,
            turn_warning             = turn_warning,
            agent_id                 = agent_id,
            private_information      = private_info,
            shared_conversation_history = list(self._conversation),
            task_description         = task_desc,
            task_id                  = self._task["id"],
            task_difficulty          = TaskDifficulty(self._task["difficulty"]),
            current_consensus_state  = self._consensus_state,
            pending_proposal         = self._pending_proposal,
            available_actions        = self._get_available_actions(),
        )

    def _get_available_actions(self) -> list[ActionType]:
        """Return legal actions for the current consensus state."""
        if self._done:
            return []
        return LEGAL_ACTIONS.get(self._consensus_state, [])

    def _advance_consensus(self, action: Action) -> None:
        """Update the consensus state machine based on the submitted action."""
        if action.action_type == ActionType.PROPOSE_CONSENSUS:
            self._consensus_state  = ConsensusState.PARTIAL
            self._pending_proposal = action.content

        elif action.action_type == ActionType.ACCEPT_CONSENSUS:
            # Only reaches REACHED if a proposal is actually on the table
            if self._consensus_state == ConsensusState.PARTIAL:
                self._consensus_state = ConsensusState.REACHED

        elif action.action_type == ActionType.REJECT_CONSENSUS:
            # Rejection resets — agents must restart negotiation
            self._consensus_state  = ConsensusState.NONE
            self._pending_proposal = None

        elif action.action_type == ActionType.CHALLENGE_PROPOSAL:
            # Challenge keeps proposal on table but signals disagreement
            self._consensus_state = ConsensusState.PARTIAL

    def _compute_step_reward(self, action: Action) -> float:
        """Delegates to rewards.py — single source of truth for all reward logic."""
        from rewards import compute_step_reward
        prev_state = self._prev_state_snapshot  # captured before action applied
        step_reward, breakdown = compute_step_reward(
            action.model_dump(), self.state(), prev_state
        )
        self._last_breakdown = breakdown
        return step_reward

    def _build_reward(self, step_reward: float) -> Reward:
        from models import RewardBreakdown
        breakdown = self._last_breakdown or RewardBreakdown()
        return Reward(
            step_reward       = step_reward,
            cumulative_reward = self._cumulative_reward,
            reward_breakdown  = breakdown,
            done              = self._done,
            truncated         = self._truncated,
            info              = {
                "turn":            self._current_turn,
                "consensus_state": self._consensus_state.value,
                "whose_turn_next": self._whose_turn.value,
                "bias_flagged":    self._bias_flagged,
                "curveball_injected": self._curveball_injected,
            },
        )

    def _validate_task(self, task: dict[str, Any]) -> None:
        """Fail loudly at construction time if the task dict is malformed."""
        required_keys = {
            "id", "difficulty", "description", "max_turns",
            "private_information_a", "private_information_b", "correct_answer",
        }
        missing = required_keys - task.keys()
        if missing:
            raise ValueError(f"Task dict is missing required keys: {missing}")
        if task["max_turns"] < 2:
            raise ValueError("max_turns must be at least 2")
        if task["difficulty"] not in ("easy", "medium", "hard"):
            raise ValueError(f"Invalid difficulty: {task['difficulty']}")

    def _validate_action(self, action: Action) -> None:
        """Reject illegal actions before they touch state."""
        if self._done:
            raise ValueError("Episode is over. Call reset() to start a new episode.")

        if action.agent_id != self._whose_turn:
            raise ValueError(
                f"It is {self._whose_turn.value}'s turn, "
                f"but action was submitted by {action.agent_id.value}."
            )

        legal = self._get_available_actions()
        if action.action_type not in legal:
            raise ValueError(
                f"Action '{action.action_type.value}' is not legal in "
                f"consensus state '{self._consensus_state.value}'. "
                f"Legal actions: {[a.value for a in legal]}"
            )