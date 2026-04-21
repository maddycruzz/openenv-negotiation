"""
environment.py — Core OpenEnv environment class
Implements reset() / step() / state() per the OpenEnv interface spec.

Round 2: 3-phase episode structure (Triage → Treatment → Complication),
hidden agenda injection, dynamic curveball injection, per-phase consensus tracking.
"""

from __future__ import annotations
import random
from typing import Any, Optional
from models import (
    Observation, Action, Reward, EpisodeResult,
    ActionType, AgentID, ConsensusState, TaskDifficulty,
    Message, RewardBreakdown, EpisodePhase, AgendaType,
)


# ---------------------------------------------------------------------------
# Phase order — canonical sequence
# ---------------------------------------------------------------------------

PHASE_ORDER = [EpisodePhase.TRIAGE, EpisodePhase.TREATMENT, EpisodePhase.COMPLICATION]


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
        ActionType.FLAG_AGENDA,
    ],
    ConsensusState.PARTIAL: [
        ActionType.ACCEPT_CONSENSUS,
        ActionType.REJECT_CONSENSUS,
        ActionType.CHALLENGE_PROPOSAL,
        ActionType.SHARE_INFORMATION,
        ActionType.FLAG_BIAS,
        ActionType.FLAG_AGENDA,
    ],
    ConsensusState.REACHED: [],   # Phase over — transition or end
    ConsensusState.FAILED:  [],   # Episode over — no actions legal
}


# ---------------------------------------------------------------------------
# NegotiationEnvironment
# ---------------------------------------------------------------------------

class NegotiationEnvironment:
    """
    Two-agent negotiation environment with 3-phase episodes.

    Internal mutable state lives as plain Python attributes.
    All outputs (Observation, Reward, EpisodeResult) are frozen Pydantic models.
    """

    def __init__(self, task: dict[str, Any]) -> None:
        self._task = task
        self._validate_task(task)

        # Phase definitions from task
        self._phases: list[dict] = task["phases"]
        self._agendas: dict = task.get("agendas", {})

        # Mutable episode state — reset on every reset() call
        self._current_turn:       int             = 0
        self._current_phase_idx:  int             = 0
        self._phase_turn:         int             = 0
        self._phase_start_turn:   int             = 0
        self._phase_start_turns:  list[int]       = [0]
        self._consensus_state:    ConsensusState  = ConsensusState.NONE
        self._pending_proposal:   Optional[str]   = None
        self._conversation:       list[Message]   = []
        self._done:               bool            = False
        self._truncated:          bool            = False
        self._cumulative_reward:  float           = 0.0
        self._whose_turn:         AgentID         = AgentID.AGENT_A
        self._bias_flagged:       bool            = False
        self._bias_flag_action:   Optional[Action]= None
        self._agenda_flagged:     bool            = False
        self._agenda_flag_action: Optional[Action]= None
        self._curveball_injected: bool            = False
        self._curveball_awarded:  bool            = False
        self._phase_decisions:    list[str]       = []
        self._phase_completed:    list[dict]      = []  # Completed phase metadata

        # Hidden agenda assignments
        self._agent_a_agenda_type: str = ""
        self._agent_b_agenda_type: str = ""
        self._agent_a_agenda_text: str = ""
        self._agent_b_agenda_text: str = ""

        # Failure logging for curriculum
        self._failure_log: list[dict] = []

        # Compute total max turns
        self._total_max_turns: int = sum(p["max_turns"] for p in self._phases)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def _current_phase(self) -> dict:
        """Current phase definition dict."""
        return self._phases[self._current_phase_idx]

    @property
    def _current_phase_enum(self) -> EpisodePhase:
        """Current phase as enum."""
        return EpisodePhase(self._current_phase["phase"])

    @property
    def _current_phase_max_turns(self) -> int:
        """Max turns for current phase."""
        return self._current_phase["max_turns"]

    # ------------------------------------------------------------------
    # OpenEnv interface — the three required methods
    # ------------------------------------------------------------------

    def reset(self) -> tuple[Observation, Observation]:
        """
        Start a fresh episode.
        Returns a tuple of (obs_for_agent_a, obs_for_agent_b).
        """
        self._current_turn      = 0
        self._current_phase_idx = 0
        self._phase_turn        = 0
        self._phase_start_turn  = 0
        self._phase_start_turns = [0]
        self._consensus_state   = ConsensusState.NONE
        self._pending_proposal  = None
        self._conversation      = []
        self._done              = False
        self._truncated         = False
        self._cumulative_reward = 0.0
        self._whose_turn        = AgentID.AGENT_A
        self._bias_flagged      = False
        self._bias_flag_action  = None
        self._agenda_flagged    = False
        self._agenda_flag_action = None
        self._curveball_injected = False
        self._curveball_awarded  = False
        self._phase_decisions   = []
        self._phase_completed   = []

        # Assign hidden agendas — Agent A and Agent B get opposite agendas
        agenda_keys = list(self._agendas.keys())
        if len(agenda_keys) >= 2:
            if random.random() < 0.5:
                self._agent_a_agenda_type = agenda_keys[0]
                self._agent_b_agenda_type = agenda_keys[1]
            else:
                self._agent_a_agenda_type = agenda_keys[1]
                self._agent_b_agenda_type = agenda_keys[0]
            self._agent_a_agenda_text = self._agendas[self._agent_a_agenda_type]
            self._agent_b_agenda_text = self._agendas[self._agent_b_agenda_type]
        else:
            self._agent_a_agenda_type = ""
            self._agent_b_agenda_type = ""
            self._agent_a_agenda_text = ""
            self._agent_b_agenda_text = ""

        # Recompute total max turns
        self._total_max_turns = sum(p["max_turns"] for p in self._phases)

        obs_a = self._build_observation(AgentID.AGENT_A)
        obs_b = self._build_observation(AgentID.AGENT_B)
        return obs_a, obs_b

    def step(self, action: Action) -> tuple[Observation, Observation, Reward]:
        """
        Agent submits an action. Environment advances one turn.
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

        # Track bias flagging
        if action.action_type == ActionType.FLAG_BIAS and not self._bias_flagged:
            self._bias_flagged     = True
            self._bias_flag_action = action

        # Track agenda flagging
        if action.action_type == ActionType.FLAG_AGENDA and not self._agenda_flagged:
            self._agenda_flagged     = True
            self._agenda_flag_action = action

        # Advance turn counter and rotate whose turn it is
        self._current_turn += 1
        self._phase_turn   += 1
        self._whose_turn = (
            AgentID.AGENT_B
            if self._whose_turn == AgentID.AGENT_A
            else AgentID.AGENT_A
        )

        # --- Phase transition logic ---
        phase_just_advanced = False
        if self._consensus_state == ConsensusState.REACHED:
            # Record the phase decision
            self._phase_decisions.append(self._pending_proposal or "")
            self._phase_completed.append({
                "phase": self._current_phase["phase"],
                "phase_idx": self._current_phase_idx,
                "start_turn": self._phase_start_turn,
                "end_turn": self._current_turn,
                "decision": self._pending_proposal or "",
            })

            if self._current_phase_idx < len(self._phases) - 1:
                # Advance to next phase
                self._current_phase_idx += 1
                self._phase_turn = 0
                self._phase_start_turn = self._current_turn
                self._phase_start_turns.append(self._current_turn)
                self._consensus_state = ConsensusState.NONE
                self._pending_proposal = None
                phase_just_advanced = True
            else:
                # All phases complete — episode ends successfully
                self._done = True

        # Check phase turn limit (fail if current phase exceeded)
        if (
            not self._done
            and self._phase_turn >= self._current_phase_max_turns
            and self._consensus_state not in (ConsensusState.REACHED,)
        ):
            self._consensus_state = ConsensusState.FAILED
            self._truncated = True
            self._done = True

        # Check total turn limit
        if (
            not self._done
            and self._current_turn >= self._total_max_turns
        ):
            if self._consensus_state not in (ConsensusState.REACHED, ConsensusState.FAILED):
                self._consensus_state = ConsensusState.FAILED
                self._truncated = True
                self._done = True

        # Curveball injection check — happens in COMPLICATION phase
        curveball = self._current_phase.get("curveball")
        if (
            curveball
            and not self._curveball_injected
            and self._current_phase_enum == EpisodePhase.COMPLICATION
            and self._phase_turn >= curveball.get("trigger_turn", 999)
            and self._consensus_state in (ConsensusState.NONE, ConsensusState.PARTIAL)
        ):
            self._curveball_injected = True

        # Natural episode termination
        if self._consensus_state == ConsensusState.FAILED:
            self._done = True

        # Compute reward
        step_reward = self._compute_step_reward(action)
        self._cumulative_reward += step_reward
        reward = self._build_reward(step_reward, phase_just_advanced)

        obs_a = self._build_observation(AgentID.AGENT_A)
        obs_b = self._build_observation(AgentID.AGENT_B)
        return obs_a, obs_b, reward

    def state(self) -> dict[str, Any]:
        """
        Returns the full internal environment state at any point.
        God-view — includes correct answers, private info, agendas.
        """
        return {
            "task_id":            self._task["id"],
            "task_difficulty":    self._task["difficulty"],
            "current_turn":       self._current_turn,
            "max_turns":          self._total_max_turns,
            "current_phase":      self._current_phase_enum.value,
            "current_phase_idx":  self._current_phase_idx,
            "phase_turn":         self._phase_turn,
            "phase_start_turns":  list(self._phase_start_turns),
            "phases":             self._phases,
            "phase_decisions":    list(self._phase_decisions),
            "phase_completed":    list(self._phase_completed),
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
            "agenda_flagged":     self._agenda_flagged,
            "agenda_flag_action": self._agenda_flag_action.model_dump()
                                  if self._agenda_flag_action else None,
            "curveball_injected": self._curveball_injected,
            "correct_answer":     self._current_phase.get("correct_answer"),
            "private_info_a":     self._current_phase["private_information_a"],
            "private_info_b":     self._current_phase["private_information_b"],
            "agent_a_agenda":     self._agent_a_agenda_type,
            "agent_b_agenda":     self._agent_b_agenda_type,
            # Task-level fields for grader access
            "bias_detection_criteria": self._task.get("bias_detection_criteria", {}),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_observation(self, agent_id: AgentID) -> Observation:
        """Build a frozen Observation for one agent — only their private info included."""
        phase = self._current_phase
        private_info = dict(
            phase["private_information_a"]
            if agent_id == AgentID.AGENT_A
            else phase["private_information_b"]
        )

        # Inject hidden agenda into private information
        if agent_id == AgentID.AGENT_A and self._agent_a_agenda_text:
            private_info["institutional_mandate"] = self._agent_a_agenda_text
        elif agent_id == AgentID.AGENT_B and self._agent_b_agenda_text:
            private_info["institutional_mandate"] = self._agent_b_agenda_text

        # Get hidden agenda text for this agent
        hidden_agenda = (
            self._agent_a_agenda_text if agent_id == AgentID.AGENT_A
            else self._agent_b_agenda_text
        ) or None

        # Build task description — phase description + curveball if injected
        task_desc = phase["description"]
        curveball = phase.get("curveball")
        if (
            self._curveball_injected
            and curveball
            and self._current_phase_enum == EpisodePhase.COMPLICATION
        ):
            task_desc = task_desc + "\n\n" + curveball["content"]

        turn_warning = self._current_turn >= int(self._total_max_turns * 0.8)

        return Observation(
            current_turn             = self._current_turn,
            max_turns                = self._total_max_turns,
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
            current_phase            = self._current_phase_enum,
            phase_turn               = self._phase_turn,
            hidden_agenda            = hidden_agenda,
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
            if self._consensus_state == ConsensusState.PARTIAL:
                self._consensus_state = ConsensusState.REACHED

        elif action.action_type == ActionType.REJECT_CONSENSUS:
            self._consensus_state  = ConsensusState.NONE
            self._pending_proposal = None

        elif action.action_type == ActionType.CHALLENGE_PROPOSAL:
            self._consensus_state = ConsensusState.PARTIAL

    def _compute_step_reward(self, action: Action) -> float:
        """Delegates to rewards.py — single source of truth for all reward logic."""
        from rewards import compute_step_reward
        prev_state = self._prev_state_snapshot
        step_reward, breakdown = compute_step_reward(
            action.model_dump(), self.state(), prev_state
        )
        self._last_breakdown = breakdown
        return step_reward

    def _build_reward(self, step_reward: float, phase_just_advanced: bool = False) -> Reward:
        from models import RewardBreakdown
        breakdown = self._last_breakdown or RewardBreakdown()
        return Reward(
            step_reward       = step_reward,
            cumulative_reward = self._cumulative_reward,
            reward_breakdown  = breakdown,
            done              = self._done,
            truncated         = self._truncated,
            info              = {
                "turn":              self._current_turn,
                "consensus_state":   self._consensus_state.value,
                "whose_turn_next":   self._whose_turn.value,
                "bias_flagged":      self._bias_flagged,
                "agenda_flagged":    self._agenda_flagged,
                "curveball_injected": self._curveball_injected,
                "current_phase":     self._current_phase_enum.value,
                "phase_turn":        self._phase_turn,
                "phase_just_advanced": phase_just_advanced,
            },
        )

    def _validate_task(self, task: dict[str, Any]) -> None:
        """Fail loudly at construction time if the task dict is malformed."""
        required_keys = {"id", "difficulty", "description", "phases"}
        missing = required_keys - task.keys()
        if missing:
            raise ValueError(f"Task dict is missing required keys: {missing}")
        if task["difficulty"] not in ("easy", "medium", "hard"):
            raise ValueError(f"Invalid difficulty: {task['difficulty']}")
        if not isinstance(task["phases"], list) or len(task["phases"]) == 0:
            raise ValueError("Task must have at least one phase in 'phases' list")
        for i, phase in enumerate(task["phases"]):
            phase_required = {"phase", "description", "max_turns",
                              "private_information_a", "private_information_b",
                              "correct_answer"}
            phase_missing = phase_required - phase.keys()
            if phase_missing:
                raise ValueError(f"Phase {i} is missing required keys: {phase_missing}")
            if phase["max_turns"] < 2:
                raise ValueError(f"Phase {i}: max_turns must be at least 2")

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

    def log_failure(self, episode_id: str, axis_scores: dict, failure_categories: list) -> None:
        """Append a failure log entry for curriculum tracking."""
        self._failure_log.append({
            "episode_id": episode_id,
            "axis_scores": dict(axis_scores),
            "failure_categories": list(failure_categories),
        })

    def get_failure_log(self) -> list[dict]:
        """Return the failure log for curriculum manager consumption."""
        return list(self._failure_log)