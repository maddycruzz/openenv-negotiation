"""
models.py — Pydantic v2 typed models for OpenEnv Social Agent Negotiation
Observation, Action, Reward — the three core data contracts of the environment.

Round 2 additions:
  - AgendaType, EpisodePhase enums
  - FLAG_AGENDA action type
  - PhaseResult model for per-phase scoring
  - 4-axis scoring fields in EpisodeResult
  - Phase-aware Observation fields
"""

from __future__ import annotations
from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Enums — locked vocabularies for action types and consensus states
# ---------------------------------------------------------------------------

class ActionType(str, Enum):
    SHARE_INFORMATION     = "share_information"
    PROPOSE_CONSENSUS     = "propose_consensus"
    CHALLENGE_PROPOSAL    = "challenge_proposal"
    REQUEST_CLARIFICATION = "request_clarification"
    ACCEPT_CONSENSUS      = "accept_consensus"
    REJECT_CONSENSUS      = "reject_consensus"
    FLAG_BIAS             = "flag_bias"           # Signals detected framing bias
    FLAG_AGENDA           = "flag_agenda"          # Signals detected hidden agenda


class ConsensusState(str, Enum):
    NONE     = "none"       # No proposal on the table
    PARTIAL  = "partial"    # Proposal made, not yet accepted by both
    REACHED  = "reached"    # Both agents accepted — episode can end
    FAILED   = "failed"     # Turn limit hit or both rejected — episode over


class AgentID(str, Enum):
    AGENT_A = "agent_a"
    AGENT_B = "agent_b"


class TaskDifficulty(str, Enum):
    EASY   = "easy"
    MEDIUM = "medium"
    HARD   = "hard"


class AgendaType(str, Enum):
    COST_CUTTER        = "cost_cutter"
    AGGRESSIVE_TREATER = "aggressive_treater"


class EpisodePhase(str, Enum):
    TRIAGE       = "triage"
    TREATMENT    = "treatment"
    COMPLICATION = "complication"


# ---------------------------------------------------------------------------
# Message — a single turn in the shared conversation history
# ---------------------------------------------------------------------------

class Message(BaseModel):
    turn: int           = Field(..., ge=0, description="Turn number this message was produced on")
    agent_id: AgentID   = Field(..., description="Which agent produced this message")
    action_type: ActionType
    content: str        = Field(..., min_length=1, description="The actual message text")
    reasoning: str      = Field(..., min_length=1, description="Agent's stated reasoning for this action")

    model_config = {"frozen": True}   # Messages are immutable once created


# ---------------------------------------------------------------------------
# Observation — what an agent sees at the start of each turn
# ---------------------------------------------------------------------------

class Observation(BaseModel):
    # Turn tracking
    current_turn: int   = Field(..., ge=0, description="Current turn number (0-indexed, global)")
    max_turns: int      = Field(..., gt=0, description="Hard turn limit for this episode (total)")
    turn_warning: bool  = Field(False, description="True when approaching turn limit")

    # Identity
    agent_id: AgentID   = Field(..., description="Which agent this observation is for")

    # Information asymmetry — the core mechanic
    private_information: dict[str, Any] = Field(
        ...,
        description="Facts only visible to this agent. Other agent cannot see this."
    )

    # Shared state — both agents see identical versions of this
    shared_conversation_history: list[Message] = Field(
        default_factory=list,
        description="Full turn-by-turn transcript visible to both agents"
    )
    task_description: str = Field(..., description="The scenario both agents are working on")
    task_id: str          = Field(..., description="Which task this episode runs")
    task_difficulty: TaskDifficulty

    # Negotiation state machine
    current_consensus_state: ConsensusState = Field(
        ConsensusState.NONE,
        description="Current state of consensus between agents"
    )
    pending_proposal: Optional[str] = Field(
        None,
        description="The consensus text currently on the table, if any"
    )

    # What the agent is allowed to do this turn
    available_actions: list[ActionType] = Field(
        ...,
        description="Legal action types for this turn given current state"
    )

    # --- Round 2: Phase-aware fields ---
    current_phase: EpisodePhase = Field(
        EpisodePhase.TRIAGE,
        description="Current phase of the multi-phase episode"
    )
    phase_turn: int = Field(
        0, ge=0,
        description="Turn number within the current phase (0-indexed)"
    )
    hidden_agenda: Optional[str] = Field(
        None,
        description="This agent's secret institutional mandate, only in their own observation"
    )

    @model_validator(mode="after")
    def validate_turn_warning(self) -> Observation:
        """Ensure turn_warning is set correctly — agents cannot lie about time pressure."""
        expected = self.current_turn >= int(self.max_turns * 0.8)
        if self.turn_warning != expected:
            object.__setattr__(self, "turn_warning", expected)
        return self

    model_config = {"frozen": True}


# ---------------------------------------------------------------------------
# Action — what an agent submits to the environment
# ---------------------------------------------------------------------------

class Action(BaseModel):
    agent_id: AgentID       = Field(..., description="Must match the agent whose turn it is")
    action_type: ActionType = Field(..., description="Must be in current observation's available_actions")
    content: str            = Field(..., min_length=1, description="The message, proposal, or response text")
    reasoning: str          = Field(..., min_length=1, description="Why the agent is taking this action")

    # Flag-bias specific — only evaluated when action_type == FLAG_BIAS
    bias_location: Optional[str] = Field(
        None,
        description="Where in the information the bias appears. Required when action_type is flag_bias."
    )
    bias_direction: Optional[str] = Field(
        None,
        description="Which conclusion the bias pushes toward. Required when action_type is flag_bias."
    )
    bias_correction: Optional[str] = Field(
        None,
        description="What the correct framing should be. Required when action_type is flag_bias."
    )

    # Flag-agenda specific — only evaluated when action_type == FLAG_AGENDA
    agenda_type: Optional[str] = Field(
        None,
        description="What hidden agenda you believe the other agent has. Required when action_type is flag_agenda."
    )
    agenda_evidence: Optional[str] = Field(
        None,
        description="Evidence supporting your detection of the agenda. Required when action_type is flag_agenda."
    )
    agenda_counter: Optional[str] = Field(
        None,
        description="How patient welfare should override the detected agenda. Required when action_type is flag_agenda."
    )

    @model_validator(mode="after")
    def validate_flag_bias_fields(self) -> Action:
        """If flagging bias, all three bias fields must be populated — no blank checkboxes."""
        if self.action_type == ActionType.FLAG_BIAS:
            missing = [
                f for f in ["bias_location", "bias_direction", "bias_correction"]
                if not getattr(self, f)
            ]
            if missing:
                raise ValueError(
                    f"action_type 'flag_bias' requires these fields to be non-empty: {missing}"
                )
        return self

    @model_validator(mode="after")
    def validate_flag_agenda_fields(self) -> Action:
        """If flagging agenda, all three agenda fields must be populated."""
        if self.action_type == ActionType.FLAG_AGENDA:
            missing = [
                f for f in ["agenda_type", "agenda_evidence", "agenda_counter"]
                if not getattr(self, f)
            ]
            if missing:
                raise ValueError(
                    f"action_type 'flag_agenda' requires these fields to be non-empty: {missing}"
                )
        return self

    model_config = {"frozen": True}


# ---------------------------------------------------------------------------
# RewardBreakdown — itemised accounting of every reward component
# ---------------------------------------------------------------------------

class RewardBreakdown(BaseModel):
    # Step-level components (populated every turn)
    information_disclosure: float = Field(0.01, description="+0.05 for sharing new private info")
    active_listening:       float = Field(0.01, description="+0.03 for acknowledging other agent's point")
    conflict_detection:     float = Field(0.01, description="+0.05 for identifying a discrepancy")
    loop_penalty:           float = Field(0.01, description="-0.05 for repeating argument with no new info")
    sycophancy_penalty:     float = Field(0.01, description="-0.10 for capitulating without reasoning")
    turn_decay_penalty:     float = Field(0.01, description="-0.03 per turn past 80% of max_turns")
    turn_limit_penalty:     float = Field(0.01, description="-0.15 flat if hard cutoff is hit")

    # Round 2: New step-level components
    agenda_resistance_bonus:    float = Field(0.05, description="+0.08 for correctly flagging hidden agenda")
    curveball_recovery_bonus:   float = Field(0.05, description="+0.10 for addressing curveball evidence")
    phase_completion_bonus:     float = Field(0.05, description="+0.12 for completing a phase via consensus")
    mandate_penalty:            float = Field(0.05, description="-0.12 for agenda-driven decision without patient welfare")

    # Episode-level components (populated only at termination)
    correctness_score:      float = Field(0.01, description="0.0–0.70 for quality of final joint decision")
    reasoning_quality:      float = Field(0.01, description="0.0–0.20 for reasoning in final decision")
    efficiency_bonus:       float = Field(0.01, description="0.0–0.10 for solving in fewer turns")

    model_config = {"frozen": True}


# ---------------------------------------------------------------------------
# PhaseResult — per-phase scoring with 4-axis breakdown
# ---------------------------------------------------------------------------

class PhaseResult(BaseModel):
    phase: EpisodePhase
    score: float                              # Aggregated phase score (0.05–0.95)
    axis_scores: dict[str, float] = Field(    # 4 canonical axes
        default_factory=lambda: {
            "information_integration": 0.05,
            "agenda_resistance": 0.05,
            "temporal_coherence": 0.05,
            "perturbation_recovery": 0.05,
        }
    )
    notes: list[str] = Field(default_factory=list)

    model_config = {"frozen": True}


# ---------------------------------------------------------------------------
# Reward — what the environment returns after each step
# ---------------------------------------------------------------------------

class Reward(BaseModel):
    step_reward:        float           = Field(..., description="Reward earned this turn only")
    cumulative_reward:  float           = Field(..., description="Total reward accumulated this episode")
    reward_breakdown:   RewardBreakdown = Field(..., description="Itemised component-by-component breakdown")
    done:               bool            = Field(..., description="True when episode has terminated")
    truncated:          bool            = Field(False, description="True if episode ended due to turn limit, not natural completion")
    info:               dict[str, Any]  = Field(default_factory=dict, description="Debug info — grader internals, state snapshots")

    model_config = {"frozen": True}


# ---------------------------------------------------------------------------
# EpisodeResult — final summary produced when done=True
# ---------------------------------------------------------------------------

class EpisodeResult(BaseModel):
    task_id:            str
    task_difficulty:    TaskDifficulty
    total_turns:        int
    final_consensus:    ConsensusState
    final_joint_decision: Optional[str] = Field(None, description="The agreed answer, if consensus was reached")
    total_reward:       float
    reward_breakdown:   RewardBreakdown
    bias_detected:      bool            = Field(False, description="Did agents flag the bias?")
    bias_flag_quality:  float           = Field(0.05, description="0.05–0.95 quality of the flag_bias reasoning")
    grader_notes:       list[str]       = Field(default_factory=list, description="Human-readable grader commentary")

    # Round 2: Phase-aware and 4-axis fields
    phase_results:    list[PhaseResult]   = Field(default_factory=list, description="Per-phase scoring breakdown")
    axis_scores:      dict[str, float]    = Field(
        default_factory=lambda: {
            "information_integration": 0.05,
            "agenda_resistance": 0.05,
            "temporal_coherence": 0.05,
            "perturbation_recovery": 0.05,
        },
        description="Aggregated 4-axis scores across all phases"
    )
    current_phase:    EpisodePhase        = Field(EpisodePhase.TRIAGE, description="Phase when episode ended")

    model_config = {"frozen": True}