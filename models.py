"""
models.py — Pydantic v2 typed models for OpenEnv Social Agent Negotiation
Observation, Action, Reward — the three core data contracts of the environment.
"""

from __future__ import annotations
from enum import Enum
from typing import Any
from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Enums — locked vocabularies for action types and consensus states
# ---------------------------------------------------------------------------

class ActionType(str, Enum):
    SHARE_INFORMATION    = "share_information"
    PROPOSE_CONSENSUS    = "propose_consensus"
    CHALLENGE_PROPOSAL   = "challenge_proposal"
    REQUEST_CLARIFICATION = "request_clarification"
    ACCEPT_CONSENSUS     = "accept_consensus"
    REJECT_CONSENSUS     = "reject_consensus"
    FLAG_BIAS            = "flag_bias"          # Task 3 only — signals detected bias


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
    current_turn: int   = Field(..., ge=0, description="Current turn number (0-indexed)")
    max_turns: int      = Field(..., gt=0, description="Hard turn limit for this episode")
    turn_warning: bool  = Field(False, description="True when current_turn >= 80% of max_turns")

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
    task_id: str          = Field(..., description="Which of the three tasks this episode runs")
    task_difficulty: TaskDifficulty

    # Negotiation state machine
    current_consensus_state: ConsensusState = Field(
        ConsensusState.NONE,
        description="Current state of consensus between agents"
    )
    pending_proposal: str | None = Field(
        None,
        description="The consensus text currently on the table, if any"
    )

    # What the agent is allowed to do this turn
    available_actions: list[ActionType] = Field(
        ...,
        description="Legal action types for this turn given current state"
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
    bias_location: str | None = Field(
        None,
        description="Where in the information the bias appears. Required when action_type is flag_bias."
    )
    bias_direction: str | None = Field(
        None,
        description="Which conclusion the bias pushes toward. Required when action_type is flag_bias."
    )
    bias_correction: str | None = Field(
        None,
        description="What the correct framing should be. Required when action_type is flag_bias."
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

    # Episode-level components (populated only at termination)
    correctness_score:      float = Field(0.01, description="0.0–0.70 for quality of final joint decision")
    reasoning_quality:      float = Field(0.01, description="0.0–0.20 for reasoning in final decision")
    efficiency_bonus:       float = Field(0.01, description="0.0–0.10 for solving in fewer turns")

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
    final_joint_decision: str | None    = Field(None, description="The agreed answer, if consensus was reached")
    total_reward:       float
    reward_breakdown:   RewardBreakdown
    bias_detected:      bool            = Field(False, description="Task 3 only — did agents flag the bias?")
    bias_flag_quality:  float           = Field(0.01, description="Task 3 only — 0.0–1.0 quality of the flag_bias reasoning")
    grader_notes:       list[str]       = Field(default_factory=list, description="Human-readable grader commentary")

    model_config = {"frozen": True}