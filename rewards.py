"""
rewards.py — Step-level and episode-level reward logic.

Two public entry points:
  compute_step_reward(action, state, prev_state)  → float, RewardBreakdown
  compute_episode_reward(grader_result, state)    → float, RewardBreakdown

Both return a (float, RewardBreakdown) tuple.
environment.py calls these and assembles the final Reward model.

DESIGN RULE: No LLM calls. No randomness. Same inputs → same outputs always.
"""

from __future__ import annotations
import re
from models import ActionType, AgentID, RewardBreakdown
from graders import GraderResult


# ---------------------------------------------------------------------------
# Constants — all reward magnitudes in one place for easy tuning
# ---------------------------------------------------------------------------

REWARD_INFORMATION_DISCLOSURE  =  0.05
REWARD_ACTIVE_LISTENING        =  0.03
REWARD_CONFLICT_DETECTION      =  0.05
PENALTY_LOOP                   = -0.05
PENALTY_SYCOPHANCY             = -0.10
PENALTY_TURN_DECAY             = -0.03   # Per turn past 80% of max_turns
PENALTY_TURN_LIMIT             = -0.15   # Flat penalty on hard cutoff

# Episode-level maximums
MAX_CORRECTNESS_SCORE          =  0.70
MAX_REASONING_QUALITY          =  0.20
MAX_EFFICIENCY_BONUS           =  0.10

# Thresholds
LOOP_SIMILARITY_THRESHOLD      =  0.70   # 70% word overlap = loop
SYCOPHANCY_MAX_WORDS           =  20     # Reasoning under this = suspect
SYCOPHANCY_KEYWORD_COUNT       =  2      # Fewer than this domain keywords = sycophantic


# ---------------------------------------------------------------------------
# Internal text utilities
# ---------------------------------------------------------------------------

def _tokenise(text: str) -> set[str]:
    """Lowercase word tokens, strips punctuation. Used for overlap comparison."""
    return set(re.findall(r'\b[a-z]{3,}\b', text.lower()))


def _word_count(text: str) -> int:
    return len(text.split())


def _overlap_ratio(text_a: str, text_b: str) -> float:
    """Jaccard-style overlap: shared words / union of words."""
    tokens_a = _tokenise(text_a)
    tokens_b = _tokenise(text_b)
    if not tokens_a or not tokens_b:
        return 0.0
    shared = tokens_a & tokens_b
    union  = tokens_a | tokens_b
    return len(shared) / len(union)


def _extract_private_info_words(private_info: dict) -> set[str]:
    """Flatten private info dict values into a set of meaningful words."""
    all_text = " ".join(
        str(v) for v in _flatten_dict(private_info).values()
    )
    return _tokenise(all_text)


def _flatten_dict(d: dict, prefix: str = "") -> dict:
    """Recursively flatten nested dicts into a single-level dict."""
    result = {}
    for k, v in d.items():
        if isinstance(v, dict):
            result.update(_flatten_dict(v, prefix=f"{prefix}{k}."))
        else:
            result[f"{prefix}{k}"] = v
    return result


def _prior_messages_from_agent(state: dict, agent_id: str) -> list[str]:
    """Return all prior message content strings from a specific agent."""
    return [
        msg.get("content", "")
        for msg in state.get("conversation", [])
        if msg.get("agent_id") == agent_id
    ]


# ---------------------------------------------------------------------------
# Step reward components — one function per component
# Each returns (reward_value: float, triggered: bool, reason: str)
# ---------------------------------------------------------------------------

def _check_information_disclosure(
    action_content: str,
    agent_id: str,
    private_info: dict,
    prior_messages: list[str],
) -> tuple[float, bool, str]:
    """
    +0.05 if the agent shares words that:
      (a) appear in their private information, AND
      (b) have NOT appeared in any of their prior messages.
    This rewards genuine new information disclosure, not repetition.
    """
    private_words  = _extract_private_info_words(private_info)
    current_words  = _tokenise(action_content)
    prior_text     = " ".join(prior_messages)
    prior_words    = _tokenise(prior_text)

    # New private words = in private info AND in this message AND not said before
    new_disclosures = private_words & current_words - prior_words

    if len(new_disclosures) >= 3:   # At least 3 new meaningful private words
        return (
            REWARD_INFORMATION_DISCLOSURE,
            True,
            f"Shared {len(new_disclosures)} new private-info terms"
        )
    return (0.0, False, "No new private information disclosed")


def _check_active_listening(
    action_content: str,
    action_reasoning: str,
    conversation: list[dict],
    agent_id: str,
) -> tuple[float, bool, str]:
    """
    +0.03 if the agent explicitly references or acknowledges the OTHER agent's last message.
    Checks for overlap between this message and the most recent message from the other agent.
    """
    # Find the most recent message from the OTHER agent
    other_agent_messages = [
        msg.get("content", "")
        for msg in conversation
        if msg.get("agent_id") != agent_id
    ]
    if not other_agent_messages:
        return (0.0, False, "No prior message from other agent to reference")

    last_other = other_agent_messages[-1]
    combined   = action_content + " " + action_reasoning
    overlap    = _overlap_ratio(combined, last_other)

    # Also check for explicit acknowledgement phrases
    ack_phrases = [
        "you mentioned", "you said", "your point", "i agree with",
        "building on", "as you noted", "you raised", "that's correct",
        "good point", "i understand your", "you're right that"
    ]
    has_ack_phrase = any(p in combined.lower() for p in ack_phrases)

    if overlap >= 0.25 or has_ack_phrase:
        return (
            REWARD_ACTIVE_LISTENING,
            True,
            f"Acknowledged other agent's message (overlap: {overlap:.2f}, explicit_ack: {has_ack_phrase})"
        )
    return (0.0, False, f"No clear acknowledgement of other agent (overlap: {overlap:.2f})")


def _check_conflict_detection(
    action_content: str,
    action_reasoning: str,
) -> tuple[float, bool, str]:
    """
    +0.05 if the agent explicitly calls out a conflict, discrepancy, or disagreement.
    Checks for conflict-detection vocabulary in content and reasoning.
    """
    conflict_phrases = [
        "conflict", "discrepancy", "disagree", "contradict", "inconsistent",
        "different from", "doesn't match", "does not match", "at odds",
        "contradicts", "mismatch", "conflicts with", "diverge", "contrary",
        "however", "but your data", "your information suggests", "that differs"
    ]
    combined = (action_content + " " + action_reasoning).lower()
    hits = [p for p in conflict_phrases if p in combined]

    if len(hits) >= 1:
        return (
            REWARD_CONFLICT_DETECTION,
            True,
            f"Conflict detection triggered by: {hits[:3]}"
        )
    return (0.0, False, "No explicit conflict identified")


def _check_loop_penalty(
    action_content: str,
    agent_id: str,
    prior_messages: list[str],
) -> tuple[float, bool, str]:
    """
    -0.05 if the current message is too similar to any prior message from the same agent.
    Uses word overlap ratio — 70%+ overlap = loop.
    """
    for prior in prior_messages:
        ratio = _overlap_ratio(action_content, prior)
        if ratio >= LOOP_SIMILARITY_THRESHOLD:
            return (
                PENALTY_LOOP,
                True,
                f"Loop detected — {ratio:.0%} overlap with prior message"
            )
    return (0.0, False, "No loop detected")


def _check_sycophancy_penalty(
    action: dict,
    task_keywords: list[str],
) -> tuple[float, bool, str]:
    """
    -0.10 if the agent accepts or rejects consensus without meaningful reasoning.
    Triggered when:
      - action_type is accept_consensus or reject_consensus, AND
      - reasoning is under SYCOPHANCY_MAX_WORDS words, AND
      - reasoning contains fewer than SYCOPHANCY_KEYWORD_COUNT domain keywords.
    """
    action_type = action.get("action_type", "")
    if action_type not in ("accept_consensus", "reject_consensus"):
        return (0.0, False, "Not a consensus action — sycophancy check skipped")

    reasoning   = action.get("reasoning", "")
    word_count  = _word_count(reasoning)
    reasoning_lower = reasoning.lower()
    keyword_hits = sum(1 for kw in task_keywords if kw.lower() in reasoning_lower)

    if word_count < SYCOPHANCY_MAX_WORDS and keyword_hits < SYCOPHANCY_KEYWORD_COUNT:
        return (
            PENALTY_SYCOPHANCY,
            True,
            f"Sycophancy: reasoning only {word_count} words with {keyword_hits} domain keywords"
        )
    return (
        0.0,
        False,
        f"Reasoning sufficient: {word_count} words, {keyword_hits} domain keywords"
    )


def _check_turn_decay(
    current_turn: int,
    max_turns: int,
    truncated: bool,
) -> tuple[float, bool, str]:
    """
    -0.03 per turn past 80% of max_turns.
    -0.15 flat penalty if the hard turn limit was hit (truncated=True).
    Returns the combined penalty for this turn.
    """
    warning_threshold = int(max_turns * 0.8)
    decay_penalty     = 0.0
    limit_penalty     = 0.0

    if current_turn > warning_threshold:
        turns_over = current_turn - warning_threshold
        decay_penalty = PENALTY_TURN_DECAY * turns_over

    if truncated:
        limit_penalty = PENALTY_TURN_LIMIT

    total = decay_penalty + limit_penalty
    if total < 0:
        return (
            total,
            True,
            f"Turn penalty: decay={decay_penalty:.2f}, hard_cutoff={limit_penalty:.2f}"
        )
    return (0.0, False, "Within turn budget")


# ---------------------------------------------------------------------------
# Public entry point 1 — Step reward
# ---------------------------------------------------------------------------

def compute_step_reward(
    action: dict,
    state: dict,
    prev_state: dict,
) -> tuple[float, RewardBreakdown]:
    """
    Compute the reward for a single agent action.

    Args:
        action:     The action dict (from Action.model_dump())
        state:      Current environment state (after action applied)
        prev_state: Environment state before the action was applied

    Returns:
        (step_reward: float, breakdown: RewardBreakdown)
    """
    agent_id       = action.get("agent_id", "")
    action_content = action.get("content", "")
    action_reason  = action.get("reasoning", "")
    action_type    = action.get("action_type", "")
    conversation   = prev_state.get("conversation", [])  # Before this action
    current_turn   = state.get("current_turn", 0)
    max_turns      = state.get("max_turns", 10)
    truncated      = state.get("truncated", False)

    # Get this agent's private info from the god-view state
    private_info = (
        state.get("private_info_a", {})
        if agent_id == AgentID.AGENT_A.value
        else state.get("private_info_b", {})
    )

    # Get all task keywords for sycophancy check (flatten correct_answer_keywords)
    raw_kw = state.get("correct_answer_keywords", [])
    if isinstance(raw_kw, dict):
        task_keywords = [kw for sublist in raw_kw.values() for kw in sublist]
    else:
        task_keywords = list(raw_kw)

    prior_messages = _prior_messages_from_agent(prev_state, agent_id)

    # --- Run each component ---
    disclosure_reward, _, disclosure_note = _check_information_disclosure(
        action_content, agent_id, private_info, prior_messages
    )
    listening_reward, _, listening_note = _check_active_listening(
        action_content, action_reason, conversation, agent_id
    )
    conflict_reward, _, conflict_note = _check_conflict_detection(
        action_content, action_reason
    )
    loop_penalty, _, loop_note = _check_loop_penalty(
        action_content, agent_id, prior_messages
    )
    sycophancy_penalty, _, syco_note = _check_sycophancy_penalty(
        action, task_keywords
    )
    turn_penalty, _, turn_note = _check_turn_decay(
        current_turn, max_turns, truncated
    )

    # --- Assemble breakdown ---
    breakdown = RewardBreakdown(
        information_disclosure = disclosure_reward,
        active_listening       = listening_reward,
        conflict_detection     = conflict_reward,
        loop_penalty           = loop_penalty,
        sycophancy_penalty     = sycophancy_penalty,
        turn_decay_penalty     = turn_penalty if not truncated else 0.0,
        turn_limit_penalty     = PENALTY_TURN_LIMIT if truncated else 0.0,
    )

    step_reward = (
        disclosure_reward +
        listening_reward  +
        conflict_reward   +
        loop_penalty      +
        sycophancy_penalty +
        turn_penalty
    )

    # Clamp step reward to prevent catastrophic single-turn scores
    step_reward = round(max(-0.49, min(0.29, step_reward)), 4)

    return step_reward, breakdown


# ---------------------------------------------------------------------------
# Public entry point 2 — Episode reward (called when done=True)
# ---------------------------------------------------------------------------

def compute_episode_reward(
    grader_result: GraderResult,
    state: dict,
) -> tuple[float, RewardBreakdown]:
    """
    Compute the episode-level reward from the grader result.
    Only called once per episode when done=True.

    Returns:
        (episode_reward: float, breakdown: RewardBreakdown)
    """
    turns_used = state.get("current_turn", 1)
    max_turns  = state.get("max_turns", 10)

    # Correctness — grader final score maps to 0.0–0.70
    correctness = grader_result.final_score * MAX_CORRECTNESS_SCORE

    # Reasoning quality — use dimension scores as a proxy
    # Average of non-correctness dimensions (conflict, synthesis, info sharing etc.)
    dim = grader_result.dimension_scores
    reasoning_dims = [
        v for k, v in dim.items()
        if k not in ("answer_correct", "consensus_reached", "efficiency")
    ]
    reasoning_raw = sum(reasoning_dims) / len(reasoning_dims) if reasoning_dims else 0.0
    reasoning     = round(reasoning_raw * MAX_REASONING_QUALITY, 4)

    # Efficiency bonus — reward solving quickly
    # Perfect efficiency (1 turn) = full bonus. Degrades linearly to 0 at max_turns.
    efficiency_ratio = max(0.0, 1.0 - (turns_used / max_turns))
    efficiency       = round(efficiency_ratio * MAX_EFFICIENCY_BONUS, 4)

    breakdown = RewardBreakdown(
        correctness_score  = round(correctness, 4),
        reasoning_quality  = reasoning,
        efficiency_bonus   = efficiency,
    )

    episode_reward = correctness + reasoning + efficiency
    episode_reward = round(min(0.99, max(0.01, episode_reward)), 4)

    return episode_reward, breakdown