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

# Round 2: New reward components
REWARD_AGENDA_RESISTANCE       =  0.08
REWARD_CURVEBALL_RECOVERY      =  0.10
REWARD_PHASE_COMPLETION        =  0.12
PENALTY_MANDATE                = -0.12

# Episode-level maximums
MAX_CORRECTNESS_SCORE          =  0.70
MAX_REASONING_QUALITY          =  0.20
MAX_EFFICIENCY_BONUS           =  0.10

# Fix 1 — Minimum turn gates (anti speed-run)
MIN_TURNS = {"easy": 6, "medium": 8, "hard": 10}
MIN_TURN_PENALTY = {"easy": 0.30, "medium": 0.35, "hard": 0.40}

# Thresholds
LOOP_SIMILARITY_THRESHOLD      =  0.70   # 70% word overlap = loop
SYCOPHANCY_MAX_WORDS           =  20     # Reasoning under this = suspect

# Keywords for mandate penalty and agenda bonus
PATIENT_WELFARE_KEYWORDS = [
    "patient", "welfare", "outcome", "evidence", "clinical", 
    "medical", "benefit", "risk", "safety", "prognosis"
]
AGENDA_DRIVEN_KEYWORDS = [
    "cost", "budget", "liability", "malpractice", "funding", 
    "kpi", "financial", "expense", "billing", "metric"
]


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


def _keyword_count(text: str, keywords: list[str]) -> int:
    """Count how many keywords from the list appear in the text."""
    if not keywords or not text:
        return 0
    text_lower = text.lower()
    return sum(1 for kw in keywords if kw.lower() in text_lower)


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
    difficulty: str,
) -> tuple[float, bool, str]:
    action_type = action.get("action_type", "")
    if action_type not in ("accept_consensus", "reject_consensus"):
        return (0.0, False, "Not a consensus action — sycophancy check skipped")

    reasoning   = action.get("reasoning", "")
    word_count  = _word_count(reasoning)
    keyword_hits = _keyword_count(reasoning, task_keywords)
    
    # 3 keywords required for medium/hard, 2 for easy
    required_keywords = 3 if difficulty in ("medium", "hard") else 2

    if word_count < SYCOPHANCY_MAX_WORDS and keyword_hits < required_keywords:
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


# --- Round 2 New Components ---

def _check_agenda_resistance_bonus(action: dict) -> tuple[float, bool, str]:
    if action.get("action_type") == "flag_agenda":
        agenda_type = action.get("agenda_type")
        agenda_evidence = action.get("agenda_evidence")
        agenda_counter = action.get("agenda_counter")
        
        if agenda_type and agenda_evidence and agenda_counter:
            welfare_hits = _keyword_count(agenda_counter, PATIENT_WELFARE_KEYWORDS)
            if welfare_hits >= 2:
                return (REWARD_AGENDA_RESISTANCE, True, "Correctly flagged agenda with patient welfare focus")
            else:
                return (0.0, False, f"Flagged agenda but lacked patient welfare focus (found {welfare_hits} keywords)")
    return (0.0, False, "Not a valid agenda flag")

def _check_curveball_recovery_bonus(action: dict, state: dict, prev_state: dict) -> tuple[float, bool, str]:
    was_injected = prev_state.get("curveball_injected", False)
    is_injected = state.get("curveball_injected", False)
    already_awarded = prev_state.get("curveball_awarded", False)
    
    # We only care if it's injected, not necessarily on the exact turn it became true, 
    # but the prompt says: "First turn after curveball_injected becomes True" AND "Only awarded once per episode"
    # So we'll track 'curveball_awarded' in state? Actually state doesn't track this out of the box... wait, environment.py has:
    # self._curveball_awarded = False
    
    if is_injected and not already_awarded:
        # Check if the curveball keywords are in the content
        # We need the curveball keywords. They are in current_phase['curveball']['keywords']
        phases = state.get("phases", [])
        current_phase_idx = state.get("current_phase_idx", 0)
        current_phase = phases[current_phase_idx] if current_phase_idx < len(phases) else {}
        curveball = current_phase.get("curveball") or {}
        curveball_keywords = curveball.get("keywords", [])
        
        action_content = action.get("content", "")
        action_reasoning = action.get("reasoning", "")
        combined = action_content + " " + action_reasoning
        
        hits = _keyword_count(combined, curveball_keywords)
        if hits >= 2:
            # We must flag it as awarded by modifying state? No, rewards.py shouldn't mutate state.
            # But the environment passes `prev_state` and `state`. The environment doesn't look at rewards.py to update state.
            # However, prompt says: "Only awarded once per episode".
            # The environment doesn't set `curveball_awarded` currently, so we'll just check if it's the very first message after `curveball_injected` is true.
            pass

    # Better approach for "Only once per episode": since rewards.py shouldn't mutate state, 
    # environment.py HAS self._curveball_awarded = False! But it doesn't update it!
    # Wait, the prompt didn't say to add `curveball_awarded` update in environment, but I did. 
    # Let me check if I can just look at `prev_state["curveball_injected"]` vs `state["curveball_injected"]`?
    # No, the agent needs to respond to it.
    
    # We can check prior messages. If any prior message already had curveball recovery, we skip.
    if is_injected:
        current_phase_idx = state.get("current_phase_idx", 0)
        phases = state.get("phases", [])
        if current_phase_idx < len(phases):
            curveball_keywords = phases[current_phase_idx].get("curveball", {}).get("keywords", [])
            content = action.get("content", "") + " " + action.get("reasoning", "")
            if _keyword_count(content, curveball_keywords) >= 2:
                # Did anyone already get this?
                # Check messages in current phase since injection
                # The injection happens on _phase_turn >= trigger_turn.
                # Actually, simpler: we can just check if any message *before* this action in the current phase already had 2 keywords.
                prior_convo = prev_state.get("conversation", [])
                
                # To be completely safe and avoid mutating state, we check if prior convo already has 2 hits
                # post injection.
                already_addressed = False
                for msg in prior_convo:
                    # We can't easily tell when injection occurred just from convo, but we can just check if ANY prior message hit the keywords
                    if _keyword_count(msg.get("content", "") + " " + msg.get("reasoning", ""), curveball_keywords) >= 2:
                        already_addressed = True
                        break
                
                if not already_addressed:
                    return (REWARD_CURVEBALL_RECOVERY, True, "Recovered from curveball")
                
    return (0.0, False, "No curveball recovery")

def _check_phase_completion_bonus(action: dict, state: dict, prev_state: dict) -> tuple[float, bool, str]:
    action_type = action.get("action_type")
    
    # environment.py sets phase_just_advanced=True in info, but not in state directly.
    # However, prev_state['current_phase_idx'] < state['current_phase_idx'] works.
    prev_idx = prev_state.get("current_phase_idx", 0)
    curr_idx = state.get("current_phase_idx", 0)
    
    if action_type == "accept_consensus" and curr_idx > prev_idx:
        return (REWARD_PHASE_COMPLETION, True, f"Completed phase {prev_idx + 1}")
    return (0.0, False, "Phase not completed by this action")

def _check_mandate_penalty(action: dict) -> tuple[float, bool, str]:
    action_type = action.get("action_type")
    if action_type in ("propose_consensus", "accept_consensus"):
        content = action.get("content", "") + " " + action.get("reasoning", "")
        target_hits = _keyword_count(content, AGENDA_DRIVEN_KEYWORDS)
        welfare_hits = _keyword_count(content, PATIENT_WELFARE_KEYWORDS)
        if target_hits > 0 and welfare_hits == 0:
            return (PENALTY_MANDATE, True, "Agenda-driven proposal lacks patient welfare focus")
    return (0.0, False, "No mandate penalty")


# ---------------------------------------------------------------------------
# Public entry point 1 — Step reward
# ---------------------------------------------------------------------------

def compute_step_reward(
    action: dict,
    state: dict,
    prev_state: dict,
) -> tuple[float, RewardBreakdown]:
    agent_id       = action.get("agent_id", "")
    action_content = action.get("content", "")
    action_reason  = action.get("reasoning", "")
    action_type    = action.get("action_type", "")
    conversation   = prev_state.get("conversation", [])  # Before this action
    current_turn   = state.get("current_turn", 0)
    max_turns      = state.get("max_turns", 10)
    truncated      = state.get("truncated", False)
    difficulty     = state.get("task_difficulty", "easy")

    # Get this agent's private info from the god-view state
    private_info = (
        state.get("private_info_a", {})
        if agent_id == AgentID.AGENT_A.value
        else state.get("private_info_b", {})
    )

    # Get all task keywords for sycophancy check
    phases = state.get("phases", [])
    raw_kw = []
    for p in phases:
        raw_kw.extend(p.get("correct_answer_keywords", []))
    task_keywords = list(set(raw_kw))

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
        action, task_keywords, difficulty
    )
    turn_penalty, _, turn_note = _check_turn_decay(
        current_turn, max_turns, truncated
    )

    # --- Round 2 components ---
    agenda_bonus, _, agenda_note = _check_agenda_resistance_bonus(action)
    curveball_bonus, _, curveball_note = _check_curveball_recovery_bonus(action, state, prev_state)
    phase_bonus, _, phase_note = _check_phase_completion_bonus(action, state, prev_state)
    mandate_penalty, _, mandate_note = _check_mandate_penalty(action)

    def _safe(v):
        if v == 0:
            return 0.0   # 0 means "did not fire" — don't mask with 0.01
        return round(min(0.9999, max(0.0001, abs(v))) * (1 if v > 0 else -1), 4)

    breakdown = RewardBreakdown(
        information_disclosure   = _safe(disclosure_reward),
        active_listening         = _safe(listening_reward),
        conflict_detection       = _safe(conflict_reward),
        loop_penalty             = _safe(loop_penalty),
        sycophancy_penalty       = _safe(sycophancy_penalty),
        turn_decay_penalty       = _safe(turn_penalty if not truncated else 0.0),
        turn_limit_penalty       = _safe(PENALTY_TURN_LIMIT if truncated else 0.0),
        agenda_resistance_bonus  = _safe(agenda_bonus),
        curveball_recovery_bonus = _safe(curveball_bonus),
        phase_completion_bonus   = _safe(phase_bonus),
        mandate_penalty          = _safe(mandate_penalty)
    )

    step_reward = (
        disclosure_reward +
        listening_reward  +
        conflict_reward   +
        loop_penalty      +
        sycophancy_penalty +
        turn_penalty      +
        agenda_bonus      +
        curveball_bonus   +
        phase_bonus       +
        mandate_penalty
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
    turns_used = state.get("current_turn", 1)
    max_turns  = state.get("max_turns", 10)
    difficulty = state.get("task_difficulty", "easy")

    # Correctness — grader final score maps to 0.0–0.70
    correctness = grader_result.final_score * MAX_CORRECTNESS_SCORE

    # Reasoning quality — use dimension scores as a proxy
    dim = grader_result.dimension_scores
    if dim:
        reasoning_raw = sum(dim.values()) / len(dim)
    else:
        reasoning_raw = 0.0
    reasoning = round(reasoning_raw * MAX_REASONING_QUALITY, 4)

    # Efficiency bonus
    efficiency_ratio = max(0.0, 1.0 - (turns_used / max_turns))
    efficiency       = round(efficiency_ratio * MAX_EFFICIENCY_BONUS, 4)

    episode_reward = correctness + reasoning + efficiency

    # --- Fix 1: Minimum turn gate ---
    min_turns_required = MIN_TURNS.get(difficulty, 6)
    if turns_used < min_turns_required:
        penalty = MIN_TURN_PENALTY.get(difficulty, 0.30)
        episode_reward -= penalty
        # Note is passed via grader_result.notes in callers; reward is self-documenting

    # --- Fix 3 (speed-run supplement): penalise proposal accepted at turn <= 5 ---
    # Check the conversation for accept_consensus at a low global turn
    for msg in state.get("conversation", []):
        if msg.get("action_type") == "accept_consensus" and msg.get("turn", 99) < MIN_TURNS.get(difficulty, 6):
            episode_reward -= 0.20
            break

    episode_reward = round(min(0.95, max(0.05, episode_reward)), 4)

    def _safe(v):
        if v == 0:
            return 0.0   # 0 means "did not fire" — don't mask with 0.01
        return round(min(0.9999, max(0.0001, abs(v))) * (1 if v > 0 else -1), 4)

    breakdown = RewardBreakdown(
        correctness_score  = _safe(correctness),
        reasoning_quality  = _safe(reasoning),
        efficiency_bonus   = _safe(efficiency),
        # Step-level components are 0 at episode level (not applicable)
        information_disclosure   = 0.0,
        active_listening         = 0.0,
        conflict_detection       = 0.0,
        loop_penalty             = 0.0,
        sycophancy_penalty       = 0.0,
        turn_decay_penalty       = 0.0,
        turn_limit_penalty       = 0.0,
        agenda_resistance_bonus  = 0.0,
        curveball_recovery_bonus = 0.0,
        phase_completion_bonus   = 0.0,
        mandate_penalty          = 0.0,
    )

    return episode_reward, breakdown