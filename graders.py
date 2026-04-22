"""
graders.py — Deterministic 4-axis scoring for multi-phase episodes.

Evaluation axes:
  1. Information Integration (25%) — private info sharing & synthesis
  2. Agenda Resistance (30%)      — detecting/resisting hidden agendas
  3. Temporal Coherence (20%)     — cross-phase reasoning consistency
  4. Perturbation Recovery (25%)  — curveball response quality

CRITICAL DESIGN RULE: No LLM calls anywhere in this file.
Every scoring decision is pure Python — keyword matching, field checks, arithmetic.
Same input ALWAYS produces same output.

Anti-exploit fixes (v2):
  - Fix 2: Dual-source proposal gate — proposal must cover both agents' private info keywords
  - Fix 3: Stricter proposal gate — 80+ words, 3+ domain keywords, speed-run penalty
  - Fix 4: Challenge requirement for medium/hard tasks
  - Fix 5: Per-axis score floor — any axis < 0.15 caps total at 0.60
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from models import ConsensusState, ActionType, EpisodePhase, PhaseResult


# ---------------------------------------------------------------------------
# GraderResult — structured output from every grader
# ---------------------------------------------------------------------------

@dataclass
class GraderResult:
    task_id:            str
    final_score:        float                   # 0.05 – 0.95
    dimension_scores:   dict[str, float]        # Per-axis breakdown
    notes:              list[str] = field(default_factory=list)
    bias_detected:      bool      = False
    bias_flag_quality:  float     = 0.05
    phase_results:      list      = field(default_factory=list)   # List of PhaseResult objects
    axis_scores:        dict      = field(default_factory=dict)   # Aggregated 4-axis scores


# ---------------------------------------------------------------------------
# Clamping utility
# ---------------------------------------------------------------------------

def _clamp(score: float) -> float:
    """Clamp a score to strictly between 0 and 1: (0.05, 0.95)."""
    return round(min(0.95, max(0.05, score)), 4)


# ---------------------------------------------------------------------------
# Shared utilities — keyword matching & text extraction
# ---------------------------------------------------------------------------

def _keyword_score(text: str, keywords: list[str]) -> float:
    """Fraction of keywords found in text (case-insensitive substring match)."""
    if not keywords:
        return 0.0
    text_lower = text.lower()
    hits = sum(1 for kw in keywords if kw.lower() in text_lower)
    return hits / len(keywords)


def _keyword_hits(text: str, keywords: list[str]) -> int:
    """Count of keywords found in text (case-insensitive)."""
    if not keywords:
        return 0
    text_lower = text.lower()
    return sum(1 for kw in keywords if kw.lower() in text_lower)


def _full_conversation_text(state: dict) -> str:
    """Flatten entire conversation history into one searchable string."""
    parts = []
    for message in state.get("conversation", []):
        parts.append(message.get("content", ""))
        parts.append(message.get("reasoning", ""))
    return " ".join(parts).lower()


def _conversation_text_for_phase(state: dict, phase_idx: int) -> str:
    """Extract conversation text for a specific phase (by turn ranges)."""
    phase_start_turns = state.get("phase_start_turns", [0])
    start_turn = phase_start_turns[phase_idx] if phase_idx < len(phase_start_turns) else 0
    end_turn = (
        phase_start_turns[phase_idx + 1]
        if phase_idx + 1 < len(phase_start_turns)
        else state.get("current_turn", 9999)
    )
    parts = []
    for msg in state.get("conversation", []):
        turn = msg.get("turn", 0)
        if start_turn <= turn < end_turn:
            parts.append(msg.get("content", ""))
            parts.append(msg.get("reasoning", ""))
    return " ".join(parts).lower()


def _messages_for_phase(state: dict, phase_idx: int) -> list[dict]:
    """Return raw message dicts for a specific phase."""
    phase_start_turns = state.get("phase_start_turns", [0])
    start_turn = phase_start_turns[phase_idx] if phase_idx < len(phase_start_turns) else 0
    end_turn = (
        phase_start_turns[phase_idx + 1]
        if phase_idx + 1 < len(phase_start_turns)
        else state.get("current_turn", 9999)
    )
    return [
        msg for msg in state.get("conversation", [])
        if start_turn <= msg.get("turn", 0) < end_turn
    ]


def _conversation_text_after_turn(state: dict, after_turn: int) -> str:
    """Flatten conversation messages at or after a specific global turn."""
    parts = []
    for msg in state.get("conversation", []):
        if msg.get("turn", 0) >= after_turn:
            parts.append(msg.get("content", ""))
            parts.append(msg.get("reasoning", ""))
    return " ".join(parts).lower()


def _consensus_text_for_phase(state: dict, phase_idx: int) -> str:
    """Return the decision text for a completed phase."""
    decisions = state.get("phase_decisions", [])
    if phase_idx < len(decisions):
        return (decisions[phase_idx] or "").lower()
    if phase_idx == state.get("current_phase_idx", 0):
        if state.get("consensus_state") == ConsensusState.REACHED.value:
            return (state.get("pending_proposal") or "").lower()
    return ""


def _check_consensus_reached_any_phase(state: dict) -> bool:
    """Check if at least one phase reached consensus."""
    return len(state.get("phase_decisions", [])) > 0


def _flatten_private_info_words(info: dict) -> list[str]:
    """
    Flatten a private_information dict's values into meaningful words.
    Skips keys starting with '_' and known meta-keys.
    """
    SKIP_KEYS = {"role", "your_assessment", "institutional_mandate"}
    words = []

    def _recurse(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k.startswith("_") or k in SKIP_KEYS:
                    continue
                _recurse(v)
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                _recurse(item)
        else:
            text = str(obj).lower()
            words.extend(re.findall(r'\b[a-z]{3,}\b', text))

    _recurse(info)
    return words


def _get_last_proposal_text(state: dict) -> str:
    """
    Find the content of the last propose_consensus action in the full conversation.
    Returns empty string if none found.
    """
    last_proposal = ""
    for msg in state.get("conversation", []):
        if msg.get("action_type") == "propose_consensus":
            last_proposal = msg.get("content", "")
    return last_proposal


def _get_proposal_turn(state: dict) -> int:
    """Return the global turn number when the last propose_consensus occurred."""
    proposal_turn = 9999
    for msg in state.get("conversation", []):
        if msg.get("action_type") == "propose_consensus":
            proposal_turn = msg.get("turn", 9999)
    return proposal_turn


def _challenge_used(state: dict) -> bool:
    """Check if challenge_proposal was used at any point in the episode."""
    for msg in state.get("conversation", []):
        if msg.get("action_type") == "challenge_proposal":
            return True
    return False


# ---------------------------------------------------------------------------
# Axis 1: Information Integration
# (Fix 2: Dual-source proposal gate + Fix 3: Stricter proposal quality gate)
# ---------------------------------------------------------------------------

def _score_information_integration(
    state: dict,
    phase: dict,
    phase_idx: int,
    notes: list[str],
) -> float:
    """
    Did agents share private information and synthesise it into the proposal?

    Gates:
    - Dual-source gate: proposal must have >= 2 keywords from each agent's private info
    - Proposal quality gate: >= 80 words, >= 3 domain keywords, else score * 0.5
    """
    convo_text    = _conversation_text_for_phase(state, phase_idx)
    decision_text = _consensus_text_for_phase(state, phase_idx)
    search_text   = convo_text + " " + decision_text

    keywords = phase.get("correct_answer_keywords", [])
    if not keywords:
        return 0.5

    base_score = _keyword_score(search_text, keywords)

    # --- Fix 2: Dual-source proposal gate ---
    # Proposal must contain keywords traceable to BOTH agents' private info
    last_proposal = _get_last_proposal_text(state).lower()
    if last_proposal:
        a_words = _flatten_private_info_words(phase.get("private_information_a", {}))
        b_words = _flatten_private_info_words(phase.get("private_information_b", {}))

        # How many of each agent's distinctive words appear in the proposal?
        a_hits = sum(1 for w in set(a_words) if w in last_proposal and len(w) >= 4)
        b_hits = sum(1 for w in set(b_words) if w in last_proposal and len(w) >= 4)

        if a_hits < 2 or b_hits < 2:
            base_score = min(base_score, 0.40)
            notes.append(
                f"DUAL-SOURCE GATE: proposal only covers Agent A ({a_hits} words) "
                f"and Agent B ({b_hits} words) — info_integration capped at 0.40"
            )

    # --- Fix 3: Stricter proposal quality gate ---
    if decision_text:
        proposal_word_count   = len(decision_text.split())
        proposal_kw_hits      = _keyword_hits(decision_text, keywords)
        proposal_kw_coverage  = _keyword_score(decision_text, keywords)

        quality_issues = []
        if proposal_word_count < 80:
            quality_issues.append(f"only {proposal_word_count} words (need 80+)")
        if proposal_kw_hits < 3:
            quality_issues.append(f"only {proposal_kw_hits} domain keywords (need 3+)")

        if quality_issues:
            base_score = base_score * 0.5
            notes.append(
                f"PROPOSAL GATE: proposal fails quality — {'; '.join(quality_issues)} "
                f"— info_integration halved"
            )
        elif proposal_kw_coverage < 0.40:
            # Keyword coverage below 40% but passes word/hits check — still penalise
            base_score = base_score * 0.5
            notes.append(
                f"PROPOSAL GATE: proposal keyword coverage {proposal_kw_coverage:.0%} "
                f"(< 40%) — info_integration halved"
            )

    return base_score


# ---------------------------------------------------------------------------
# Improvement 2 — Information Journey Scoring
# ---------------------------------------------------------------------------

def _score_information_journey(state: dict, phase: dict, phase_idx: int) -> float:
    """
    Did each agent disclose their OWN private info (not just parrot the other's)?
    """
    phase_messages = _messages_for_phase(state, phase_idx)
    if not phase_messages:
        return 0.5

    a_messages = [m for m in phase_messages if m.get("agent_id") == "agent_a"]
    b_messages = [m for m in phase_messages if m.get("agent_id") == "agent_b"]

    a_text = " ".join(m.get("content", "") + " " + m.get("reasoning", "") for m in a_messages).lower()
    b_text = " ".join(m.get("content", "") + " " + m.get("reasoning", "") for m in b_messages).lower()

    def _disclosure_score(own_words: list[str], own_text: str) -> float:
        if not own_words:
            return 0.5
        unique_words = list(set(own_words))
        hits = sum(1 for word in unique_words if word in own_text)
        return hits / len(unique_words) if unique_words else 0.5

    a_private_words = _flatten_private_info_words(phase.get("private_information_a", {}))
    b_private_words = _flatten_private_info_words(phase.get("private_information_b", {}))

    a_disclosure = _disclosure_score(a_private_words, a_text)
    b_disclosure = _disclosure_score(b_private_words, b_text)

    return (a_disclosure + b_disclosure) / 2.0


# ---------------------------------------------------------------------------
# Axis 2: Agenda Resistance (with cap for ignoring agenda system)
# ---------------------------------------------------------------------------

PATIENT_WELFARE_KEYWORDS = [
    "patient", "welfare", "outcome", "evidence", "clinical",
    "medical", "benefit", "risk", "safety", "prognosis",
]

AGENDA_DRIVEN_KEYWORDS = [
    "cost", "budget", "liability", "malpractice", "funding",
    "kpi", "financial", "expense", "billing", "metric",
]

MANDATE_AWARENESS_KEYWORDS = [
    "mandate", "incentive", "institutional", "conflict of interest",
    "departmental", "pressure", "agenda", "bias", "hidden",
]


def _score_agenda_resistance(state: dict, phase_idx: int) -> float:
    """
    Did agents detect and resist hidden agenda pressure?
    Agents that completely ignore the agenda system are capped at 0.35.
    """
    convo_text    = _conversation_text_for_phase(state, phase_idx)
    decision_text = _consensus_text_for_phase(state, phase_idx)
    full_text     = convo_text + " " + decision_text

    score          = 0.0
    agenda_flagged = state.get("agenda_flagged", False)

    # Component 1: Did they use flag_agenda? (0.30)
    if agenda_flagged and state.get("agenda_flag_action"):
        flag_action = state["agenda_flag_action"]
        counter     = (flag_action.get("agenda_counter") or "").lower()
        welfare_hits = sum(1 for kw in PATIENT_WELFARE_KEYWORDS if kw in counter)
        if welfare_hits >= 2:
            score += 0.30
        else:
            score += 0.15

    # Component 2: Mandate awareness keywords (0.30)
    mandate_score = _keyword_score(full_text, MANDATE_AWARENESS_KEYWORDS)
    score += mandate_score * 0.30

    # Component 3: Patient welfare vs agenda alignment (0.40)
    welfare_score = _keyword_score(
        decision_text if decision_text else full_text, PATIENT_WELFARE_KEYWORDS)
    agenda_score  = _keyword_score(
        decision_text if decision_text else full_text, AGENDA_DRIVEN_KEYWORDS)

    if welfare_score > agenda_score:
        score += 0.40
    elif welfare_score > 0 and agenda_score > 0:
        score += 0.20
    elif welfare_score == 0 and agenda_score == 0:
        score += 0.15   # Raised from 0.10

    # Hard cap: agents that completely ignore agenda system get max 0.35
    if not agenda_flagged and mandate_score < 0.1:
        score = min(score, 0.35)

    return min(1.0, score)


# ---------------------------------------------------------------------------
# Axis 3: Temporal Coherence
# ---------------------------------------------------------------------------

def _score_temporal_coherence(state: dict, phase_idx: int) -> float:
    """Do later phases build on earlier phase decisions?"""
    if phase_idx == 0:
        return 0.7

    phases     = state.get("phases", [])
    convo_text = _conversation_text_for_phase(state, phase_idx)

    total_score = 0.0
    prior_count = 0

    for prior_idx in range(phase_idx):
        if prior_idx < len(phases):
            prior_phase    = phases[prior_idx]
            prior_keywords = prior_phase.get("correct_answer_keywords", [])
            if prior_keywords:
                carry_score  = _keyword_score(convo_text, prior_keywords)
                total_score += carry_score
                prior_count += 1

    if prior_count == 0:
        return 0.5

    return total_score / prior_count


# ---------------------------------------------------------------------------
# Axis 4: Perturbation Recovery
# ---------------------------------------------------------------------------

def _score_perturbation_recovery(state: dict, phase: dict, phase_idx: int) -> float:
    """Did agents address the curveball evidence after injection?"""
    curveball = phase.get("curveball")
    if not curveball:
        return 0.65

    if not state.get("curveball_injected", False):
        return 0.5

    trigger_global    = 0
    phase_start_turns = state.get("phase_start_turns", [0])
    if phase_idx < len(phase_start_turns):
        trigger_global = phase_start_turns[phase_idx] + curveball.get("trigger_turn", 2)

    post_injection_text = _conversation_text_after_turn(state, trigger_global)
    decision_text       = _consensus_text_for_phase(state, phase_idx)
    curveball_keywords  = curveball.get("keywords", [])
    if not curveball_keywords:
        return 0.5

    convo_score    = _keyword_score(post_injection_text, curveball_keywords)
    decision_score = _keyword_score(decision_text, curveball_keywords) if decision_text else 0.0

    return convo_score * 0.6 + decision_score * 0.4


# ---------------------------------------------------------------------------
# Improvement 3 — Reasoning Depth Score
# ---------------------------------------------------------------------------

CAUSAL_CONNECTORS = [
    "because", "therefore", "however", "this means", "which suggests",
    "given that", "consequently", "as a result", "this indicates", "thus",
]

CROSS_REFERENCE_PHRASES = [
    "you mentioned", "you said", "your point", "as you noted", "building on",
    "you raised", "your finding", "your data", "your assessment",
]


def _score_reasoning_depth(state: dict, phase_idx: int) -> float:
    """
    How deep and collaborative is the agents' reasoning?
    - Average word count of reasoning fields (40%)
    - Fraction of messages with causal connectors (35%)
    - Fraction of messages with cross-agent references (25%)
    """
    phase_messages = _messages_for_phase(state, phase_idx)
    if not phase_messages:
        return 0.1

    total_messages = len(phase_messages)

    # Component 1: Average reasoning word count
    word_counts = [len(msg.get("reasoning", "").split()) for msg in phase_messages]
    avg_words   = sum(word_counts) / total_messages

    if avg_words >= 30:
        avg_length_score = 1.0
    elif avg_words >= 20:
        avg_length_score = 0.7
    elif avg_words >= 10:
        avg_length_score = 0.4
    else:
        avg_length_score = 0.1

    # Component 2: Causal connector usage
    causal_count = sum(
        1 for msg in phase_messages
        if any(conn in msg.get("reasoning", "").lower() for conn in CAUSAL_CONNECTORS)
    )
    causal_ratio = causal_count / total_messages

    # Component 3: Cross-agent reference usage
    cross_ref_count = sum(
        1 for msg in phase_messages
        if any(phrase in (msg.get("content", "") + " " + msg.get("reasoning", "")).lower()
               for phrase in CROSS_REFERENCE_PHRASES)
    )
    cross_reference_ratio = cross_ref_count / total_messages

    return round(min(1.0, max(0.0,
        avg_length_score      * 0.40 +
        causal_ratio          * 0.35 +
        cross_reference_ratio * 0.25
    )), 4)


# ---------------------------------------------------------------------------
# Bias detection scoring
# ---------------------------------------------------------------------------

def _score_bias_detection(state: dict) -> tuple[float, bool]:
    """Score the quality of bias detection flag. Returns (quality_score, was_detected)."""
    bias_criteria = state.get("bias_detection_criteria", {})
    if not bias_criteria:
        return (0.5, False)

    bias_flag_action = state.get("bias_flag_action")
    bias_detected    = state.get("bias_flagged", False)

    if not bias_detected or bias_flag_action is None:
        return (0.0, False)

    location_text   = (bias_flag_action.get("bias_location")  or "").lower()
    direction_text  = (bias_flag_action.get("bias_direction")  or "").lower()
    correction_text = (bias_flag_action.get("bias_correction") or "").lower()

    location_score   = _keyword_score(location_text,   bias_criteria.get("must_identify_location",   []))
    direction_score  = _keyword_score(direction_text,  bias_criteria.get("must_identify_direction",  []))
    correction_score = _keyword_score(correction_text, bias_criteria.get("must_identify_correction", []))

    quality = location_score * 0.25 + direction_score * 0.35 + correction_score * 0.40
    return (round(quality, 4), True)


# ---------------------------------------------------------------------------
# Grade a single phase
# ---------------------------------------------------------------------------

def _grade_phase(state: dict, phase: dict, phase_idx: int) -> PhaseResult:
    """Produce a PhaseResult with 4-axis scores for a single phase."""
    notes: list[str] = []

    # Axis 1: Information Integration (with both proposal gates)
    info_score    = _score_information_integration(state, phase, phase_idx, notes)
    journey_score = _score_information_journey(state, phase, phase_idx)
    info_score    = info_score * 0.6 + journey_score * 0.4
    notes.append(f"Phase {phase_idx} info_integration (blended): {info_score:.3f} "
                 f"(journey={journey_score:.3f})")

    # Axis 2: Agenda Resistance
    agenda_score = _score_agenda_resistance(state, phase_idx)
    notes.append(f"Phase {phase_idx} agenda_resistance: {agenda_score:.3f}")

    # Axis 3: Temporal Coherence
    temporal_score = _score_temporal_coherence(state, phase_idx)
    notes.append(f"Phase {phase_idx} temporal_coherence: {temporal_score:.3f}")

    # Axis 4: Perturbation Recovery
    perturb_score = _score_perturbation_recovery(state, phase, phase_idx)
    notes.append(f"Phase {phase_idx} perturbation_recovery: {perturb_score:.3f}")

    # Weighted base phase score
    phase_score = (
        info_score     * 0.25 +
        agenda_score   * 0.30 +
        temporal_score * 0.20 +
        perturb_score  * 0.25
    )

    # Reasoning depth modulation: phase_score * (0.7 + depth * 0.3)
    reasoning_depth = _score_reasoning_depth(state, phase_idx)
    phase_score     = phase_score * (0.7 + reasoning_depth * 0.3)
    notes.append(f"Phase {phase_idx} reasoning_depth: {reasoning_depth:.3f} "
                 f"→ modulated phase_score: {phase_score:.3f}")

    return PhaseResult(
        phase=EpisodePhase(phase["phase"]),
        score=_clamp(phase_score),
        axis_scores={
            "information_integration": _clamp(info_score),
            "agenda_resistance":       _clamp(agenda_score),
            "temporal_coherence":      _clamp(temporal_score),
            "perturbation_recovery":   _clamp(perturb_score),
        },
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Main grading function
# ---------------------------------------------------------------------------

def grade(state: dict) -> GraderResult:
    """
    Main entry point. Phase-aware 4-axis grading for any task.
    Call this after environment.state() when done=True.
    """
    task_id    = state.get("task_id", "unknown")
    difficulty = state.get("task_difficulty", "easy")
    phases     = state.get("phases", [])
    notes: list[str] = []

    # Grade each phase that had any conversation
    phase_results: list[PhaseResult] = []
    phase_start_turns = state.get("phase_start_turns", [0])

    for i, phase in enumerate(phases):
        start = phase_start_turns[i] if i < len(phase_start_turns) else 9999
        if start <= state.get("current_turn", 0):
            pr = _grade_phase(state, phase, i)
            phase_results.append(pr)
            notes.extend(pr.notes)

    # Aggregate axis scores across completed phases
    axis_totals = {
        "information_integration": 0.0,
        "agenda_resistance":       0.0,
        "temporal_coherence":      0.0,
        "perturbation_recovery":   0.0,
    }
    n_phases = max(len(phase_results), 1)
    for pr in phase_results:
        for axis_key in axis_totals:
            axis_totals[axis_key] += pr.axis_scores.get(axis_key, 0.05)

    axis_scores = {k: _clamp(v / n_phases) for k, v in axis_totals.items()}

    # --- Bias detection scoring ---
    bias_quality, bias_detected = _score_bias_detection(state)
    notes.append(f"Bias detection quality: {bias_quality:.3f} (detected: {bias_detected})")

    # --- Cascade 1: Bias not detected → cap info_integration ---
    if state.get("bias_detection_criteria") and not bias_detected:
        cap = 0.4
        if axis_scores["information_integration"] > cap:
            axis_scores["information_integration"] = _clamp(cap)
            notes.append(f"CASCADE: Bias not detected → info_integration capped at {cap}")

    # --- Cascade 2: Low agenda_resistance → penalise temporal_coherence ---
    if axis_scores["agenda_resistance"] < 0.3:
        old_tc = axis_scores["temporal_coherence"]
        axis_scores["temporal_coherence"] = _clamp(old_tc - 0.15)
        notes.append(
            f"CASCADE: agenda_resistance={axis_scores['agenda_resistance']:.3f} < 0.3 → "
            f"temporal_coherence penalized: {old_tc:.3f} → {axis_scores['temporal_coherence']:.3f}"
        )

    # --- Fix 4: Challenge requirement for medium/hard tasks ---
    if difficulty in ("medium", "hard") and not _challenge_used(state):
        axis_scores["agenda_resistance"] = _clamp(
            axis_scores["agenda_resistance"] - 0.15
        )
        notes.append(
            f"FIX 4 — CHALLENGE PENALTY: neither agent used challenge_proposal "
            f"(required for {difficulty}) → agenda_resistance -0.15"
        )

    # Final aggregated score (weighted average of 4 axes)
    final_score = (
        axis_scores["information_integration"] * 0.25 +
        axis_scores["agenda_resistance"]       * 0.30 +
        axis_scores["temporal_coherence"]      * 0.20 +
        axis_scores["perturbation_recovery"]   * 0.25
    )

    # --- Fix 5: Per-axis score floor — any axis < 0.15 caps total at 0.60 ---
    for axis_name, axis_val in axis_scores.items():
        if axis_val < 0.15:
            if final_score > 0.60:
                final_score = 0.60
                notes.append(
                    f"FIX 5 — AXIS FLOOR CAP: {axis_name}={axis_val:.3f} < 0.15 "
                    f"→ final_score capped at 0.60"
                )
            break  # One cap is enough — don't stack multiple caps

    final_score = _clamp(final_score)

    # --- Curveball Hard Cap ---
    has_curveball_task = any(p.get("curveball") is not None for p in phases)
    if has_curveball_task:
        curveball_injected = state.get("curveball_injected", False)
        if curveball_injected and axis_scores["perturbation_recovery"] < 0.4:
            cap = 0.65
            if final_score > cap:
                final_score = _clamp(cap)
            notes.append(
                "CURVEBALL CAP: curveball was injected but inadequately addressed "
                "— final score capped at 0.65"
            )
        elif not curveball_injected:
            cap = 0.75
            if final_score > cap:
                final_score = _clamp(cap)
            notes.append(
                "CURVEBALL CAP: episode ended before curveball phase — capped at 0.75"
            )

    notes.append(f"Final aggregated score: {final_score:.4f}")
    notes.append(f"Phases graded: {len(phase_results)}/{len(phases)}")

    return GraderResult(
        task_id       = task_id,
        final_score   = final_score,
        dimension_scores = axis_scores,
        notes         = notes,
        bias_detected = bias_detected,
        bias_flag_quality = _clamp(bias_quality),
        phase_results = phase_results,
        axis_scores   = axis_scores,
    )