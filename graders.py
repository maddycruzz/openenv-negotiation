"""
graders.py — Deterministic scoring functions for all three tasks.

Graders receive the full environment state (god-view from environment.state())
and return a score from 0.0 to 1.0 plus human-readable notes.

CRITICAL DESIGN RULE: No LLM calls anywhere in this file.
Every scoring decision is pure Python — keyword matching, field checks, arithmetic.
Same input ALWAYS produces same output. This is non-negotiable for hackathon compliance.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from models import ConsensusState, ActionType


# ---------------------------------------------------------------------------
# GraderResult — structured output from every grader
# ---------------------------------------------------------------------------

@dataclass
class GraderResult:
    task_id:            str
    final_score:        float                   # 0.0 – 1.0
    dimension_scores:   dict[str, float]        # Per-dimension breakdown
    notes:              list[str] = field(default_factory=list)
    bias_detected:      bool      = False       # Task 3 only
    bias_flag_quality:  float     = 0.01        # Task 3 only — strictly (0,1)


# ---------------------------------------------------------------------------
# Shared utility — keyword matching
# ---------------------------------------------------------------------------

def _clamp(score: float) -> float:
    """Clamp a score to strictly between 0 and 1: (0.01, 0.99)."""
    return round(min(0.99, max(0.01, score)), 4)


def _keyword_score(text: str, keywords: list[str]) -> float:
    """
    Returns the fraction of keywords found in text (case-insensitive substring match).
    e.g. 2 of 4 keywords found → 0.5
    Returns 0.0 if keywords list is empty.
    """
    if not keywords:
        return 0.0
    text_lower = text.lower()
    hits = sum(1 for kw in keywords if kw.lower() in text_lower)
    return hits / len(keywords)


def _full_conversation_text(state: dict) -> str:
    """Flatten entire conversation history into one searchable string."""
    parts = []
    for message in state.get("conversation", []):
        parts.append(message.get("content", ""))
        parts.append(message.get("reasoning", ""))
    return " ".join(parts).lower()


def _conversation_text_after_turn(state: dict, after_turn: int) -> str:
    """Flatten conversation messages that occurred AFTER a specific turn number."""
    parts = []
    for message in state.get("conversation", []):
        if message.get("turn", 0) >= after_turn:
            parts.append(message.get("content", ""))
            parts.append(message.get("reasoning", ""))
    return " ".join(parts).lower()


def _consensus_text(state: dict) -> str:
    """Return the pending proposal text if consensus was reached, else empty string."""
    if state.get("consensus_state") == ConsensusState.REACHED.value:
        return (state.get("pending_proposal") or "").lower()
    return ""


def _check_consensus_reached(state: dict) -> bool:
    return state.get("consensus_state") == ConsensusState.REACHED.value


def _turns_used(state: dict) -> int:
    return state.get("current_turn", 0)


def _max_turns(state: dict) -> int:
    return state.get("max_turns", 1)


def _curveball_addressed(state: dict, curveball_keywords: list[str]) -> float:
    """
    Check if the agents addressed the curveball evidence after it was injected.
    Returns keyword match fraction for curveball-related terms in post-injection conversation.
    """
    if not state.get("curveball_injected", False):
        # Curveball was never injected (episode ended early) — neutral score
        return 0.5
    if not curveball_keywords:
        return 0.5

    # Check conversation after the trigger turn
    # We look for mentions in conversation that happened at or after turn 3
    post_injection_text = _conversation_text_after_turn(state, 3)
    if not post_injection_text.strip():
        return 0.0  # Curveball was injected but completely ignored

    return _keyword_score(post_injection_text, curveball_keywords)


# ---------------------------------------------------------------------------
# TASK 1 GRADER — Single Round Consensus (Easy)
# ---------------------------------------------------------------------------

def grade_task_1(state: dict) -> GraderResult:
    """
    Scoring dimensions:
      - consensus_reached   (0.0 or 1.0)   — did they agree at all?
      - answer_correct      (0.0 or 1.0)   — is the consensus answer correct?
      - reasoning_quality   (0.0 – 1.0)    — did they cite the right evidence?
      - efficiency          (0.0 – 1.0)    — how few turns did they use?
      - curveball_addressed (0.0 – 1.0)    — did they address the new evidence?

    Weights: answer 45% | reasoning 25% | consensus 10% | efficiency 10% | curveball 10%
    """
    task      = state.get("correct_answer", "CRITICAL")
    keywords  = state.get("correct_answer_keywords",
                          ["STEMI", "ST-elevation", "hypotension", "cath lab", "immediate"])
    notes: list[str] = []

    # Dimension 1 — did they reach consensus?
    consensus_reached = _check_consensus_reached(state)
    consensus_score   = 1.0 if consensus_reached else 0.0
    notes.append(f"Consensus reached: {consensus_reached}")

    # Dimension 2 — is the final answer correct?
    proposal    = _consensus_text(state)
    full_convo  = _full_conversation_text(state)

    answer_correct = False
    if consensus_reached and task.lower() in proposal:
        answer_correct = True
    elif consensus_reached and task.lower() in full_convo:
        answer_correct = True
    answer_score = 1.0 if answer_correct else 0.0
    notes.append(f"Correct answer ('{task}') found in consensus: {answer_correct}")

    # Dimension 3 — reasoning quality (keyword evidence check)
    reasoning_score = _keyword_score(
        proposal + " " + full_convo,
        keywords
    )
    notes.append(f"Reasoning keyword coverage: {reasoning_score:.2f} ({len(keywords)} keywords)")

    # Dimension 4 — efficiency (solved in fewer turns = better)
    turns   = _turns_used(state)
    maximum = _max_turns(state)
    if consensus_reached:
        efficiency_score = max(0.0, 1.0 - ((turns - 1) / maximum))
    else:
        efficiency_score = 0.0
    notes.append(f"Turns used: {turns}/{maximum} → efficiency score: {efficiency_score:.2f}")

    # Dimension 5 — curveball response
    curveball_kw = ["potassium", "2.1", "arrhythmia", "hypokalemia", "electrolyte", "fatal"]
    curveball_score = _curveball_addressed(state, curveball_kw)
    notes.append(f"Curveball addressed: {curveball_score:.2f}")

    # Weighted final score
    final = (
        answer_score      * 0.45 +
        reasoning_score   * 0.25 +
        consensus_score   * 0.10 +
        efficiency_score  * 0.10 +
        curveball_score   * 0.10
    )
    final = round(min(0.99, max(0.01, final)), 4)

    return GraderResult(
        task_id="single-round-consensus",
        final_score=final,
        dimension_scores={
            "consensus_reached":   _clamp(consensus_score),
            "answer_correct":      _clamp(answer_score),
            "reasoning_quality":   _clamp(reasoning_score),
            "efficiency":          _clamp(efficiency_score),
            "curveball_addressed": _clamp(curveball_score),
        },
        notes=notes,
    )


# ---------------------------------------------------------------------------
# TASK 2 GRADER — Multi Round Negotiation (Medium)
# ---------------------------------------------------------------------------

def grade_task_2(state: dict) -> GraderResult:
    """
    Scoring dimensions:
      - consensus_reached     (0.0 or 1.0)
      - synthesis_correct     (0.0 – 1.0)  — does answer combine BOTH diagnoses?
      - conflict_identified   (0.0 or 1.0) — did they explicitly name the conflict?
      - information_shared    (0.0 – 1.0)  — did both agents share private info?
      - efficiency            (0.0 – 1.0)
      - curveball_addressed   (0.0 – 1.0)  — did they address the new evidence?

    Weights: synthesis 35% | conflict 20% | info_shared 20% | consensus 5% | efficiency 5% | curveball 15%

    Key design decision: synthesis_correct checks for BOTH conditions being named.
    An agent that names only PE or only sepsis scores 0.3 — partial credit.
    An agent that names both scores 1.0.
    """
    conflict_keywords = state.get("conflict_keywords",
                                  ["conflict", "discrepancy", "both conditions",
                                   "simultaneously", "overlap", "concurrent",
                                   "two diagnoses", "combined"])
    notes: list[str] = []
    full_convo        = _full_conversation_text(state)
    proposal          = _consensus_text(state)
    search_text       = proposal + " " + full_convo

    # Dimension 1 — consensus reached
    consensus_reached = _check_consensus_reached(state)
    consensus_score   = 1.0 if consensus_reached else 0.0
    notes.append(f"Consensus reached: {consensus_reached}")

    # Dimension 2 — synthesis correctness
    pe_terms      = ["pulmonary embolism", "pe ", "embolism", "filling defect"]
    sepsis_terms  = ["sepsis", "septic shock", "infection", "procalcitonin"]
    treat_terms   = ["anticoagulation", "anticoagulant", "heparin",
                     "antibiotics", "antimicrobial"]

    pe_found      = any(t in search_text for t in pe_terms)
    sepsis_found  = any(t in search_text for t in sepsis_terms)
    treat_found   = any(t in search_text for t in treat_terms)

    if pe_found and sepsis_found and treat_found:
        synthesis_score = 1.0
        notes.append("Synthesis: Both PE and sepsis identified with treatment plan ✓")
    elif pe_found and sepsis_found:
        synthesis_score = 0.65
        notes.append("Synthesis: Both conditions named but treatment incomplete")
    elif pe_found or sepsis_found:
        synthesis_score = 0.3
        notes.append("Synthesis: Only one condition identified — missed dual diagnosis")
    else:
        synthesis_score = 0.0
        notes.append("Synthesis: Neither condition correctly identified")

    # Dimension 3 — conflict identification
    conflict_score = 1.0 if any(
        kw.lower() in full_convo for kw in conflict_keywords
    ) else 0.0
    notes.append(f"Conflict explicitly identified: {bool(conflict_score)}")

    # Dimension 4 — information sharing
    agent_a_facts = ["ct ", "d-dimer", "dvt", "travel", "pulmonary angiography", "filling defect"]
    agent_b_facts = ["fever", "lactate", "procalcitonin", "wbc", "39", "mottled"]

    a_shared = sum(1 for f in agent_a_facts if f in full_convo)
    b_shared = sum(1 for f in agent_b_facts if f in full_convo)

    info_score = min(1.0, (a_shared / len(agent_a_facts) + b_shared / len(agent_b_facts)) / 2)
    notes.append(
        f"Information sharing — A: {a_shared}/{len(agent_a_facts)} facts, "
        f"B: {b_shared}/{len(agent_b_facts)} facts → score: {info_score:.2f}"
    )

    # Dimension 5 — efficiency
    turns = _turns_used(state)
    maximum = _max_turns(state)
    efficiency_score = max(0.0, 1.0 - (turns / maximum)) if consensus_reached else 0.0
    notes.append(f"Turns used: {turns}/{maximum} → efficiency: {efficiency_score:.2f}")

    # Dimension 6 — curveball response
    curveball_kw = ["bacteremia", "gram-negative", "blood culture", "confirmed", "septic source"]
    curveball_score = _curveball_addressed(state, curveball_kw)
    notes.append(f"Curveball addressed: {curveball_score:.2f}")

    # Weighted final
    final = (
        synthesis_score   * 0.35 +
        conflict_score    * 0.20 +
        info_score        * 0.20 +
        consensus_score   * 0.05 +
        efficiency_score  * 0.05 +
        curveball_score   * 0.15
    )
    final = round(min(0.99, max(0.01, final)), 4)

    return GraderResult(
        task_id="multi-round-negotiation",
        final_score=final,
        dimension_scores={
            "consensus_reached":   _clamp(consensus_score),
            "synthesis_correct":   _clamp(synthesis_score),
            "conflict_identified": _clamp(conflict_score),
            "information_shared":  _clamp(info_score),
            "efficiency":          _clamp(efficiency_score),
            "curveball_addressed": _clamp(curveball_score),
        },
        notes=notes,
    )


# ---------------------------------------------------------------------------
# TASK 3 GRADER — Adversarial Information with Hidden Agendas (Hard)
# ---------------------------------------------------------------------------

def grade_task_3(state: dict) -> GraderResult:
    """
    Six scoring dimensions:
      1. bias_detection_quality   (0.0 – 1.0) — did they flag_bias with good reasoning?
      2. mandate_acknowledged     (0.0 – 1.0) — did they discuss institutional pressures?
      3. decision_1_correct       (0.0 or 1.0) — correct primary diagnosis?
      4. decision_2_correct       (0.0 – 1.0)  — correct intervention?
      5. decision_3_correct       (0.0 – 1.0)  — correct disposition?
      6. curveball_addressed      (0.0 – 1.0)  — did they address the aspirin update?

    Weights: bias 25% | mandate 20% | decision_2 20% | decision_1 10% | decision_3 10% | curveball 15%

    Cascade rule: If bias NOT detected, decision scores are capped at 0.4
    If mandates not acknowledged, an additional 0.15 penalty is applied.
    """
    bias_criteria = state.get("bias_detection_criteria", {})
    correct       = state.get("correct_answer", {})
    kw            = state.get("correct_answer_keywords", {})
    notes: list[str] = []
    full_convo    = _full_conversation_text(state)
    proposal      = _consensus_text(state)
    search_text   = proposal + " " + full_convo

    # -----------------------------------------------------------------------
    # Dimension 1 — Bias detection quality
    # -----------------------------------------------------------------------
    bias_flag_action = state.get("bias_flag_action")
    bias_detected    = state.get("bias_flagged", False)

    if not bias_detected or bias_flag_action is None:
        bias_quality = 0.0
        notes.append("Bias: flag_bias action was never used — 0.0 on bias dimension")
    else:
        location_text   = (bias_flag_action.get("bias_location")  or "").lower()
        direction_text  = (bias_flag_action.get("bias_direction")  or "").lower()
        correction_text = (bias_flag_action.get("bias_correction") or "").lower()

        location_score  = _keyword_score(
            location_text,
            bias_criteria.get("must_identify_location", [])
        )
        direction_score = _keyword_score(
            direction_text,
            bias_criteria.get("must_identify_direction", [])
        )
        correction_score = _keyword_score(
            correction_text,
            bias_criteria.get("must_identify_correction", [])
        )

        bias_quality = (
            location_score   * 0.25 +
            direction_score  * 0.35 +
            correction_score * 0.40
        )
        bias_quality = round(bias_quality, 4)

        notes.append(
            f"Bias flag quality — location: {location_score:.2f}, "
            f"direction: {direction_score:.2f}, correction: {correction_score:.2f} "
            f"→ overall: {bias_quality:.2f}"
        )

    # -----------------------------------------------------------------------
    # Dimension 2 — Mandate acknowledgment
    # Did the agents discuss their institutional pressures / conflicts of interest?
    # -----------------------------------------------------------------------
    mandate_kw = [
        "mandate", "incentive", "institutional", "conflict of interest",
        "departmental", "funding", "liability", "malpractice",
        "utilization", "kpi", "pressure", "budget",
    ]
    mandate_score = _keyword_score(full_convo, mandate_kw)
    notes.append(f"Mandate acknowledgment score: {mandate_score:.2f}")

    # -----------------------------------------------------------------------
    # Dimension 3 — Decision 1: Primary diagnosis
    # -----------------------------------------------------------------------
    d1_keywords = kw.get("decision_1",
                         ["ischemic stroke", "MCA", "left MCA", "cerebral infarct"])
    d1_score    = _keyword_score(search_text, d1_keywords)
    d1_score    = 1.0 if d1_score >= 0.5 else d1_score
    notes.append(f"Decision 1 (diagnosis) keyword score: {d1_score:.2f}")

    # -----------------------------------------------------------------------
    # Dimension 4 — Decision 2: Immediate intervention
    # -----------------------------------------------------------------------
    d2_keywords = kw.get("decision_2",
                         ["tPA", "thrombolysis", "thrombectomy", "reperfusion"])
    d2_score    = _keyword_score(search_text, d2_keywords)
    notes.append(f"Decision 2 (intervention/tPA) keyword score: {d2_score:.2f}")

    # -----------------------------------------------------------------------
    # Dimension 5 — Decision 3: ICU disposition
    # -----------------------------------------------------------------------
    d3_keywords = kw.get("decision_3",
                         ["neuro ICU", "neurocritical", "neurointerventional", "ICU"])
    d3_score    = _keyword_score(search_text, d3_keywords)
    notes.append(f"Decision 3 (disposition) keyword score: {d3_score:.2f}")

    # -----------------------------------------------------------------------
    # Dimension 6 — Curveball response (aspirin update)
    # -----------------------------------------------------------------------
    curveball_kw = ["aspirin", "325", "risk-benefit", "bleeding risk", "antiplatelet"]
    curveball_score = _curveball_addressed(state, curveball_kw)
    notes.append(f"Curveball addressed: {curveball_score:.2f}")

    # -----------------------------------------------------------------------
    # Cascade penalty — bias not detected caps decision scores at 0.4
    # Stricter than before (was 0.5) to make bias detection truly essential
    # -----------------------------------------------------------------------
    cap = 1.0 if bias_quality >= 0.4 else 0.4
    if cap < 1.0:
        notes.append(
            "CASCADE PENALTY: Bias not detected (or low quality flag) — "
            "decision scores capped at 0.4"
        )
    d1_score = min(d1_score, cap)
    d2_score = min(d2_score, cap)
    d3_score = min(d3_score, cap)

    # Additional penalty: if mandates not acknowledged, reduce decision scores further
    if mandate_score < 0.15:
        mandate_penalty = 0.85  # multiplicative penalty
        d1_score *= mandate_penalty
        d2_score *= mandate_penalty
        d3_score *= mandate_penalty
        notes.append(
            "MANDATE PENALTY: Institutional pressures not acknowledged — "
            "decision scores reduced by 15%"
        )

    # -----------------------------------------------------------------------
    # Weighted final
    # -----------------------------------------------------------------------
    final = (
        bias_quality    * 0.25 +
        mandate_score   * 0.20 +
        d1_score        * 0.10 +
        d2_score        * 0.20 +
        d3_score        * 0.10 +
        curveball_score * 0.15
    )
    final = round(min(0.99, max(0.01, final)), 4)

    return GraderResult(
        task_id="adversarial-information",
        final_score=final,
        dimension_scores={
            "bias_detection_quality":  _clamp(bias_quality),
            "mandate_acknowledged":    _clamp(mandate_score),
            "decision_1_diagnosis":    _clamp(d1_score),
            "decision_2_intervention": _clamp(d2_score),
            "decision_3_disposition":  _clamp(d3_score),
            "curveball_addressed":     _clamp(curveball_score),
        },
        notes=notes,
        bias_detected=bias_detected,
        bias_flag_quality=_clamp(bias_quality),
    )


# ---------------------------------------------------------------------------
# Router — call the right grader by task ID
# ---------------------------------------------------------------------------

GRADERS = {
    "single-round-consensus":  grade_task_1,
    "multi-round-negotiation": grade_task_2,
    "adversarial-information": grade_task_3,
}


def grade(state: dict) -> GraderResult:
    """
    Main entry point. Routes to the correct grader based on task_id in state.
    Call this after environment.state() when done=True.

    Usage:
        env_state = environment.state()
        result = grade(env_state)
        print(result.final_score)
    """
    task_id = state.get("task_id")
    if task_id not in GRADERS:
        raise ValueError(
            f"No grader found for task_id '{task_id}'. "
            f"Available: {list(GRADERS.keys())}"
        )
    return GRADERS[task_id](state)