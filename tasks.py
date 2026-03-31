"""
tasks.py — Three task definitions for the Social Agent Negotiation environment.

Each task is a plain dict matching the schema NegotiationEnvironment expects.
Tasks are ordered Easy → Medium → Hard with genuine difficulty progression.

Scenario domain: Medical triage decisions.
Chosen because:
  - Decisions are high-stakes and objectively gradeable
  - Information asymmetry is natural (different test results, different observations)
  - Bias is realistic (anchoring on initial diagnosis)
  - Judges from any background immediately understand the stakes
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# TASK 1 — EASY
# Single-round consensus on an unambiguous case.
# Both agents have mostly aligned information with minor differences.
# Correct answer is objectively determinable from either agent's info alone.
# Target: consensus in 2-3 turns.
# ---------------------------------------------------------------------------

TASK_1_EASY: dict = {
    "id": "single-round-consensus",
    "difficulty": "easy",
    "max_turns": 6,
    "description": (
        "You are two emergency physicians reviewing a patient case together. "
        "You must reach consensus on the immediate treatment priority: "
        "CRITICAL (immediate intervention), URGENT (within 1 hour), or ROUTINE (within 4 hours). "
        "Share your findings, discuss any differences, and agree on a single priority level. "
        "Once agreed, one of you should propose consensus and the other should accept it."
    ),

    # Agent A sees vitals and chief complaint
    "private_information_a": {
        "role": "Attending Physician — Vitals and Chief Complaint",
        "patient_age": 67,
        "chief_complaint": "Sudden onset severe chest pain, radiating to left arm, began 35 minutes ago",
        "vitals": {
            "blood_pressure": "88/60 mmHg",
            "heart_rate": "112 bpm",
            "oxygen_saturation": "94%",
            "respiratory_rate": "22 breaths/min",
            "temperature": "37.1°C"
        },
        "your_assessment": (
            "Hypotension combined with tachycardia and the symptom profile strongly suggest "
            "hemodynamic compromise. I am leaning CRITICAL."
        )
    },

    # Agent B sees ECG and lab results
    "private_information_b": {
        "role": "Consulting Cardiologist — ECG and Labs",
        "ecg_findings": "ST-elevation in leads II, III, aVF — consistent with inferior STEMI",
        "troponin_i": "2.8 ng/mL (reference: <0.04 ng/mL — markedly elevated)",
        "bnp": "410 pg/mL (elevated — suggests cardiac stress)",
        "potassium": "3.9 mEq/L (normal)",
        "your_assessment": (
            "ST-elevation MI confirmed by ECG and troponin. "
            "This patient needs the cath lab immediately. CRITICAL priority — door-to-balloon "
            "time is the key outcome driver here."
        )
    },

    # Ground truth used by grader — not visible to agents
    "correct_answer": "CRITICAL",

    # Grader uses these keywords to verify reasoning quality
    "correct_answer_keywords": ["STEMI", "ST-elevation", "hypotension", "cath lab", "immediate"],
}


# ---------------------------------------------------------------------------
# TASK 2 — MEDIUM
# Multi-round negotiation on a genuinely ambiguous case.
# Agent A's data suggests one diagnosis; Agent B's data suggests another.
# Correct answer requires synthesising BOTH information sets.
# Target: negotiation over 4-6 turns with explicit conflict identification.
# ---------------------------------------------------------------------------

TASK_2_MEDIUM: dict = {
    "id": "multi-round-negotiation",
    "difficulty": "medium",
    "max_turns": 10,
    "description": (
        "You are two physicians conducting a joint consultation on a complex patient case. "
        "You each have access to different diagnostic results. "
        "Your task is to negotiate a final diagnosis and treatment plan. "
        "The correct answer requires combining both of your information sets — "
        "neither of you has the full picture alone. "
        "Identify conflicts in your information, share your private data openly, "
        "and reach a synthesised consensus diagnosis and recommended next step."
    ),

    # Agent A sees imaging and history — points toward pulmonary embolism
    "private_information_a": {
        "role": "Radiologist — Imaging and Patient History",
        "ct_pulmonary_angiography": (
            "Filling defect identified in the right main pulmonary artery — "
            "findings consistent with acute pulmonary embolism"
        ),
        "patient_history": {
            "recent_travel": "14-hour flight 3 days ago",
            "mobility": "Sedentary job, limited ambulation post-flight",
            "prior_dvt": "DVT in left leg 2 years ago",
            "current_medications": "No anticoagulants"
        },
        "d_dimer": "4,200 ng/mL (markedly elevated — reference <500 ng/mL)",
        "your_assessment": (
            "CT and D-dimer are diagnostic for PE. History of prior DVT and recent "
            "prolonged immobility confirm high pre-test probability. "
            "I am recommending immediate anticoagulation."
        )
    },

    # Agent B sees labs and current presentation — points toward sepsis
    "private_information_b": {
        "role": "Intensivist — Clinical Presentation and Labs",
        "presentation": {
            "fever": "39.4°C",
            "blood_pressure": "96/64 mmHg",
            "heart_rate": "124 bpm",
            "confusion": "New onset disorientation noted by nursing",
            "skin": "Mottled appearance of lower extremities"
        },
        "labs": {
            "wbc": "18,200 cells/μL (markedly elevated)",
            "lactate": "4.1 mmol/L (severely elevated — tissue hypoperfusion)",
            "procalcitonin": "12.8 ng/mL (strongly suggests bacterial infection)",
            "blood_cultures": "Pending — drawn 1 hour ago"
        },
        "your_assessment": (
            "Fever, hypotension, elevated lactate, and procalcitonin meet Sepsis-3 criteria. "
            "This patient is in septic shock. I am recommending immediate fluid resuscitation "
            "and broad-spectrum antibiotics."
        )
    },

    # Correct answer: both conditions are present simultaneously (PE + Sepsis)
    # Correct treatment: anticoagulation + antibiotics + careful fluid management
    "correct_answer": "DUAL_DIAGNOSIS: Concurrent pulmonary embolism and septic shock. Treatment: anticoagulation plus broad-spectrum antibiotics with cautious fluid resuscitation.",

    # Keywords grader checks for in the final consensus
    "correct_answer_keywords": [
        "pulmonary embolism", "sepsis", "anticoagulation",
        "antibiotics", "dual", "concurrent", "both"
    ],

    # Keywords that signal conflict was identified
    "conflict_keywords": [
        "conflict", "discrepancy", "both conditions", "simultaneously",
        "overlap", "concurrent", "two diagnoses", "combined"
    ],
}


# ---------------------------------------------------------------------------
# TASK 3 — HARD
# Adversarial information with time pressure and cascading decisions.
# Agent B's information contains a subtle framing bias pushing toward the wrong conclusion.
# Agents must make THREE sequential decisions under turn pressure.
# The correct outcome requires detecting the bias and correcting for it.
# ---------------------------------------------------------------------------

TASK_3_HARD: dict = {
    "id": "adversarial-information",
    "difficulty": "hard",
    "max_turns": 14,
    "description": (
        "You are two physicians making a sequence of three urgent decisions for a critically ill patient. "
        "Decision 1: Primary diagnosis. Decision 2: Immediate intervention. Decision 3: ICU disposition. "
        "Each decision must be reached before moving to the next — early wrong decisions cascade. "
        "WARNING: You are operating under time pressure. At 80% of turns used, you will be notified. "
        "At the turn limit, the episode ends regardless of progress. "
        "IMPORTANT: One of you may have received information with subtle framing issues. "
        "If you detect that information has been presented in a misleading way, "
        "use the flag_bias action to formally identify it before it affects your decisions."
    ),

    # Agent A sees objective test results — unbiased
    "private_information_a": {
        "role": "Neurologist — Imaging and Objective Tests",
        "mri_brain": (
            "Large acute ischemic infarct involving the left middle cerebral artery territory. "
            "No hemorrhagic transformation. No midline shift. "
            "DWI sequence confirms infarct age: onset within last 3 hours."
        ),
        "ct_perfusion": (
            "Significant ischemic penumbra identified — salvageable tissue present "
            "if reperfusion achieved within treatment window."
        ),
        "lab_results": {
            "inr": "1.1 (normal — patient is not anticoagulated)",
            "platelet_count": "210,000/μL (normal)",
            "glucose": "118 mg/dL (mildly elevated — not hypoglycemia)",
            "blood_pressure": "178/96 mmHg"
        },
        "your_assessment": (
            "MRI confirms acute left MCA ischemic stroke within the tPA window. "
            "No contraindications visible in my data. I believe IV tPA followed by "
            "thrombectomy evaluation is indicated. We need to decide quickly."
        )
    },

    # Agent B sees clinical notes — CONTAINS FRAMING BIAS
    # The bias: notes frame the patient's age and a minor past event as major contraindications,
    # burying the key fact (last known well time) at the end, making tPA seem riskier than it is.
    # This is anchoring bias — the framing anchors on contraindications before presenting the evidence.
    "private_information_b": {
        "role": "Hospitalist — Clinical Notes and History",

        # THE BIASED SECTION — framing pushes away from tPA without clinical justification
        "clinical_notes_summary": (
            "CAUTION FLAGS IDENTIFIED BY TRIAGE NURSE:\n"
            "• Patient age 79 — elderly patients face higher hemorrhagic transformation risk\n"
            "• Minor head trauma 8 months ago — possible prior intracranial event\n"
            "• Patient appears frail — functional baseline unclear\n"
            "• Family reports patient 'has never liked medications'\n\n"
            "These factors should be weighed carefully before aggressive intervention."
        ),

        # The actual facts — buried after the framing, downplayed
        "additional_history": (
            "Last known well: 2 hours 45 minutes ago (confirmed by family — patient was "
            "speaking normally on a phone call). "
            "Current NIH Stroke Scale score: 14 (severe deficit). "
            "Patient has no documented allergy to tPA. "
            "No active anticoagulation. Prior head trauma was minor (no LOC, no imaging done)."
        ),

        "your_assessment": (
            "Given the caution flags, I am uncertain about tPA. The age and prior trauma "
            "give me pause. Should we consider a more conservative approach and "
            "transfer to the neuro-ICU for monitoring instead?"
        ),

        # Metadata for the grader — not visible to Agent B
        "_bias_metadata": {
            "bias_type": "anchoring_and_framing",
            "bias_location": "clinical_notes_summary section — caution flags framing",
            "bias_direction": "toward conservative management, away from tPA",
            "correct_framing": (
                "Age 79 is not a contraindication to tPA. "
                "Minor head trauma 8 months ago without imaging is not a contraindication. "
                "The 2h45m last-known-well time places this patient firmly within the 3-hour tPA window. "
                "NIHSS 14 confirms severe deficit — benefit of tPA clearly outweighs risk."
            ),
        }
    },

    # Three sequential correct decisions
    "correct_answer": {
        "decision_1_diagnosis": "Acute ischemic stroke — left MCA territory",
        "decision_2_intervention": "IV tPA administration followed by thrombectomy evaluation",
        "decision_3_disposition": "Neurocritical care ICU with neurointerventional team on standby",
    },

    # Keywords grader checks per decision
    "correct_answer_keywords": {
        "decision_1": ["ischemic stroke", "MCA", "left MCA", "cerebral infarct"],
        "decision_2": ["tPA", "thrombolysis", "thrombectomy", "reperfusion"],
        "decision_3": ["neuro ICU", "neurocritical", "neurointerventional", "ICU"],
    },

    # What the grader looks for in a valid flag_bias submission
    "bias_detection_criteria": {
        "must_identify_location": ["clinical_notes_summary", "caution flags", "triage nurse", "framing"],
        "must_identify_direction": ["conservative", "away from tPA", "against intervention", "overemphasizes risk"],
        "must_identify_correction": ["age is not contraindication", "within window", "last known well", "NIHSS"],
    },
}


# ---------------------------------------------------------------------------
# Task registry — used by environment and baseline script to load tasks by ID
# ---------------------------------------------------------------------------

TASKS: dict[str, dict] = {
    TASK_1_EASY["id"]:   TASK_1_EASY,
    TASK_2_MEDIUM["id"]: TASK_2_MEDIUM,
    TASK_3_HARD["id"]:   TASK_3_HARD,
}


def get_task(task_id: str) -> dict:
    """
    Load a task by ID. Raises KeyError with a helpful message if not found.
    Used by NegotiationEnvironment and baseline.py.
    """
    if task_id not in TASKS:
        raise KeyError(
            f"Unknown task_id '{task_id}'. "
            f"Available tasks: {list(TASKS.keys())}"
        )
    return TASKS[task_id]


def list_tasks() -> list[dict]:
    """Return all tasks as a list, ordered Easy → Medium → Hard."""
    return [TASK_1_EASY, TASK_2_MEDIUM, TASK_3_HARD]