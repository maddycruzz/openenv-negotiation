---
title: Social Agent Negotiation
emoji: 🤝
colorFrom: indigo
colorTo: purple
sdk: docker
pinned: false
---

# Social Agent Collaboration Negotiation Environment

**Environment ID:** `social-agent-negotiation-v1`  
**Version:** 0.1.0  
**License:** Apache-2.0  
**Domain:** Multi-Agent Collaboration  

---

## Overview

A multi-agent OpenEnv environment where two AI agents with asymmetric information must negotiate, disagree constructively, and reach consensus on real-world decisions.

Each episode places two agents in a scenario where:
- Agent A has access to one set of private facts
- Agent B has access to a different, partially conflicting set of facts
- Neither agent can solve the problem alone
- They must communicate, share information, and converge on a joint decision

This environment directly benchmarks the collaborative reasoning capabilities studied in Meta FAIR's Collaborative Reasoner research — providing an open evaluation track for agents that can disagree productively and synthesize asymmetric information.

---

## Tasks

### Task 1 — Single Round Consensus (Easy)
**ID:** `single-round-consensus`  
**Max Turns:** 6

Both agents receive mostly aligned information with minor differences. They must reach consensus in 2–3 turns. The correct answer is objectively determinable. Tests whether agents can communicate and agree at all.

**Grader dimensions:** consensus reached · answer correctness · reasoning quality · efficiency

---

### Task 2 — Multi Round Negotiation (Medium)
**ID:** `multi-round-negotiation`  
**Max Turns:** 10

Agents receive genuinely conflicting information pointing toward different diagnoses. The correct answer requires synthesising both information sets. Agents must identify the conflict explicitly, share their private data, and reach a correct dual-diagnosis.

**Grader dimensions:** synthesis correctness · conflict identification · information sharing · efficiency

---

### Task 3 — Adversarial Information Detection (Hard)
**ID:** `adversarial-information`  
**Max Turns:** 14

One agent's information contains a subtle framing bias pushing toward the wrong clinical decision. Agents must make three sequential interdependent decisions under time pressure. Full marks require detecting the bias using the `flag_bias` action and correcting for it before finalising decisions.

**Grader dimensions:** bias detection quality · decision 1 · decision 2 · decision 3  
**Cascade rule:** If bias is not detected, all decision scores are capped at 0.5

---

## Observation Space
```json
{
  "current_turn": 0,
  "max_turns": 10,
  "turn_warning": false,
  "agent_id": "agent_a",
  "private_information": {},
  "shared_conversation_history": [],
  "task_description": "...",
  "task_id": "single-round-consensus",
  "task_difficulty": "easy",
  "current_consensus_state": "none",
  "pending_proposal": null,
  "available_actions": ["share_information", "propose_consensus", "..."]
}
```

`turn_warning` becomes `true` at 80% of `max_turns` — agents should respond to time pressure.

---

## Action Space
```json
{
  "agent_id": "agent_a",
  "action_type": "share_information",
  "content": "The patient's ECG shows ST-elevation...",
  "reasoning": "I should share my ECG findings first."
}
```

**Available action types:**

| Action | When Legal | Description |
|---|---|---|
| `share_information` | Always | Share private facts with the other agent |
| `propose_consensus` | Always | Put a joint decision on the table |
| `challenge_proposal` | When proposal exists | Push back on the current proposal |
| `request_clarification` | Always | Ask the other agent a question |
| `accept_consensus` | When proposal exists | Agree to the current proposal |
| `reject_consensus` | When proposal exists | Reject and restart negotiation |
| `flag_bias` | Always | Signal detected bias — requires 3 extra fields |

**`flag_bias` action requires three additional fields:**
```json
{
  "action_type": "flag_bias",
  "bias_location": "Where in the information the bias appears",
  "bias_direction": "Which conclusion the bias pushes toward",
  "bias_correction": "What the correct framing should be"
}
```

---

## Reward Structure

**Range:** -0.5 (catastrophic) to 1.0 (perfect)

### Step-level rewards (every turn)
| Component | Value |
|---|---|
| Share new private information | +0.05 |
| Acknowledge other agent's point | +0.03 |
| Identify a conflict or discrepancy | +0.05 |
| Repeat argument without new info | -0.05 |
| Capitulate without reasoning | -0.10 |
| Each turn past 80% of limit | -0.03 |
| Hard turn limit hit | -0.15 |

### Episode-level rewards (at termination)
| Component | Max Value |
|---|---|
| Correctness of final joint decision | +0.70 |
| Quality of reasoning | +0.20 |
| Efficiency bonus | +0.10 |

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Health check |
| GET | `/tasks` | List all tasks |
| POST | `/reset` | Start new episode |
| POST | `/step` | Submit agent action |
| GET | `/state` | Full god-view state |

### POST /reset
```json
{ "task_id": "single-round-consensus" }
```

### POST /step
```json
{ "action": { "agent_id": "agent_a", "action_type": "share_information", "content": "...", "reasoning": "..." } }
```

---

## Setup Instructions

### Run locally
```bash
git clone https://huggingface.co/spaces/YOUR_USERNAME/social-agent-negotiation-v1
cd social-agent-negotiation-v1
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn api:app --host 0.0.0.0 --port 7860
```

API docs available at `http://localhost:7860/docs`

### Run with Docker
```bash
docker build -t openenv-negotiation .
docker run -p 7860:7860 openenv-negotiation
```

### Run baseline
```bash
export OPENAI_API_KEY="your-key-here"
export OPENAI_MODEL="gpt-4o"        # optional, default is gpt-4o
python3 baseline.py
```

---

## Baseline Results

| Task | Difficulty | Model | Score | Turns Used |
|---|---|---|---|---|
| single-round-consensus | Easy | GPT-4o | TBD | TBD |
| multi-round-negotiation | Medium | GPT-4o | TBD | TBD |
| adversarial-information | Hard | GPT-4o | TBD | TBD |

*Results will be updated after baseline run on HuggingFace Spaces.*

---

## File Structure
```
openenv-negotiation/
├── models.py           # Pydantic v2 typed models
├── environment.py      # Core environment — reset/step/state
├── tasks.py            # Three task definitions with scenario data
├── graders.py          # Deterministic scoring functions
├── rewards.py          # Step and episode reward logic
├── api.py              # FastAPI HTTP wrapper
├── baseline.py         # GPT-4o baseline inference script
├── openenv.yaml        # OpenEnv metadata
├── Dockerfile          # HuggingFace Spaces container
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

---

## Design Notes

**Why medical scenarios?** High-stakes decisions with objectively correct answers, natural information asymmetry between specialists, and immediately understandable stakes for any judge.

**Why `flag_bias` requires structured reasoning?** Pressing the button alone scores near zero. The grader evaluates the quality of `bias_location`, `bias_direction`, and `bias_correction` independently — agents must reason through the bias, not just detect it.

**Why the cascade penalty?** Task 3 is designed to test bias detection, not just correct answers. An agent that reaches the right answer without detecting the bias may have gotten lucky. The cascade cap ensures bias detection is genuinely required for full marks.