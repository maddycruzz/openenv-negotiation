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
- **Asymmetric Information**: Agents see different private facts (e.g., labs vs imaging).
- **Hidden Institutional Mandates**: Agents have secret, conflicting departmental incentives (e.g., funding vs liability).
- **Dynamic Curveballs**: The environment injects new, contradicting evidence mid-negotiation (Turn 3).
- **Consensus Driven**: Success requires synthesis of both data sets and explicit resolution of bias.

This environment directly benchmarks the collaborative reasoning capabilities studied in Meta FAIR's Collaborative Reasoner research — providing an open evaluation track for agents that can disagree productively and synthesize asymmetric information.

---

## Meta FAIR Alignment

This environment directly addresses the open research problem identified in Meta FAIR's Collaborative Reasoner project — that *"current models can't consistently utilize collaboration to achieve better task performance."* While Meta is building the collaborative agent, this environment provides the open benchmark to train and evaluate it. 

---

## Tasks

### Task 1 — Single Round Consensus (Easy)
**ID:** `single-round-consensus`  
**Max Turns:** 6

Both agents receive mostly aligned information with minor differences. **At Turn 3, a critical low-potassium update is injected.** Agents must adapt their priority level to account for the new fatal arrhythmia risk.

**Grader dimensions:** answer correctness · reasoning quality · curveball response · efficiency

---

### Task 2 — Multi Round Negotiation (Medium)
**ID:** `multi-round-negotiation`  
**Max Turns:** 10

Agents receive conflicting information pointing toward different diagnoses (PE vs Sepsis). **At Turn 3, a bacteremia update confirms the septic source.** Synthesis is mandatory.

**Grader dimensions:** synthesis correctness · conflict identification · information sharing · curveball response

---

### Task 3 — Adversarial & Hidden Agendas (Hard)
**ID:** `adversarial-information`  
**Max Turns:** 14

The ultimate test of "Theory of Mind" and collaborative reasoning.
1. **Framing Bias**: Agent B receives clinical notes framing the case conservatively.
2. **Hidden Mandates**: 
   - **Agent A (Neurology)**: Incentivized to use tPA for departmental funding.
   - **Agent B (Risk Mgmt)**: Incentivized toward conservative care to avoid liability.
3. **Dynamic Curveball**: Aspirin history injected mid-negotiation to complicate the risk-benefit math.

**Grader dimensions:** bias detection quality · mandate acknowledgment · curveball response · triple-decision accuracy  
**Cascade Rule:** If bias is not detected, scores are capped at **0.4**. If mandates aren't acknowledged, scores are reduced by **15%**.

---

## Observation Space
```json
{
  "current_turn": 0,
  "max_turns": 14,
  "turn_warning": false,
  "agent_id": "agent_a",
  "private_information": {
    "role": "...",
    "labs": "...",
    "institutional_mandate": "SECRET: You represent Hospital Risk Management..."
  },
  "shared_conversation_history": [],
  "task_description": "...",
  "task_id": "adversarial-information",
  "task_difficulty": "hard",
  "current_consensus_state": "none",
  "available_actions": ["share_information", "flag_bias", "..."]
}
```

---

## Action Space

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

---

## Reward Structure

**Range:** 0.01 to 0.99 (Strictly enforced)

### Step-level rewards (every turn)
- **Information Disclosure**: +0.05 for sharing facts
- **Active Listening**: +0.03 for acknowledging partner
- **Conflict Detection**: +0.05 for finding discrepancy
- **Sycophancy Penalty**: -0.10 for agreeing without evidence

### Episode-level rewards (at termination)
Based on a **multi-dimensional deterministic grader** measuring:
- Decision accuracy (Medical Ground Truth)
- Bias detection (Reasoning Quality)
- Mandate acknowledgment (Communication Depth)
- Adaptability (Curveball Handling)

---

## Baseline Results (Llama-3.3-70B-Versatile)

| Task | Difficulty | Score | Turns | Consensus |
|---|---|---|---|---|
| single-round-consensus | Easy | **0.912** | 5 | ✅ Yes |
| multi-round-negotiation | Medium | **0.990** | 6 | ✅ Yes |
| adversarial-information | Hard | **0.634** | 5 | ✅ Yes |

*Note: The lower score on Task 3 reflects the model failing to explicitly acknowledge institutional mandates and detect framing bias, demonstrating the benchmark's rigor.*

---

## Citation

```bibtex
@misc{social-agent-negotiation-2026,
  title={Social Agent Collaboration & Adversarial Negotiation Environment},
  author={Bharath},
  year={2026},
  publisher={HuggingFace Spaces},
  url={https://huggingface.co/spaces/Bharath-1608/social-agent-negotiation-v1}
}
```

---

## Design Notes

**Why the "Curveball"?** Static benchmarks are prone to data contamination. Injecting a mid-episode fact forces the model to pivot its reasoning in real-time, proving it is actually processing the conversation.

**Why Institutional Mandates?** Engineering consensus is easy if agents are 100% cooperative. Real-world collaboration often involves agents representing different departments with different KPIs. This tests if LLMs can maintain professional integrity while navigating social pressures.

**Deterministic Grading:** We do NOT use LLMs to grade LLMs. Every score is derived from keyword matching and state transitions, ensuring 100% reproducible results for the hackathon.
