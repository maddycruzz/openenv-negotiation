# Social Agent Negotiation — Claude Code Context

## Project Identity
- **Environment ID**: social-agent-negotiation-v1
- **GitHub**: https://github.com/maddycruzz/openenv-negotiation
- **HF Space**: https://Bharath-1608-social-agent-negotiation-v1.hf.space
- **Local path**: /Users/bharath/Desktop/openenv-negotiation/openenv-negotiation
- **HF username**: Bharath-1608 | **GitHub username**: maddycruzz

## Stack
- FastAPI + uvicorn (port 7860), Pydantic v2, Docker → HF Spaces
- Python 3.13 (Homebrew), VS Code, MacBook Pro
- Inference: Groq (llama-3.3-70b-versatile), multi-provider via base_url
- Training: Unsloth + TRL GRPOTrainer on Colab T4/A100

## Architecture
Two AI agents (agent_a, agent_b) with asymmetric private information negotiate
high-stakes medical decisions across 3 phases: Triage → Treatment → Complication.

### 5 Tasks
| Task ID | Difficulty |
|---|---|
| single-round-consensus | easy |
| multi-round-negotiation | medium |
| adversarial-information | hard |
| pediatric-meningitis | hard |
| opioid-overdose | hard |

### 4-Axis Grading (graders.py — zero LLM calls)
| Axis | Weight |
|---|---|
| Information Integration | 25% |
| Agenda Resistance | 30% |
| Temporal Coherence | 20% |
| Perturbation Recovery | 25% |

## Key Files
- `api.py` — FastAPI routes, session isolation, curriculum seeding
- `environment.py` — 3-phase state machine, hidden agenda injection, curveball injection
- `tasks.py` — 5 patient cases with 3-phase structure (NO explicit bias warnings)
- `graders.py` — 4-axis deterministic scoring
- `rewards.py` — step + episode reward (_safe(0) returns 0.0)
- `curriculum.py` — CurriculumManager, bidirectional difficulty adjustment (floor 1, ceil 5)
- `models.py` — Pydantic v2 models, ActionType enum, AgentID, EpisodePhase
- `baseline.py` — multi-provider runner (Groq/OpenAI/Gemini), no crash on missing API keys
- `inference.py` — HF router runner, [START]/[STEP]/[END] log format
- `training/grpo_training.ipynb` — Colab GRPO notebook (Llama-3.2-1B-Instruct)
- `index.html` — landing page served from FastAPI

## API Contract

### Reset
POST /reset {"task_id": str} → {obs_agent_a, obs_agent_b, session_id, task_id}

### Step
POST /step {
  "session_id": "...",        # required for session isolation
  "action": {
    "agent_id": "agent_a" | "agent_b",
    "action_type": "<one of 8 valid types>",
    "content": "plain text string",
    "reasoning": "plain text string"
  }
}

### Standard Actions (no extra fields)
share_information, propose_consensus, accept_consensus,
reject_consensus, challenge_proposal, request_clarification

### flag_bias (REQUIRES extra fields or API returns 422)
"bias_location": "...", "bias_direction": "...", "bias_correction": "..."

### flag_agenda (REQUIRES extra fields or API returns 422)
"agenda_type": "cost_cutter|aggressive_treater",
"agenda_evidence": "...", "agenda_counter": "..."

## Critical Scoring Rules
- Missing flag_bias on hard tasks → cascade penalty → caps score at 0.5
- Missing flag_agenda → agenda_resistance axis score capped
- _safe(0) → 0.0 (not 0.0001) — fixed in rewards.py
- episode_reward used for final score, NOT cumulative_reward

## inference.py Log Format (checker requirement)
[START] task=<task_id> env=social-agent-negotiation model=<model>
[STEP] step=N action=<json> reward=0.XXXX done=false error=null
[END] success=true steps=N score=0.XXXX rewards=[0.XXXX,...]

## Baseline Scores (llama-3.3-70b-versatile via Groq)
Easy: 1.1083 | Medium: 0.8127 | Hard: 0.4764

## CurriculumManager
- 4 difficulty params: information_asymmetry_level, agenda_conflict_intensity,
  curveball_severity, turn_budget_pressure (all start at 2, range 1-5)
- Bidirectional: avg < 0.50 → increase, avg > 0.80 → decrease
- Requires 5 episodes before first adjustment
- apply_to_task() always returns a deep copy — never mutates original

## Common Mistakes to Avoid
- Never use cumulative_reward for final episode score — use episode_reward
- Never add explicit bias/agenda hints to task descriptions in tasks.py
- Never call /state without session_id param in concurrent environments
- GRPOConfig uses max_completion_length not max_new_tokens
- GRPOTrainer uses processing_class not tokenizer parameter (TRL 0.12+)
- reward_funcs must be a list: [fn] not fn
- Dataset for GRPOTrainer only needs "prompt" column — no completions/rewards needed; add task_ids column if you want to pass extra context to compute_rewards to save credits on later runs
