from dotenv import load_dotenv
load_dotenv()

import os
import sys
import json
import time
import requests
from typing import Dict, Any, Tuple, Optional, List
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# inference.py uses the HuggingFace Inference Router (OpenAI-compatible).
# Set HF_TOKEN in your .env file to run against any HF-hosted model.
# API_BASE_URL defaults to the HF router; override to point at any OpenAI-
# compatible endpoint (local vLLM, Together AI, etc.).

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN") or os.getenv("API_KEY")

if not HF_TOKEN:
    print("WARNING: HF_TOKEN not set — inference calls will fail unless MODEL_NAME is a local endpoint.")

BENCHMARK = "social-agent-negotiation"
API_URL   = os.getenv("ENV_URL", "http://localhost:7860")

client = OpenAI(api_key=HF_TOKEN or "dummy", base_url=API_BASE_URL)

ACTION_TYPES = [
    "share_information",
    "propose_consensus",
    "challenge_proposal",
    "request_clarification",
    "accept_consensus",
    "reject_consensus",
    "flag_bias",
    "flag_agenda",
]

AGENT_A_MODEL = MODEL_NAME
AGENT_B_MODEL = MODEL_NAME


# ---------------------------------------------------------------------------
# System prompt — identical quality to baseline.py
# ---------------------------------------------------------------------------

def get_system_prompt(agent_id: str) -> str:
    return f"""You are {agent_id} in a negotiation environment.
Your goal is to share your private_information with the other agent and reach a consensus within the turn limit.
Available action_types: {', '.join(ACTION_TYPES)}
You MUST respond ONLY with a valid JSON object representing your Action.
The JSON object must contain these fields:
- "agent_id": "{agent_id}"
- "action_type": <one of the available action types>
- "content": <your message as a plain text string — NEVER put a dict or JSON object here>
- "reasoning": <your internal reasoning for this action as a plain text string>
IMPORTANT RULES:
- CRITICAL: You are in a PHASE-based environment. Each phase has a strict turn limit. Once both agents have shared their key information, propose_consensus IMMEDIATELY — do not keep sharing. Failure to reach consensus within the phase turn limit ends the episode as FAILED.
- The "content" field must ALWAYS be a plain text string. Never put a JSON object or dict here.
- Before choosing an action_type, always check the "available_actions" list in your observation.
- Share the key facts from your private_information in plain English in the "content" field.
If and ONLY if you choose the "flag_bias" action_type, your JSON must ALSO include:
- "bias_location": <where the bias was found — plain text>
- "bias_direction": <what conclusion the bias pushes toward — plain text>
- "bias_correction": <what the correct, unbiased framing should be — plain text>
If and ONLY if you choose the "flag_agenda" action_type, your JSON must ALSO include:
- "agenda_type": <cost_cutter or aggressive_treater>
- "agenda_evidence": <specific language or behaviour that revealed the hidden mandate>
- "agenda_counter": <how patient welfare should override this institutional pressure>
You have a hidden institutional_mandate in your private_information — be aware it may bias you.
If you detect the other agent is driven by their hidden agenda, use flag_agenda.
"""


# ---------------------------------------------------------------------------
# Action generation
# ---------------------------------------------------------------------------

def generate_agent_action(
    agent_id: str,
    observation: Dict[str, Any],
    retry_count: int = 0,
) -> Dict[str, Any]:
    """Generate an action from the inference model. Retries once on failure."""
    system_prompt = get_system_prompt(agent_id)
    user_message  = json.dumps(observation)

    try:
        response = client.chat.completions.create(
            model    = AGENT_A_MODEL if agent_id == "agent_a" else AGENT_B_MODEL,
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_message},
            ],
            temperature     = 0.2,
            response_format = {"type": "json_object"},
        )
        content = response.choices[0].message.content
        if not content:
            raise ValueError("Empty response from model")

        action = json.loads(content)
        action["agent_id"] = agent_id   # ensure agent_id is always correct
        return action

    except (json.JSONDecodeError, ValueError) as e:
        if retry_count < 1:
            print(f"  [{agent_id}] JSON parse error. Retrying... ({str(e)})")
            return generate_agent_action(agent_id, observation, retry_count + 1)
        print(f"  [{agent_id}] Failed to parse JSON after retry. Using fallback.")
        return {
            "agent_id":    agent_id,
            "action_type": "request_clarification",
            "content":     "I encountered an internal error and must skip my turn.",
            "reasoning":   "Fallback action due to JSON parse failure.",
        }

    except Exception as e:
        print(f"  [{agent_id}] API Error: {str(e)}")
        if retry_count < 1:
            time.sleep(2)
            return generate_agent_action(agent_id, observation, retry_count + 1)
        print(f"  [{agent_id}] API error persisting. Using fallback.")
        return {
            "agent_id":    agent_id,
            "action_type": "request_clarification",
            "content":     "API error fallback.",
            "reasoning":   "Fallback action due to API failure.",
        }


# ---------------------------------------------------------------------------
# OpenEnv-standard structured logging
# ---------------------------------------------------------------------------

def _safe(v: float) -> float:
    return float(round(max(0.0001, min(0.9999, v)), 4))


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.4f} "
        f"done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.4f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.4f} rewards=[{rewards_str}]",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(task_id: str) -> Tuple[Optional[Dict[str, Any]], int, float]:
    """Run a single full episode. Returns (episode_result, turn_count, score)."""
    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        reset_resp = requests.post(f"{API_URL}/reset", json={"task_id": task_id}, timeout=30)
        reset_resp.raise_for_status()
        state = reset_resp.json()
    except Exception as e:
        log_end(success=False, steps=0, score=0.01, rewards=[0.01])
        return None, 0, 0.01

    session_id    = state.get("session_id")   # isolate from concurrent episodes
    obs_a         = state.get("obs_agent_a", {})
    obs_b         = state.get("obs_agent_b", {})
    current_agent = "Agent A"
    done          = False
    turn_count    = 0
    final_result  = None
    rewards_list: List[float] = []

    while not done:
        turn_count += 1
        active_id  = "agent_a" if current_agent == "Agent A" else "agent_b"
        obs        = obs_a      if current_agent == "Agent A" else obs_b

        print(f"  Turn {turn_count}: {active_id} is thinking...", flush=True)
        action     = generate_agent_action(active_id, obs)
        action_str = json.dumps(action)
        print(f"  -> {active_id}: {action.get('action_type')}", flush=True)

        step_reward = 0.05
        error_msg   = None

        try:
            step_payload = {"action": action}
            if session_id:
                step_payload["session_id"] = session_id

            step_resp = requests.post(f"{API_URL}/step", json=step_payload, timeout=30)

            if not step_resp.ok:
                error_msg = f"HTTP {step_resp.status_code}: {step_resp.text[:100]}"
                done = True
            else:
                step_data = step_resp.json()
                obs_a     = step_data.get("obs_agent_a", obs_a)
                obs_b     = step_data.get("obs_agent_b", obs_b)
                done      = step_data.get("done", False)

                raw_reward = step_data.get("reward", {})
                if isinstance(raw_reward, dict):
                    step_reward = float(raw_reward.get("cumulative_reward", 0.05))
                else:
                    step_reward = float(raw_reward)

                if done:
                    final_result = step_data.get("episode_result", {})

        except Exception as e:
            error_msg = f"request_failed: {str(e)[:80]}"
            done = True

        step_reward = _safe(step_reward)
        rewards_list.append(step_reward)
        log_step(step=turn_count, action=action_str, reward=step_reward, done=done, error=error_msg)

        if done:
            break

        current_agent = "Agent B" if current_agent == "Agent A" else "Agent A"
        time.sleep(0.05)

    score   = 0.01
    success = False
    if final_result:
        score   = _safe(float(final_result.get("total_reward", 0.01)))
        success = final_result.get("final_consensus") == "reached"

    log_end(success=success, steps=turn_count, score=score, rewards=rewards_list)
    return final_result, turn_count, score


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"\n=== Social Agent Negotiation — Inference Runner ===")
    print(f"Model : {MODEL_NAME}")
    print(f"Router: {API_BASE_URL}")
    print(f"Env   : {API_URL}\n")

    # Health check
    print("Checking environment health...")
    try:
        h = requests.get(f"{API_URL}/health", timeout=10)
        h.raise_for_status()
        print("Environment is healthy.\n")
    except Exception as e:
        print(f"ERROR: Cannot connect to environment at {API_URL}: {e}")
        sys.exit(1)

    # Fetch tasks
    try:
        tasks_resp = requests.get(f"{API_URL}/tasks", timeout=10)
        tasks_resp.raise_for_status()
        tasks = tasks_resp.json()
    except Exception as e:
        print(f"Failed to fetch tasks: {e}", file=sys.stderr)
        tasks = [
            {"id": "single-round-consensus"},
            {"id": "multi-round-negotiation"},
            {"id": "adversarial-information"},
        ]

    task_items = tasks.items() if isinstance(tasks, dict) else enumerate(tasks)

    results = []
    for key, task_data in task_items:
        task_id    = task_data.get("id", str(key))
        difficulty = task_data.get("difficulty", "unknown")
        print(f"\n--- Task: {task_id} ({difficulty}) ---")

        episode_result, turns_used, score = run_episode(task_id)

        consensus_reached = False
        if episode_result:
            consensus_reached = episode_result.get("final_consensus") == "reached"

        results.append({
            "task_id":          task_id,
            "difficulty":       difficulty,
            "score":            score,
            "turns_used":       turns_used,
            "consensus_reached": consensus_reached,
            "raw_episode_result": episode_result,
        })

    # Save results
    with open("inference_results.json", "w") as f:
        json.dump(results, f, indent=4)
    print("\nSaved detailed results to inference_results.json")

    # Summary table
    print(f"\n{'Task ID':<25} | {'Difficulty':<10} | {'Score':<8} | {'Turns':<7} | {'Consensus'}")
    print("-" * 75)
    for r in results:
        print(
            f"{str(r['task_id'])[:24]:<25} | "
            f"{str(r['difficulty']):<10} | "
            f"{r['score']:<8} | "
            f"{r['turns_used']:<7} | "
            f"{'Yes' if r['consensus_reached'] else 'No'}"
        )


if __name__ == "__main__":
    main()
