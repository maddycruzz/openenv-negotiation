from dotenv import load_dotenv
load_dotenv()

import os
import sys
import json
import time
import requests
from typing import Dict, Any, Tuple, Optional, List

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
HF_TOKEN  = os.getenv("HF_TOKEN")
ENV_URL   = os.getenv("ENV_URL", "https://Bharath-1608-social-agent-negotiation-v1.hf.space")

BASE_MODEL    = "unsloth/Llama-3.2-1B-Instruct"
ADAPTER_REPO  = "Bharath-1608/negotiation-agent-grpo"
MODEL_NAME    = ADAPTER_REPO   # used in log lines
BENCHMARK     = "social-agent-negotiation"
MAX_NEW_TOKENS = 512

# Baseline scores from Groq llama-3.3-70b-versatile (stored for comparison)
BASELINE_SCORES = {
    "single-round-consensus": 1.1083,
    "adversarial-information": 0.4764,
    "opioid-overdose": 0.4764,
}

EVAL_TASKS = ["single-round-consensus", "adversarial-information", "opioid-overdose"]

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


# ---------------------------------------------------------------------------
# Model loading  (unsloth 4-bit + PEFT LoRA adapter)
# ---------------------------------------------------------------------------

def load_model():
    """Load base model with 4-bit quantisation, then apply the trained LoRA."""
    print(f"Loading base model: {BASE_MODEL} (4-bit) ...", flush=True)
    try:
        from unsloth import FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=BASE_MODEL,
            max_seq_length=2048,
            load_in_4bit=True,
            token=HF_TOKEN,
        )
        FastLanguageModel.for_inference(model)
    except ImportError:
        # Fallback: plain transformers + bitsandbytes if unsloth not installed
        print("  unsloth not found — falling back to transformers BitsAndBytes.", flush=True)
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        import torch
        bnb_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, token=HF_TOKEN)
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, quantization_config=bnb_cfg, device_map="auto", token=HF_TOKEN
        )

    print(f"Loading LoRA adapter: {ADAPTER_REPO} ...", flush=True)
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, ADAPTER_REPO, token=HF_TOKEN)
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Model ready.\n", flush=True)
    return model, tokenizer


# Global model handle — loaded once at startup.
_model = None
_tokenizer = None


def get_model():
    global _model, _tokenizer
    if _model is None:
        _model, _tokenizer = load_model()
    return _model, _tokenizer


# ---------------------------------------------------------------------------
# System prompt
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
# Action generation — local inference
# ---------------------------------------------------------------------------

def _build_chat_input(agent_id: str, observation: Dict[str, Any]) -> str:
    """Format a chat-template prompt string for the model."""
    model, tokenizer = get_model()
    messages = [
        {"role": "system", "content": get_system_prompt(agent_id)},
        {"role": "user",   "content": json.dumps(observation)},
    ]
    # apply_chat_template with add_generation_prompt so the model continues from here
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def _extract_json(text: str) -> Dict[str, Any]:
    """Pull the first valid JSON object out of raw model output."""
    # Try whole text first
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass
    # Find the outermost { ... } block
    start = text.find("{")
    end   = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass
    raise ValueError(f"No valid JSON found in model output: {text[:200]}")


def generate_agent_action(
    agent_id: str,
    observation: Dict[str, Any],
    retry_count: int = 0,
) -> Dict[str, Any]:
    """Generate an action from the local model. Retries once on parse failure."""
    model, tokenizer = get_model()
    import torch

    prompt = _build_chat_input(agent_id, observation)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    try:
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=0.3,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        # Decode only the newly generated tokens
        new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        raw_text = tokenizer.decode(new_ids, skip_special_tokens=True)

        action = _extract_json(raw_text)
        action["agent_id"] = agent_id
        return action

    except (json.JSONDecodeError, ValueError) as e:
        if retry_count < 1:
            print(f"  [{agent_id}] JSON parse error — retrying. ({e})", flush=True)
            return generate_agent_action(agent_id, observation, retry_count + 1)
        print(f"  [{agent_id}] JSON parse failed after retry — using fallback.", flush=True)
        return {
            "agent_id":    agent_id,
            "action_type": "request_clarification",
            "content":     "I encountered an internal error and must skip my turn.",
            "reasoning":   "Fallback action due to JSON parse failure.",
        }

    except Exception as e:
        print(f"  [{agent_id}] Generation error: {e}", flush=True)
        if retry_count < 1:
            return generate_agent_action(agent_id, observation, retry_count + 1)
        return {
            "agent_id":    agent_id,
            "action_type": "request_clarification",
            "content":     "Generation error fallback.",
            "reasoning":   "Fallback action due to generation failure.",
        }


# ---------------------------------------------------------------------------
# Structured logging — identical format required by checker
# ---------------------------------------------------------------------------

def _safe(v: float) -> float:
    return float(round(max(0.0, min(1.0, v)), 4))


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
    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        reset_resp = requests.post(f"{ENV_URL}/reset", json={"task_id": task_id}, timeout=30)
        reset_resp.raise_for_status()
        state = reset_resp.json()
    except Exception as e:
        print(f"  Reset failed: {e}", flush=True)
        log_end(success=False, steps=0, score=0.0, rewards=[0.0])
        return None, 0, 0.0

    session_id    = state.get("session_id")
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

        print(f"  Turn {turn_count}: {active_id} thinking...", flush=True)
        action     = generate_agent_action(active_id, obs)
        action_str = json.dumps(action)
        print(f"  -> {active_id}: {action.get('action_type')}", flush=True)

        step_reward = 0.05
        error_msg   = None

        try:
            step_payload = {"action": action}
            if session_id:
                step_payload["session_id"] = session_id

            step_resp = requests.post(f"{ENV_URL}/step", json=step_payload, timeout=30)

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
        time.sleep(0.1)

    score   = 0.0
    success = False
    if final_result:
        score   = _safe(float(final_result.get("total_reward", 0.0)))
        success = final_result.get("final_consensus") == "reached"

    log_end(success=success, steps=turn_count, score=score, rewards=rewards_list)
    return final_result, turn_count, score


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"\n=== Social Agent Negotiation — Trained Model Inference ===")
    print(f"Base  : {BASE_MODEL}")
    print(f"Adapter: {ADAPTER_REPO}")
    print(f"Env   : {ENV_URL}\n")

    # Eagerly load model so startup errors surface before any episodes run
    get_model()

    # Health check
    print("Checking environment health...")
    try:
        h = requests.get(f"{ENV_URL}/health", timeout=10)
        h.raise_for_status()
        print("Environment is healthy.\n")
    except Exception as e:
        print(f"ERROR: Cannot connect to environment at {ENV_URL}: {e}")
        sys.exit(1)

    results = []
    for task_id in EVAL_TASKS:
        print(f"\n{'='*60}")
        print(f"Task: {task_id}")
        print(f"{'='*60}")
        episode_result, turns_used, score = run_episode(task_id)

        consensus_reached = False
        if episode_result:
            consensus_reached = episode_result.get("final_consensus") == "reached"

        results.append({
            "task_id":           task_id,
            "score":             score,
            "turns_used":        turns_used,
            "consensus_reached": consensus_reached,
            "raw_episode_result": episode_result,
        })

    # Save results
    with open("inference_results.json", "w") as f:
        json.dump(results, f, indent=4)
    print("\nSaved results to inference_results.json")

    # Before / after comparison table
    print(f"\n{'Task':<30} | {'Baseline (Groq 70B)':<20} | {'Trained 1B':<12} | {'Delta':<8} | Consensus")
    print("-" * 90)
    for r in results:
        tid      = r["task_id"]
        baseline = BASELINE_SCORES.get(tid, float("nan"))
        trained  = r["score"]
        delta    = trained - baseline if baseline == baseline else 0.0
        sign     = "+" if delta >= 0 else ""
        print(
            f"{tid[:29]:<30} | "
            f"{baseline:<20.4f} | "
            f"{trained:<12.4f} | "
            f"{sign}{delta:<7.4f} | "
            f"{'Yes' if r['consensus_reached'] else 'No'}"
        )

    avg_trained  = sum(r["score"] for r in results) / len(results) if results else 0
    avg_baseline = sum(BASELINE_SCORES.get(r["task_id"], 0) for r in results) / len(results) if results else 0
    print(f"\nAverage — Baseline: {avg_baseline:.4f}  |  Trained: {avg_trained:.4f}  |  Delta: {'+' if avg_trained >= avg_baseline else ''}{avg_trained - avg_baseline:.4f}")


if __name__ == "__main__":
    main()
