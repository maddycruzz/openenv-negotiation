# =============================================================================
# Social Agent Negotiation — GRPO Training Script
# =============================================================================
# Target hardware : HuggingFace Spaces A10G (24 GB VRAM)
# Model           : unsloth/Qwen2.5-14B-Instruct-bnb-4bit
# Budget ceiling  : $30  (A10G @ $1.05/hr — hard 2.5-hr wall enforced below)
#
# Usage:
#   Set HF_TOKEN environment variable, then:
#   python training/train.py
# =============================================================================

import os
import sys
import json
import re
import time
import random
import requests
import traceback
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Optional

from trl import GRPOTrainer, GRPOConfig
from huggingface_hub import login

# ---------------------------------------------------------------------------
# Unsloth is mandatory — 14B will NOT fit on A10G without 4-bit + Unsloth
# ---------------------------------------------------------------------------
try:
    from unsloth import FastLanguageModel
except ImportError:
    print("ERROR: unsloth is not installed. 14B model will not fit on A10G without it.")
    print("       Install with: pip install unsloth")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_URL           = "https://Bharath-1608-social-agent-negotiation-v1.hf.space"
MODEL_NAME        = "unsloth/Qwen2.5-14B-Instruct-bnb-4bit"
CHECKPOINT_DIR    = "./checkpoints"
HF_REPO_ID        = "Bharath-1608/negotiation-agent-grpo"
REWARD_CURVE_PATH = "reward_curve.png"

TASK_IDS = [
    "single-round-consensus",
    "multi-round-negotiation",
    "adversarial-information",
    "pediatric-meningitis",
    "opioid-overdose",
]

TRAINING_ROUNDS       = 2
EPISODES_PER_TASK     = 6
MAX_TURNS_PER_EPISODE = 20
API_TIMEOUT           = 30
MAX_NEW_TOKENS        = 512
MAX_SEQ_LENGTH        = 2048

TIME_LIMIT_SECONDS    = 2.5 * 3600   # 2.5 hours → $2.63 on A10G

MEDICAL_KEYWORDS = [
    "patient", "diagnosis", "treatment", "symptoms", "vital", "blood pressure",
    "heart rate", "ecg", "troponin", "oxygen", "saturation", "critical",
    "urgent", "triage", "medication", "dosage", "clinical", "prognosis",
    "assessment", "evidence", "imaging", "labs", "antibiotic", "heparin",
    "tpa", "naloxone", "meningitis", "sepsis", "embolism", "stroke",
]


# =============================================================================
# HuggingFace Auth
# =============================================================================

def setup_hf_auth() -> str:
    """Login using HF_TOKEN environment variable (HF Spaces Docker — no Colab)."""
    hf_token = os.environ.get("HF_TOKEN", "").strip()
    if not hf_token:
        print("WARNING: HF_TOKEN not set — Hub push will be skipped at the end.")
        return ""
    login(token=hf_token)
    print("HF_TOKEN loaded from environment, logged in to HuggingFace Hub.")
    return hf_token


# =============================================================================
# Model Loading
# =============================================================================

def load_model():
    """Load Qwen2.5-14B in 4-bit with Unsloth, then attach LoRA adapters."""
    print(f"\nLoading model: {MODEL_NAME}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name     = MODEL_NAME,
        max_seq_length = MAX_SEQ_LENGTH,
        dtype          = None,        # auto: bfloat16 on A10G
        load_in_4bit   = True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r              = 16,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
        lora_alpha     = 16,
        lora_dropout   = 0,
        bias           = "none",
        use_gradient_checkpointing = "unsloth",
        random_state   = 42,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token    = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    print("Model loaded: Qwen2.5-14B 4-bit + LoRA (r=16, alpha=16)")
    return model, tokenizer


# =============================================================================
# Environment Health Check
# =============================================================================

def check_environment_health(api_url: str) -> bool:
    """Verify the negotiation environment API is reachable."""
    print(f"\nChecking environment at: {api_url}")
    try:
        resp = requests.get(f"{api_url}/health", timeout=API_TIMEOUT)
        if resp.status_code == 200:
            data = resp.json()
            print(f"Environment healthy — {data.get('tasks_available', '?')} tasks available")
            return True
        print(f"Health check failed: HTTP {resp.status_code}")
        return False
    except Exception as e:
        print(f"Cannot reach environment: {e}")
        return False


# =============================================================================
# Prompt Formatting
# =============================================================================

def format_observation_prompt(obs: dict, agent_id: str) -> str:
    """Format an environment observation into the model's input prompt."""
    private_info = obs.get("private_information", {})
    private_info_str = json.dumps(private_info, indent=2)
    if len(private_info_str) > 800:
        private_info_str = private_info_str[:800] + "\n... (truncated)"

    history = obs.get("shared_conversation_history", [])
    recent  = history[-4:] if len(history) > 4 else history
    history_str = ""
    for msg in recent:
        aid     = msg.get("agent_id", "?")
        atype   = msg.get("action_type", "?")
        content = msg.get("content", "")[:300]
        history_str += f"  [{aid}] {atype}: {content}\n"
    if not history_str:
        history_str = "  (No messages yet — you go first)"

    available_actions = obs.get("available_actions", [])
    hidden_agenda     = obs.get("hidden_agenda", None)
    current_phase     = obs.get("current_phase", "triage")
    phase_turn        = obs.get("phase_turn", 0)
    max_turns         = obs.get("max_turns", 16)
    turn_warning      = obs.get("turn_warning", False)

    agenda_section  = f"\nHidden agenda (confidential): {hidden_agenda[:300]}" if hidden_agenda else ""
    warning_section = "\nTURN WARNING: Approaching turn limit. Propose consensus NOW." if turn_warning else ""

    return f"""You are {agent_id} in a medical negotiation environment.
Current phase: {current_phase} (phase turn {phase_turn}, total max turns: {max_turns})
Your private information:
{private_info_str}

Conversation so far:
{history_str}{agenda_section}{warning_section}

Available actions: {', '.join(available_actions)}

Respond ONLY with a valid JSON object with no markdown:
{{
  "agent_id": "{agent_id}",
  "action_type": "<one of available_actions>",
  "content": "<your message in plain English>",
  "reasoning": "<why you are taking this action, minimum 30 words, use medical reasoning>"
}}
If action_type is flag_bias, also include: "bias_location", "bias_direction", "bias_correction".
If action_type is flag_agenda, also include: "agenda_type", "agenda_evidence", "agenda_counter".
CRITICAL: If you have already shared your key information and {phase_turn} >= 2, use propose_consensus immediately."""


# =============================================================================
# Model Inference
# =============================================================================

def generate_action(model, tokenizer, prompt: str, agent_id: str) -> tuple[dict, str]:
    """Run model inference to generate an action JSON."""
    import torch

    inputs = tokenizer(
        prompt,
        return_tensors = "pt",
        truncation     = True,
        max_length     = 1536,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens = MAX_NEW_TOKENS,
            do_sample      = True,
            temperature    = 0.7,
            top_p          = 0.9,
            pad_token_id   = tokenizer.pad_token_id,
            eos_token_id   = tokenizer.eos_token_id,
        )

    new_tokens  = outputs[0][inputs["input_ids"].shape[1]:]
    raw_output  = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    action      = _parse_action_json(raw_output, agent_id)
    return action, raw_output


def _parse_action_json(raw_output: str, agent_id: str) -> dict:
    """Try to extract a valid Action JSON from model output."""
    try:
        action = json.loads(raw_output)
        action["agent_id"] = agent_id
        return action
    except json.JSONDecodeError:
        pass

    match = re.search(r'\{[^{}]*\}', raw_output, re.DOTALL)
    if match:
        try:
            action = json.loads(match.group())
            action["agent_id"] = agent_id
            return action
        except json.JSONDecodeError:
            pass

    raw_lower = raw_output.lower()
    if "accept_consensus"  in raw_lower: action_type = "accept_consensus"
    elif "propose_consensus" in raw_lower: action_type = "propose_consensus"
    elif "flag_agenda"       in raw_lower: action_type = "flag_agenda"
    elif "flag_bias"         in raw_lower: action_type = "flag_bias"
    else:                                  action_type = "share_information"

    return {
        "agent_id":    agent_id,
        "action_type": action_type,
        "content":     raw_output[:500] if raw_output else "I need to share my clinical findings with you.",
        "reasoning":   "Fallback action because JSON parsing failed. Sharing available medical information to progress the negotiation.",
    }


# =============================================================================
# Episode Runner
# =============================================================================

def run_episode(
    model,
    tokenizer,
    task_id: str,
    api_url: str,
) -> list[tuple[str, str, float]]:
    """
    Run one full episode on the environment.
    Returns list of (prompt, model_response, step_reward) tuples for GRPO training.
    """
    experiences = []

    try:
        reset_resp = requests.post(
            f"{api_url}/reset",
            json    = {"task_id": task_id},
            timeout = API_TIMEOUT,
        )
        reset_resp.raise_for_status()
        state = reset_resp.json()
    except Exception as e:
        print(f"  Reset failed for {task_id}: {e}")
        return []

    session_id       = state.get("session_id", "")
    obs_a            = state.get("obs_agent_a", {})
    obs_b            = state.get("obs_agent_b", {})
    done             = False
    turn             = 0
    current_agent_id = "agent_a"
    current_obs      = obs_a

    while not done and turn < MAX_TURNS_PER_EPISODE:
        turn += 1

        available = current_obs.get("available_actions", [])
        if not available:
            break

        prompt = format_observation_prompt(current_obs, current_agent_id)

        try:
            action, raw_response = generate_action(model, tokenizer, prompt, current_agent_id)
        except Exception as e:
            print(f"  Generate failed turn {turn}: {e}")
            action = {
                "agent_id":    current_agent_id,
                "action_type": "share_information",
                "content":     "I need to share my medical findings with you.",
                "reasoning":   "Fallback due to generation error. Continuing to share information to reach consensus.",
            }
            raw_response = json.dumps(action)

        if action.get("action_type") not in available:
            if "share_information" in available:
                action["action_type"] = "share_information"
                action["content"]     = "Let me share my findings: " + action.get("content", "")[:200]
            elif "accept_consensus" in available:
                action["action_type"] = "accept_consensus"
            elif available:
                action["action_type"] = available[0]

        try:
            step_resp = requests.post(
                f"{api_url}/step",
                json    = {"session_id": session_id, "action": action},
                timeout = API_TIMEOUT,
            )
            if not step_resp.ok:
                print(f"  Step failed (HTTP {step_resp.status_code}): {step_resp.text[:200]}")
                break
            step_data = step_resp.json()
        except Exception as e:
            print(f"  Step request failed: {e}")
            break

        reward_obj  = step_data.get("reward", {})
        step_reward = float(reward_obj.get("step_reward", 0.0)) if isinstance(reward_obj, dict) else float(reward_obj or 0.0)
        done        = step_data.get("done", False)

        experiences.append((prompt, raw_response, step_reward))

        obs_a = step_data.get("obs_agent_a", obs_a)
        obs_b = step_data.get("obs_agent_b", obs_b)

        if current_agent_id == "agent_a":
            current_agent_id = "agent_b"
            current_obs      = obs_b
        else:
            current_agent_id = "agent_a"
            current_obs      = obs_a

        time.sleep(0.3)

    return experiences


# =============================================================================
# GRPO Reward Function
# =============================================================================

def negotiation_reward_fn(prompts: list, completions: list, **kwargs) -> list[float]:
    """
    Deterministic reward function for GRPO training.
    No LLM calls — pure rule-based scoring.
    """
    for i in range(len(completions)):
        if isinstance(completions[i], list) and completions[i] and isinstance(completions[i][-1], dict):
            completions[i] = completions[i][-1].get("content", "")

    return [_score_single_completion(c) for c in completions]


def _score_single_completion(completion: str) -> float:
    """Score one model completion. Returns float clamped to (-1.0, 1.0)."""
    reward = 0.0

    action = None
    try:
        action = json.loads(completion)
    except (json.JSONDecodeError, ValueError):
        pass

    if action is None:
        match = re.search(r'\{[^{}]*\}', completion, re.DOTALL)
        if match:
            try:
                action = json.loads(match.group())
            except (json.JSONDecodeError, ValueError):
                pass

    if action is None:
        return -0.5

    action_type = action.get("action_type", "")
    content     = action.get("content", "")
    reasoning   = action.get("reasoning", "")

    VALID_ACTIONS = {
        "share_information", "propose_consensus", "challenge_proposal",
        "request_clarification", "accept_consensus", "reject_consensus",
        "flag_bias", "flag_agenda",
    }
    if action_type not in VALID_ACTIONS:
        return -0.3

    content_lower = (content + " " + reasoning).lower()
    medical_hits  = sum(1 for kw in MEDICAL_KEYWORDS if kw in content_lower)

    if action_type == "share_information":
        reward += 0.3 if medical_hits >= 3 else 0.1 if medical_hits >= 1 else 0.0

    elif action_type == "propose_consensus":
        reward += 0.4

    elif action_type == "accept_consensus":
        reward += 0.5

    elif action_type == "flag_agenda":
        has_type     = bool(action.get("agenda_type", "").strip())
        has_evidence = bool(action.get("agenda_evidence", "").strip())
        has_counter  = bool(action.get("agenda_counter", "").strip())
        reward += 0.6 if (has_type and has_evidence and has_counter) else 0.1

    elif action_type == "flag_bias":
        has_location   = bool(action.get("bias_location", "").strip())
        has_direction  = bool(action.get("bias_direction", "").strip())
        has_correction = bool(action.get("bias_correction", "").strip())
        reward += 0.6 if (has_location and has_direction and has_correction) else 0.1

    elif action_type == "challenge_proposal":
        reward += 0.2

    elif action_type == "request_clarification":
        reward += 0.05

    elif action_type == "reject_consensus":
        reward += 0.0

    if len(reasoning.split()) < 20:
        reward -= 0.2

    return max(-1.0, min(1.0, round(reward, 4)))


# =============================================================================
# Training Utilities
# =============================================================================

def collect_episodes(
    model,
    tokenizer,
    task_ids: list,
    episodes_per_task: int,
    api_url: str,
    round_num: int,
) -> tuple[list, list, list]:
    """Run episodes for all tasks, return flat (prompts, responses, rewards)."""
    all_prompts, all_responses, all_rewards = [], [], []

    print(f"\n  Round {round_num} — Collecting {episodes_per_task * len(task_ids)} episodes...")

    for task_id in task_ids:
        task_rewards = []
        for ep in range(episodes_per_task):
            experiences = run_episode(model, tokenizer, task_id, api_url)
            for prompt, response, reward in experiences:
                all_prompts.append(prompt)
                all_responses.append(response)
                all_rewards.append(reward)
                task_rewards.append(reward)

        avg = sum(task_rewards) / max(len(task_rewards), 1)
        print(f"    [{task_id}] {len(task_rewards)} steps, avg_reward={avg:.4f}")

    return all_prompts, all_responses, all_rewards


def build_grpo_dataset(prompts: list, responses: list, rewards: list):
    """Package collected experience into a HuggingFace Dataset for GRPOTrainer."""
    from datasets import Dataset
    return Dataset.from_dict({"prompt": prompts})


def save_checkpoint(model, tokenizer, round_num: int) -> None:
    """Save LoRA adapter checkpoint after each round."""
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"round_{round_num}")
    os.makedirs(ckpt_path, exist_ok=True)
    model.save_pretrained(ckpt_path)
    tokenizer.save_pretrained(ckpt_path)
    print(f"  Checkpoint saved → {ckpt_path}")


# =============================================================================
# Main Training Loop
# =============================================================================

def train():
    """End-to-end GRPO training loop with time-limit enforcement."""

    print("=" * 65)
    print("  Social Agent Negotiation — GRPO Training")
    print(f"  Model : {MODEL_NAME}")
    print(f"  Rounds: {TRAINING_ROUNDS}  |  Episodes/task: {EPISODES_PER_TASK}")
    print(f"  Time limit: 2.5 hours")
    print("=" * 65)

    start_time = time.time()

    hf_token         = setup_hf_auth()
    model, tokenizer = load_model()

    env_ok = check_environment_health(API_URL)
    if not env_ok:
        print("Environment unreachable — offline reward function will be used.")

    training_args = GRPOConfig(
        output_dir                  = "./grpo_output",
        num_train_epochs            = 1,
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 4,
        learning_rate               = 2e-5,
        max_completion_length       = MAX_NEW_TOKENS,
        num_generations             = 2,
        report_to                   = "none",
    )
    print("GRPOConfig created.")

    round_rewards = []
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    for round_num in range(1, TRAINING_ROUNDS + 1):
        # Hard time-limit check before starting each round
        elapsed = time.time() - start_time
        if elapsed >= TIME_LIMIT_SECONDS:
            print(f"\nTime limit reached ({elapsed/3600:.2f}h elapsed) — saving and exiting before round {round_num}.")
            save_checkpoint(model, tokenizer, round_num - 1)
            break

        remaining = TIME_LIMIT_SECONDS - elapsed
        print(f"\n{'='*65}")
        print(f"  ROUND {round_num} / {TRAINING_ROUNDS}  "
              f"(elapsed {elapsed/3600:.2f}h, {remaining/3600:.2f}h remaining)")
        print(f"{'='*65}")

        prompts, responses, rewards = collect_episodes(
            model, tokenizer, TASK_IDS,
            EPISODES_PER_TASK, API_URL, round_num,
        )

        if not prompts:
            print("No experiences collected this round — skipping GRPO update.")
            round_rewards.append(0.0)
            save_checkpoint(model, tokenizer, round_num)
            continue

        avg_round_reward = sum(rewards) / len(rewards)
        round_rewards.append(avg_round_reward)
        print(f"\n  Round {round_num} avg step reward: {avg_round_reward:.4f} ({len(prompts)} samples)")

        dataset = build_grpo_dataset(prompts, responses, rewards)
        print(f"  Running GRPO update ({len(dataset)} samples)...")

        trainer = GRPOTrainer(
            model            = model,
            args             = training_args,
            reward_funcs     = [negotiation_reward_fn],
            train_dataset    = dataset,
            processing_class = tokenizer,
        )

        try:
            trainer.train()
            print(f"  GRPO update complete for round {round_num}.")
        except Exception as e:
            print(f"  GRPO update failed: {e}")
            traceback.print_exc()

        save_checkpoint(model, tokenizer, round_num)

    # -------------------------------------------------------------------------
    # Reward curve
    # -------------------------------------------------------------------------
    if round_rewards:
        plt.figure(figsize=(10, 5))
        plt.plot(
            range(1, len(round_rewards) + 1),
            round_rewards,
            marker    = "o",
            linewidth = 2,
            color     = "#4A90D9",
            markersize = 8,
            label     = "Avg Step Reward",
        )
        plt.axhline(y=0, color="gray", linestyle="--", linewidth=1, alpha=0.5)
        plt.fill_between(range(1, len(round_rewards) + 1), round_rewards, alpha=0.15, color="#4A90D9")
        plt.xlabel("Training Round", fontsize=13)
        plt.ylabel("Average Step Reward", fontsize=13)
        plt.title(
            "GRPO Training — Social Agent Negotiation\nReward Improvement per Round",
            fontsize=14, fontweight="bold",
        )
        plt.xticks(range(1, len(round_rewards) + 1))
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=11)
        plt.tight_layout()
        plt.savefig(REWARD_CURVE_PATH, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\nReward curve saved to {REWARD_CURVE_PATH}")

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n--- Training Summary ---")
    print(f"{'Round':<8} {'Avg Step Reward':<20}")
    print("-" * 30)
    for i, r in enumerate(round_rewards, 1):
        symbol = "^" if (i > 1 and r > round_rewards[i - 2]) else "v" if i > 1 else " "
        print(f"{i:<8} {r:<20.4f} {symbol}")

    total_time = time.time() - start_time
    print(f"\nTotal training time: {total_time/3600:.2f}h  (${total_time/3600 * 1.05:.2f} estimated)")

    # -------------------------------------------------------------------------
    # Push to Hub
    # -------------------------------------------------------------------------
    if hf_token:
        print(f"\nPushing fine-tuned model to {HF_REPO_ID} ...")
        try:
            model.push_to_hub(HF_REPO_ID, token=hf_token)
            tokenizer.push_to_hub(HF_REPO_ID, token=hf_token)
            print(f"Model pushed to https://huggingface.co/{HF_REPO_ID}")
        except Exception as e:
            print(f"Push to Hub failed: {e}")
            traceback.print_exc()
    else:
        print("Skipping Hub push — HF_TOKEN not set.")

    print("\nTraining complete.")
    return round_rewards


if __name__ == "__main__":
    train()
