# =============================================================================
# Social Agent Negotiation — GRPO Training Script
# =============================================================================
# Run this on Google Colab (T4 GPU recommended, free tier works)
# Before running: set your HF_TOKEN in Colab Secrets (key icon on left sidebar)
#
# Usage:
#   1. Upload this file to Colab or open from GitHub
#   2. Set HF_TOKEN in Colab secrets
#   3. Runtime → Run All
# =============================================================================

# @title Step 1 — Install dependencies (run once, restart runtime after)
# %%capture
# !pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
# !pip install trl transformers requests matplotlib datasets accelerate peft
# !pip install --upgrade huggingface_hub

# =============================================================================
# @title Step 2 — Imports & Configuration
# =============================================================================

import os
import json
import re
import time
import random
import requests
import traceback
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for Colab
import matplotlib.pyplot as plt
from typing import Optional

# HuggingFace imports
from transformers import AutoTokenizer
from trl import GRPOTrainer, GRPOConfig
from huggingface_hub import login

# Unsloth for fast loading on T4
try:
    from unsloth import FastLanguageModel
    UNSLOTH_AVAILABLE = True
    print("✅ Unsloth available — using fast model loading")
except ImportError:
    UNSLOTH_AVAILABLE = False
    print("⚠️  Unsloth not available — falling back to standard HuggingFace")
    from transformers import AutoModelForCausalLM

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_URL = "https://Bharath-1608-social-agent-negotiation-v1.hf.space"

MODEL_NAME       = "unsloth/Llama-3.2-1B-Instruct"
OUTPUT_DIR       = "./grpo_negotiation"
HF_REPO_ID       = "Bharath-1608/negotiation-agent-grpo"
REWARD_CURVE_PATH = "reward_curve.png"

TASK_IDS = [
    "single-round-consensus",
    "multi-round-negotiation",
    "adversarial-information",
    "pediatric-meningitis",
    "opioid-overdose",
]

TRAINING_ROUNDS      = 3
EPISODES_PER_TASK    = 10
MAX_TURNS_PER_EPISODE = 20    # Safety cap — environment enforces its own limit too
API_TIMEOUT          = 30     # seconds per request
MAX_NEW_TOKENS       = 512

MEDICAL_KEYWORDS = [
    "patient", "diagnosis", "treatment", "symptoms", "vital", "blood pressure",
    "heart rate", "ecg", "troponin", "oxygen", "saturation", "critical",
    "urgent", "triage", "medication", "dosage", "clinical", "prognosis",
    "assessment", "evidence", "imaging", "labs", "antibiotic", "heparin",
    "tpa", "naloxone", "meningitis", "sepsis", "embolism", "stroke",
]

# =============================================================================
# @title Step 3 — HuggingFace Login
# =============================================================================

def setup_hf_auth() -> str:
    """Login to HuggingFace using Colab Secrets or env variable."""
    try:
        from google.colab import userdata
        hf_token = userdata.get("HF_TOKEN")
        print("✅ HF_TOKEN loaded from Colab Secrets")
    except Exception:
        hf_token = os.environ.get("HF_TOKEN", "")
        if hf_token:
            print("✅ HF_TOKEN loaded from environment variable")
        else:
            print("❌ HF_TOKEN not found — push to Hub will fail")
            return ""

    if hf_token:
        login(token=hf_token)
        print("✅ Logged in to HuggingFace Hub")
    return hf_token


# =============================================================================
# @title Step 4 — Model Loading
# =============================================================================

def load_model():
    """Load the model and tokenizer using Unsloth if available."""
    print(f"\n📦 Loading model: {MODEL_NAME}")

    if UNSLOTH_AVAILABLE:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name   = MODEL_NAME,
            max_seq_length = 2048,
            dtype          = None,      # Auto-detect: bfloat16 on A100, float16 on T4
            load_in_4bit   = True,      # 4-bit quantisation for T4 memory constraints
        )
        # Add LoRA adapters for efficient fine-tuning
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
        print(f"✅ Model loaded with Unsloth (4-bit LoRA)")
    else:
        from transformers import AutoModelForCausalLM
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            load_in_4bit = True,
            device_map   = "auto",
        )
        print(f"✅ Model loaded with standard HuggingFace")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer


# =============================================================================
# @title Step 5 — Environment Health Check
# =============================================================================

def check_environment_health(api_url: str) -> bool:
    """Verify the negotiation environment API is reachable."""
    print(f"\n🏥 Checking environment at: {api_url}")
    try:
        resp = requests.get(f"{api_url}/health", timeout=API_TIMEOUT)
        if resp.status_code == 200:
            data = resp.json()
            tasks_avail = data.get("tasks_available", "?")
            print(f"✅ Environment healthy — {tasks_avail} tasks available")
            return True
        else:
            print(f"❌ Health check failed: HTTP {resp.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot reach environment: {e}")
        return False


# =============================================================================
# @title Step 6 — Prompt Formatting
# =============================================================================

def format_observation_prompt(obs: dict, agent_id: str) -> str:
    """
    Format an environment observation into the model's input prompt.
    Converts structured observation into a clear natural language prompt.
    """
    private_info = obs.get("private_information", {})
    # Truncate private info to avoid overly long prompts
    private_info_str = json.dumps(private_info, indent=2)
    if len(private_info_str) > 800:
        private_info_str = private_info_str[:800] + "\n... (truncated)"

    # Conversation history — last 4 messages to stay within context
    history = obs.get("shared_conversation_history", [])
    recent = history[-4:] if len(history) > 4 else history
    history_str = ""
    for msg in recent:
        aid = msg.get("agent_id", "?")
        atype = msg.get("action_type", "?")
        content = msg.get("content", "")[:300]
        history_str += f"  [{aid}] {atype}: {content}\n"
    if not history_str:
        history_str = "  (No messages yet — you go first)"

    available_actions = obs.get("available_actions", [])
    hidden_agenda = obs.get("hidden_agenda", None)

    current_phase = obs.get("current_phase", "triage")
    phase_turn    = obs.get("phase_turn", 0)
    max_turns     = obs.get("max_turns", 16)
    turn_warning  = obs.get("turn_warning", False)

    agenda_section = ""
    if hidden_agenda:
        agenda_section = f"\nHidden agenda (confidential): {hidden_agenda[:300]}"

    warning_section = ""
    if turn_warning:
        warning_section = "\n⚠️  TURN WARNING: Approaching turn limit. Propose consensus NOW."

    prompt = f"""You are {agent_id} in a medical negotiation environment.
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

    return prompt


# =============================================================================
# @title Step 7 — Model Inference (JSON action generation)
# =============================================================================

def generate_action(model, tokenizer, prompt: str, agent_id: str) -> dict:
    """
    Run model inference to generate an action JSON.
    Falls back to a safe default if JSON parsing fails.
    """
    import torch

    # Tokenize
    inputs = tokenizer(
        prompt,
        return_tensors = "pt",
        truncation     = True,
        max_length     = 1536,
    ).to(model.device)

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens  = MAX_NEW_TOKENS,
            do_sample       = True,
            temperature     = 0.7,
            top_p           = 0.9,
            pad_token_id    = tokenizer.pad_token_id,
            eos_token_id    = tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    raw_output = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    # Parse JSON — try multiple extraction strategies
    action = _parse_action_json(raw_output, agent_id)
    return action, raw_output


def _parse_action_json(raw_output: str, agent_id: str) -> dict:
    """Try to extract a valid Action JSON from model output."""
    # Strategy 1: Direct parse
    try:
        action = json.loads(raw_output)
        action["agent_id"] = agent_id
        return action
    except json.JSONDecodeError:
        pass

    # Strategy 2: Extract first {...} block
    match = re.search(r'\{[^{}]*\}', raw_output, re.DOTALL)
    if match:
        try:
            action = json.loads(match.group())
            action["agent_id"] = agent_id
            return action
        except json.JSONDecodeError:
            pass

    # Strategy 3: Heuristic — if model said "propose_consensus" or "accept", use that
    raw_lower = raw_output.lower()
    if "accept_consensus" in raw_lower:
        action_type = "accept_consensus"
    elif "propose_consensus" in raw_lower:
        action_type = "propose_consensus"
    elif "flag_agenda" in raw_lower:
        action_type = "flag_agenda"
    elif "flag_bias" in raw_lower:
        action_type = "flag_bias"
    else:
        action_type = "share_information"

    # Fallback action
    return {
        "agent_id":    agent_id,
        "action_type": action_type,
        "content":     raw_output[:500] if raw_output else "I need to share my clinical findings with you.",
        "reasoning":   "Fallback action because JSON parsing failed. Sharing available medical information to progress the negotiation.",
    }


# =============================================================================
# @title Step 8 — Episode Runner
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
    experiences = []   # List of (prompt, response, reward)

    # --- Reset ---
    try:
        reset_resp = requests.post(
            f"{api_url}/reset",
            json    = {"task_id": task_id},
            timeout = API_TIMEOUT,
        )
        reset_resp.raise_for_status()
        state = reset_resp.json()
    except Exception as e:
        print(f"  ⚠️  Reset failed for {task_id}: {e}")
        return []

    obs_a = state.get("obs_agent_a", {})
    obs_b = state.get("obs_agent_b", {})
    done  = False
    turn  = 0

    # Alternate: A goes first
    current_agent_id  = "agent_a"
    current_obs       = obs_a

    while not done and turn < MAX_TURNS_PER_EPISODE:
        turn += 1

        available = current_obs.get("available_actions", [])
        if not available:
            break

        # Format prompt
        prompt = format_observation_prompt(current_obs, current_agent_id)

        # Generate action
        try:
            action, raw_response = generate_action(model, tokenizer, prompt, current_agent_id)
        except Exception as e:
            print(f"  ⚠️  Generate failed turn {turn}: {e}")
            action = {
                "agent_id":    current_agent_id,
                "action_type": "share_information",
                "content":     "I need to share my medical findings with you.",
                "reasoning":   "Fallback due to generation error. Continuing to share information to reach consensus.",
            }
            raw_response = json.dumps(action)

        # Validate action_type is legal
        if action.get("action_type") not in available:
            # Pick a safe default from available actions
            if "share_information" in available:
                action["action_type"] = "share_information"
                action["content"] = "Let me share my findings: " + action.get("content", "")[:200]
            elif "accept_consensus" in available:
                action["action_type"] = "accept_consensus"
            elif available:
                action["action_type"] = available[0]

        # Step the environment
        try:
            step_resp = requests.post(
                f"{api_url}/step",
                json    = {"action": action},
                timeout = API_TIMEOUT,
            )
            if not step_resp.ok:
                print(f"  ⚠️  Step failed (HTTP {step_resp.status_code}): {step_resp.text[:200]}")
                break

            step_data = step_resp.json()
        except Exception as e:
            print(f"  ⚠️  Step request failed: {e}")
            break

        # Extract reward
        reward_obj   = step_data.get("reward", {})
        step_reward  = float(reward_obj.get("step_reward", 0.0))
        done         = step_data.get("done", False)

        # Record experience
        experiences.append((prompt, raw_response, step_reward))

        # Update observations
        obs_a = step_data.get("obs_agent_a", obs_a)
        obs_b = step_data.get("obs_agent_b", obs_b)

        # Rotate agents
        if current_agent_id == "agent_a":
            current_agent_id = "agent_b"
            current_obs      = obs_b
        else:
            current_agent_id = "agent_a"
            current_obs      = obs_a

        # Small delay to be kind to HF Space rate limits
        time.sleep(0.3)

    return experiences


# =============================================================================
# @title Step 9 — GRPO Reward Function
# =============================================================================

def negotiation_reward_fn(prompts: list, completions: list, **kwargs) -> list[float]:
    """
    Deterministic reward function for GRPO training.
    Evaluates each model completion based on JSON validity,
    action quality, and reasoning depth.
    No LLM calls — pure rule-based scoring.
    """
    rewards = []

    for prompt, completion in zip(prompts, completions):
        reward = _score_single_completion(completion)
        rewards.append(reward)

    return rewards


def _score_single_completion(completion: str) -> float:
    """Score one model completion. Returns float clamped to (-1.0, 1.0)."""
    reward = 0.0

    # --- Parse JSON ---
    action = None
    # Try direct parse
    try:
        action = json.loads(completion)
    except (json.JSONDecodeError, ValueError):
        pass

    # Try extracting first {} block
    if action is None:
        match = re.search(r'\{[^{}]*\}', completion, re.DOTALL)
        if match:
            try:
                action = json.loads(match.group())
            except (json.JSONDecodeError, ValueError):
                pass

    if action is None:
        return max(-1.0, min(1.0, -0.5))   # JSON invalid

    action_type = action.get("action_type", "")
    content     = action.get("content", "")
    reasoning   = action.get("reasoning", "")

    # Note: we can't easily check available_actions here without the env state,
    # so we score based on action quality instead
    VALID_ACTIONS = {
        "share_information", "propose_consensus", "challenge_proposal",
        "request_clarification", "accept_consensus", "reject_consensus",
        "flag_bias", "flag_agenda"
    }
    if action_type not in VALID_ACTIONS:
        return max(-1.0, min(1.0, -0.3))

    # --- Action-type rewards ---
    content_lower  = (content + " " + reasoning).lower()
    medical_hits   = sum(1 for kw in MEDICAL_KEYWORDS if kw in content_lower)

    if action_type == "share_information":
        if medical_hits >= 3:
            reward += 0.3
        elif medical_hits >= 1:
            reward += 0.1
        else:
            reward += 0.0

    elif action_type == "propose_consensus":
        reward += 0.4

    elif action_type == "accept_consensus":
        reward += 0.5

    elif action_type == "flag_agenda":
        has_type     = bool(action.get("agenda_type", "").strip())
        has_evidence = bool(action.get("agenda_evidence", "").strip())
        has_counter  = bool(action.get("agenda_counter", "").strip())
        if has_type and has_evidence and has_counter:
            reward += 0.6
        else:
            reward += 0.1  # Partial — tried but incomplete

    elif action_type == "flag_bias":
        has_location   = bool(action.get("bias_location", "").strip())
        has_direction  = bool(action.get("bias_direction", "").strip())
        has_correction = bool(action.get("bias_correction", "").strip())
        if has_location and has_direction and has_correction:
            reward += 0.6
        else:
            reward += 0.1

    elif action_type == "challenge_proposal":
        reward += 0.2   # Constructive engagement

    elif action_type == "request_clarification":
        reward += 0.05  # Low reward — prefer active engagement

    elif action_type == "reject_consensus":
        reward += 0.0   # Neutral — may be appropriate but don't incentivise

    # --- Reasoning depth penalty ---
    reasoning_words = len(reasoning.split())
    if reasoning_words < 20:
        reward -= 0.2   # Penalise shallow reasoning

    # Clamp to (-1.0, 1.0)
    return max(-1.0, min(1.0, round(reward, 4)))


# =============================================================================
# @title Step 10 — Training Utilities
# =============================================================================

def collect_episodes(
    model,
    tokenizer,
    task_ids: list,
    episodes_per_task: int,
    api_url: str,
    round_num: int,
) -> tuple[list, list, list]:
    """
    Run episodes for all tasks and collect (prompt, response, reward) triples.
    Returns (prompts, responses, rewards) as flat lists.
    """
    all_prompts   = []
    all_responses = []
    all_rewards   = []

    print(f"\n  📊 Round {round_num} — Collecting {episodes_per_task * len(task_ids)} episodes...")

    for task_id in task_ids:
        task_rewards = []
        for ep in range(episodes_per_task):
            experiences = run_episode(model, tokenizer, task_id, api_url)
            if experiences:
                for prompt, response, reward in experiences:
                    all_prompts.append(prompt)
                    all_responses.append(response)
                    all_rewards.append(reward)
                    task_rewards.append(reward)

        avg_task_reward = sum(task_rewards) / max(len(task_rewards), 1)
        print(f"    [{task_id}] {len(task_rewards)} steps, avg_reward={avg_task_reward:.4f}")

    return all_prompts, all_responses, all_rewards


def build_grpo_dataset(prompts: list, responses: list, rewards: list):
    """
    Package collected experience into a HuggingFace Dataset for GRPOTrainer.
    """
    from datasets import Dataset
    data = {
        "prompt":     prompts,
        "completion": responses,
        "reward":     rewards,
    }
    return Dataset.from_dict(data)


# =============================================================================
# @title Step 11 — Main Training Loop
# =============================================================================

def train():
    """End-to-end GRPO training loop."""

    print("=" * 65)
    print("  Social Agent Negotiation — GRPO Training")
    print("=" * 65)

    # --- Auth ---
    hf_token = setup_hf_auth()

    # --- Model ---
    model, tokenizer = load_model()

    # --- Environment health check ---
    env_ok = check_environment_health(API_URL)
    if not env_ok:
        print("❌ Environment unreachable. Training will use offline reward function only.")
        print("   (Model quality won't degrade from offline scoring, but env rewards won't be captured)")

    # --- GRPO Config ---
    training_args = GRPOConfig(
        output_dir                  = OUTPUT_DIR,
        num_train_epochs            = 1,
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 4,
        learning_rate               = 2e-5,
        max_completion_length       = MAX_NEW_TOKENS,
        num_generations             = 2,
        report_to                   = "none",
    )

    print("\n✅ GRPOConfig created")

    # --- Initialize trainer once — we update the dataset each round ---
    round_rewards    = []             # Average reward per training round
    task_reward_log  = {tid: [] for tid in TASK_IDS}

    # ==========================================================================
    # Main Training Loop
    # ==========================================================================

    for round_num in range(1, TRAINING_ROUNDS + 1):
        print(f"\n{'='*65}")
        print(f"  ROUND {round_num} / {TRAINING_ROUNDS}")
        print(f"{'='*65}")

        # --- Collect episodes ---
        prompts, responses, rewards = collect_episodes(
            model, tokenizer, TASK_IDS,
            EPISODES_PER_TASK, API_URL, round_num,
        )

        if not prompts:
            print("⚠️  No experiences collected this round — skipping GRPO update")
            round_rewards.append(0.0)
            continue

        avg_round_reward = sum(rewards) / len(rewards)
        round_rewards.append(avg_round_reward)
        print(f"\n  Round {round_num} avg step reward: {avg_round_reward:.4f} "
              f"({len(prompts)} samples)")

        # --- Build dataset ---
        dataset = build_grpo_dataset(prompts, responses, rewards)

        # --- GRPO Update ---
        print(f"  Running GRPO update ({len(dataset)} samples)...")

        trainer = GRPOTrainer(
            model         = model,
            args          = training_args,
            reward_funcs  = negotiation_reward_fn,
            train_dataset = dataset,
            tokenizer     = tokenizer,
        )

        try:
            trainer.train()
            print(f"  ✅ GRPO update complete for round {round_num}")
        except Exception as e:
            print(f"  ⚠️  GRPO update failed: {e}")
            traceback.print_exc()

    # ==========================================================================
    # Post-Training: Plot Reward Curve
    # ==========================================================================

    print(f"\n{'='*65}")
    print("  Post-Training Analysis")
    print(f"{'='*65}")

    # --- Reward curve plot ---
    plt.figure(figsize=(10, 5))
    plt.plot(
        range(1, len(round_rewards) + 1),
        round_rewards,
        marker   = "o",
        linewidth = 2,
        color    = "#4A90D9",
        markersize = 8,
        label    = "Avg Step Reward",
    )
    plt.axhline(y=0, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    plt.fill_between(
        range(1, len(round_rewards) + 1),
        round_rewards,
        alpha = 0.15,
        color = "#4A90D9",
    )
    plt.xlabel("Training Round",  fontsize=13)
    plt.ylabel("Average Step Reward", fontsize=13)
    plt.title("GRPO Training — Social Agent Negotiation\nReward Improvement per Round",
              fontsize=14, fontweight="bold")
    plt.xticks(range(1, len(round_rewards) + 1))
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(REWARD_CURVE_PATH, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"📈 Reward curve saved to {REWARD_CURVE_PATH}")

    # --- Summary table ---
    print("\n--- Training Summary ---")
    print(f"{'Round':<8} {'Avg Step Reward':<20}")
    print("-" * 30)
    for i, r in enumerate(round_rewards, 1):
        symbol = "📈" if (i > 1 and r > round_rewards[i - 2]) else "📉" if i > 1 else "  "
        print(f"{i:<8} {r:<20.4f} {symbol}")

    # --- Push to HuggingFace Hub ---
    if hf_token:
        print(f"\n🚀 Pushing fine-tuned model to {HF_REPO_ID}...")
        try:
            if UNSLOTH_AVAILABLE:
                model.push_to_hub(HF_REPO_ID, token=hf_token)
                tokenizer.push_to_hub(HF_REPO_ID, token=hf_token)
            else:
                model.push_to_hub(HF_REPO_ID, token=hf_token)
                tokenizer.push_to_hub(HF_REPO_ID, token=hf_token)
            print(f"✅ Model pushed to https://huggingface.co/{HF_REPO_ID}")
        except Exception as e:
            print(f"⚠️  Push to Hub failed: {e}")
    else:
        print("⚠️  Skipping Hub push — HF_TOKEN not set")

    print("\nTraining complete. Reward curve saved to reward_curve.png")
    return round_rewards


# =============================================================================
# @title Step 12 — Run Training
# =============================================================================

if __name__ == "__main__":
    round_rewards = train()
