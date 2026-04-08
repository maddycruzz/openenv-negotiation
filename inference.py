import os
import sys
import json
import time
import requests
from typing import Dict, Any, Tuple, Optional, List
from openai import OpenAI

# Hackathon-standard environment variables
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN") or os.getenv("API_KEY")

if not HF_TOKEN:
    print("ERROR: HF_TOKEN environment variable is not set.", file=sys.stderr)
    # Don't strictly exit just in case they have a local mock without api key
    # sys.exit(1)

BENCHMARK    = "social-agent-negotiation"
API_URL      = "http://localhost:7860"

client = OpenAI(api_key=HF_TOKEN or "dummy", base_url=API_BASE_URL)

ACTION_TYPES = [
    "share_information", 
    "propose_consensus", 
    "challenge_proposal", 
    "request_clarification", 
    "accept_consensus", 
    "reject_consensus", 
    "flag_bias"
]

def get_system_prompt(agent_id: str) -> str:
    return f"""You are {agent_id} in a negotiation environment.
Your goal is to share your private_information with the other agent and try to reach a consensus within the turn limit.
Available action_types: {', '.join(ACTION_TYPES)}
You MUST respond ONLY with a valid JSON object representing your Action.
The JSON object must contain these fields:
- "agent_id": "{agent_id}"
- "action_type": <one of the available action types>
- "content": <your message as a plain text string>
- "reasoning": <your internal reasoning for this action as a plain text string>
IMPORTANT RULES:
- The "content" field must ALWAYS be a plain text string. Never put a JSON object.
- Before choosing an action_type, always check the "available_actions" list in your observation.
If and ONLY if you choose the "flag_bias" action_type, your JSON must ALSO include:
- "bias_location": <where the bias was found>
- "bias_direction": <direction of bias>
- "bias_correction": <suggested correction>
"""

def generate_agent_action(agent_id: str, observation: Dict[str, Any], retry_count: int = 0) -> Dict[str, Any]:
    system_prompt = get_system_prompt(agent_id)
    user_message = json.dumps(observation)
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=0.2,
            response_format={"type": "json_object"}
        )
        content = response.choices[0].message.content
        if not content:
            raise ValueError("Empty response from model")

        action = json.loads(content)
        action["agent_id"] = agent_id
        return action
        
    except Exception as e:
        if retry_count < 1:
            time.sleep(1)
            return generate_agent_action(agent_id, observation, retry_count + 1)
        return {
            "agent_id": agent_id,
            "action_type": "request_clarification",
            "content": f"API error fallback: {str(e)[:50]}",
            "reasoning": "Fallback action due to API failure."
        }

def _safe(v: float) -> float:
    return float(round(max(0.0001, min(0.9999, v)), 4))

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

def run_episode(task_id: str) -> Tuple[Optional[Dict[str, Any]], int, float]:
    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)
    
    try:
        reset_resp = requests.post(f"{API_URL}/reset", json={"task_id": task_id}, timeout=30)
        reset_resp.raise_for_status()
        state = reset_resp.json()
    except Exception as e:
        log_end(success=False, steps=0, score=0.01, rewards=[0.01])
        return None, 0, 0.01
        
    obs_a = state.get("obs_agent_a", {})
    obs_b = state.get("obs_agent_b", {})
    
    current_agent = "Agent A"
    done = False
    turn_count = 0
    final_result = None
    
    rewards_list = []
    
    while not done:
        turn_count += 1
        
        active_id = "agent_a" if current_agent == "Agent A" else "agent_b"
        obs = obs_a if current_agent == "Agent A" else obs_b
            
        action = generate_agent_action(active_id, obs)
        action_str = json.dumps(action)
        
        step_reward = 0.05
        error_msg = None
        
        try:
            step_resp = requests.post(f"{API_URL}/step", json={"action": action}, timeout=30)
            
            if not step_resp.ok:
                error_msg = f"HTTP {step_resp.status_code}"
                done = True
            else:
                step_data = step_resp.json()
                obs_a = step_data.get("obs_agent_a", obs_a)
                obs_b = step_data.get("obs_agent_b", obs_b)
                done = step_data.get("done", False)
                # Parse step reward safely
                raw_reward = step_data.get("reward", {})
                if isinstance(raw_reward, dict):
                    step_reward = float(raw_reward.get("cumulative_reward", 0.05))
                else:
                    step_reward = float(raw_reward)
                
                if done:
                    final_result = step_data.get("episode_result", {})
                
        except Exception as e:
            error_msg = f"request_failed:{str(e)[:50]}"
            done = True
            
        step_reward = _safe(step_reward)
        rewards_list.append(step_reward)
        
        log_step(step=turn_count, action=action_str, reward=step_reward, done=done, error=error_msg)
        
        if done:
            break
            
        current_agent = "Agent B" if current_agent == "Agent A" else "Agent A"
        time.sleep(0.05)
        
    score = 0.01
    success = False
    if final_result:
        raw_score = final_result.get("total_reward", 0.01)
        score = _safe(float(raw_score))
        success = final_result.get("final_consensus") == "reached"
        
    log_end(success=success, steps=turn_count, score=score, rewards=rewards_list)
    return final_result, turn_count, score

def main():
    try:
        tasks_resp = requests.get(f"{API_URL}/tasks", timeout=10)
        tasks_resp.raise_for_status()
        tasks = tasks_resp.json()
    except Exception as e:
        print(f"Failed to fetch tasks: {e}", file=sys.stderr)
        # Dummy tasks to not crash entirely immediately if server slow
        tasks = [{"id": "single-round-consensus"}, {"id": "multi-round-negotiation"}, {"id": "adversarial-information"}]

    task_items = tasks.items() if isinstance(tasks, dict) else enumerate(tasks)

    for key, task_data in task_items:
        task_id = task_data.get("id", str(key))
        run_episode(task_id)

if __name__ == "__main__":
    main()