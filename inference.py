from dotenv import load_dotenv
load_dotenv()
import os
import sys
import json
import time
import requests
from typing import Dict, Any, Tuple, Optional
from openai import OpenAI

# Hackathon-standard environment variables
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")
HF_TOKEN     = os.getenv("HF_TOKEN") or os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")

if not HF_TOKEN:
    print("ERROR: HF_TOKEN environment variable is not set.", file=sys.stderr)
    sys.exit(1)

# Single OpenAI-compatible client — works with Groq, OpenAI, or any compatible endpoint
client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)

# Environment API
API_URL = "http://localhost:7860"
# Available actions from models.py
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
    """
    Constructs the system prompt for the specified agent.
    Forces JSON output and includes the rules for flag_bias.
    """
    return f"""You are {agent_id} in a negotiation environment.
Your goal is to share your private_information with the other agent and try to reach a consensus within the turn limit.
Available action_types: {', '.join(ACTION_TYPES)}
You MUST respond ONLY with a valid JSON object representing your Action.
The JSON object must contain these fields:
- "agent_id": "{agent_id}"
- "action_type": <one of the available action types>
- "content": <your message as a plain text string — NEVER put a dict or JSON object here, always write a human-readable sentence>
- "reasoning": <your internal reasoning for this action as a plain text string>
IMPORTANT RULES:
- The "content" field must ALWAYS be a plain text string. Never put a JSON object, dict, or any structured data as the value of "content". Always write a natural language sentence.
- Before choosing an action_type, always check the "available_actions" list in your observation. Only use action types that appear in that list. Never use an action that is not listed.
- Share the key facts from your private_information in plain English inside the "content" field.
If and ONLY if you choose the "flag_bias" action_type, your JSON must ALSO include:
- "bias_location": <where the bias was found as a plain text string>
- "bias_direction": <direction of bias as a plain text string>
- "bias_correction": <suggested correction as a plain text string>
"""
def generate_agent_action(agent_id: str, observation: Dict[str, Any], retry_count: int = 0) -> Dict[str, Any]:
    """
    Generates an action using OpenAI API. Handles JSON parsing and retries once on failure.
    """
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
        
    except (json.JSONDecodeError, ValueError) as e:
        if retry_count < 1:
            print(f"  [{agent_id}] JSON parse error. Retrying... ({str(e)})")
            return generate_agent_action(agent_id, observation, retry_count + 1)
        else:
            print(f"  [{agent_id}] Failed to parse JSON after retry. Skipping turn.")
            return {
                "agent_id": agent_id,
                "action_type": "request_clarification",
                "content": "I encountered an internal error and must skip my turn.",
                "reasoning": "Fallback action due to JSON parse failure."
            }
    except Exception as e:
        print(f"  [{agent_id}] API Error: {str(e)}")
        if retry_count < 1:
            time.sleep(2)  # Delay before retry
            return generate_agent_action(agent_id, observation, retry_count + 1)
            
        print(f"  [{agent_id}] API Error persisting. Skipping turn.")
        return {
            "agent_id": agent_id,
            "action_type": "request_clarification",
            "content": "API error fallback.",
            "reasoning": "Fallback action due to API failure."
        }
def run_episode(task_id: str) -> Tuple[Optional[Dict[str, Any]], int]:
    """
    Runs a single episode for a given task_id by coordinating Agent A and Agent B sequentially.
    """
    print(f"[START] task_id={task_id}")
    
    try:
        # Reset the environment
        reset_resp = requests.post(f"{API_URL}/reset", json={"task_id": task_id})
        reset_resp.raise_for_status()
        state = reset_resp.json()
    except Exception as e:
        print(f"Failed to reset environment for task {task_id}: {str(e)}")
        return None, 0
        
    obs_a = state.get("obs_agent_a", {})
    obs_b = state.get("obs_agent_b", {})
    
    current_agent = "Agent A" # In api context, usually Agent A or Agent B. We use Agent A format
    # Mappings for id
    agent_id_map = {"Agent A": "Agent A", "Agent B": "Agent B"} 
    # Adjusting to generic "agent_a" / "agent_b" if env expects lowercase
    
    done = False
    turn_count = 0
    final_result = None
    
    while not done:
        turn_count += 1
        
        # Determine active agent parameters
        if current_agent == "Agent A":
            active_id = "agent_a"
            obs = obs_a
        else:
            active_id = "agent_b"
            obs = obs_b
            
        print(f"[STEP] turn={turn_count} agent={active_id} thinking...")
        
        # Get action from the LLM
        action = generate_agent_action(active_id, obs)
        print(f"[STEP] turn={turn_count} agent={active_id} action={action.get('action_type')}")
        
        try:
            # Step the environment
            step_resp = requests.post(f"{API_URL}/step", json={"action": action})
            
            # Display detailed error if HTTP 400 etc.
            if not step_resp.ok:
                print(f"  -> Environment API Error: {step_resp.text}")
                # We break entirely on environment failure as standard operation failed
                break
                
            step_data = step_resp.json()
            
            # Update observations
            obs_a = step_data.get("obs_agent_a", obs_a)
            obs_b = step_data.get("obs_agent_b", obs_b)
            done = step_data.get("done", False)
            
            if done:
                final_result = step_data.get("episode_result", {})
                print(f"[END] task_id={task_id} turns={turn_count}\n")
                break
                
        except Exception as e:
            print(f"  -> Failed to step environment: {str(e)}")
            break
            
        # Alternate turns
        current_agent = "Agent B" if current_agent == "Agent A" else "Agent A"
        
    return final_result, turn_count
def main():
    print(f"\n=== Social Agent Negotiation — Inference Script ===")
    print(f"Endpoint: {API_BASE_URL}")
    print(f"Model: {MODEL_NAME}\n")

    print("Checking Environment Health...")
    try:
        health_resp = requests.get(f"{API_URL}/health")
        health_resp.raise_for_status()
        print("Environment is healthy.")
    except Exception as e:
        print(f"ERROR: Cannot connect to Environment API at {API_URL}. Is it running?")
        print(str(e))
        sys.exit(1)

    print("\nFetching tasks...")
    try:
        tasks_resp = requests.get(f"{API_URL}/tasks")
        tasks_resp.raise_for_status()
        tasks = tasks_resp.json()
    except Exception as e:
        print(f"ERROR: Failed to fetch tasks: {str(e)}")
        sys.exit(1)

    def _safe(v):
        """Clamp a single float to strictly (0, 1)."""
        return round(max(0.0001, min(0.9999, v)), 4)

    def _sanitize(obj):
        """Recursively clamp ALL floats in dicts/lists to (0.0001, 0.9999)."""
        if obj is None:
            return obj
        if isinstance(obj, bool):
            return obj
        if isinstance(obj, float):
            return _safe(obj)
        if isinstance(obj, int):
            return obj
        if isinstance(obj, str):
            return obj
        if isinstance(obj, dict):
            return {k: _sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_sanitize(v) for v in obj]
        return obj

    results = []
    task_items = tasks.items() if isinstance(tasks, dict) else enumerate(tasks)

    for key, task_data in task_items:
        task_id = task_data.get("id", str(key))
        difficulty = task_data.get("difficulty", "Unknown")
        episode_result, turns_used = run_episode(task_id)

        score = 0.01
        consensus_reached = False
        if episode_result:
            raw_score = episode_result.get("total_reward", 0.5)
            score = _safe(raw_score)
            consensus_reached = episode_result.get("final_consensus") == "reached"

        results.append(_sanitize({
            "task_id": task_id,
            "difficulty": difficulty,
            "score": score,
            "turns_used": turns_used,
            "consensus_reached": consensus_reached,
            "raw_episode_result": episode_result
        }))

    with open("inference_results.json", "w") as f:
        json.dump(results, f, indent=4)
        print("\nSaved results to inference_results.json")

    print(f"\n--- Summary Results ---")
    print(f"{'Task ID':<25} \t      | {'Difficulty':<12} | {'Score':<8} | {'Turns':<8} | {'Consensus'}")
    print("-" * 75)
    for r in results:
        print(f"[END] task_id={r['task_id']} score={r['score']} turns={r['turns_used']} consensus={r['consensus_reached']}")
if __name__ == "__main__":
    main()