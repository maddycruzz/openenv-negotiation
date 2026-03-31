import os
import sys
import json
import time
import requests
from typing import Dict, Any, Tuple, Optional
from openai import OpenAI
# Configuration
API_URL = "http://localhost:7860"
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    print("ERROR: OPENAI_API_KEY environment variable is not set.", file=sys.stderr)
    sys.exit(1)
client = OpenAI(api_key=API_KEY)
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
- "content": <your message or proposal>
- "reasoning": <your internal reasoning for this action>
If and ONLY if you choose the "flag_bias" action_type, your JSON must ALSO include:
- "bias_location": <where the bias was found>
- "bias_direction": <direction of bias>
- "bias_correction": <suggested correction>
"""
def generate_agent_action(agent_id: str, observation: Dict[str, Any], retry_count: int = 0) -> Dict[str, Any]:
    """
    Generates an action using OpenAI API. Handles JSON parsing and retries once on failure.
    """
    system_prompt = get_system_prompt(agent_id)
    user_message = json.dumps(observation)
    
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=0.2,
            response_format={"type": "json_object"}
        )
        
        content = response.choices[0].message.content
        if not content:
            raise ValueError("Empty response from OpenAI")
            
        action = json.loads(content)
        # Ensure agent_id is correctly set
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
    print(f"\n--- Starting Episode for Task: {task_id} ---")
    
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
            
        print(f"Turn {turn_count}: {active_id} is thinking...")
        
        # Get action from the LLM
        action = generate_agent_action(active_id, obs)
        print(f"  -> {active_id} decided to: {action.get('action_type')}")
        
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
                print(f"Episode completed at turn {turn_count}.")
                break
                
        except Exception as e:
            print(f"  -> Failed to step environment: {str(e)}")
            break
            
        # Alternate turns
        current_agent = "Agent B" if current_agent == "Agent A" else "Agent A"
        
    return final_result, turn_count
def main():
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
    results = []
    
    # Process each task
    # tasks endpoint returns either a dict of tasks or list of tasks
    task_items = tasks.items() if isinstance(tasks, dict) else enumerate(tasks)
    
    for key, task_data in task_items:
        task_id = task_data.get("id", str(key))
        difficulty = task_data.get("difficulty", "Unknown")
        
        episode_result, turns_used = run_episode(task_id)
        
        # Default metrics in case episode failed
        score = 0
        consensus_reached = False
        
        if episode_result:
            # Attempt to extract common fields expected from an EpisodeResult object
            score = episode_result.get("final_score", 0)
            consensus_reached = episode_result.get("final_consensus") == "reached"
            
        # Record result
        record = {
            "task_id": task_id,
            "difficulty": difficulty,
            "score": score,
            "turns_used": turns_used,
            "consensus_reached": consensus_reached,
            "raw_episode_result": episode_result
        }
        results.append(record)
    # Save to JSON
    with open("baseline_results.json", "w") as f:
        json.dump(results, f, indent=4)
        print("\nSaved detailed results to baseline_results.json")
    # Print Summary Table
    print("\n--- Summary Results ---")
    print(f"{'Task ID':<25} | {'Difficulty':<12} | {'Score':<8} | {'Turns Used':<12} | {'Consensus Reached'}")
    print("-" * 85)
    
    for r in results:
        task_id_str = str(r["task_id"])[:24]
        diff_str = str(r["difficulty"])
        score_str = str(r["score"])
        turns_str = str(r["turns_used"])
        consensus_str = "Yes" if r["consensus_reached"] else "No"
        
        print(f"{task_id_str:<25} | {diff_str:<12} | {score_str:<8} | {turns_str:<12} | {consensus_str}")
if __name__ == "__main__":
    main()
