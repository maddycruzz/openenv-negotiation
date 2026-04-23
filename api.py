from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict
from typing import Dict, Any, Optional, List, Union
import uvicorn
# Model imports based on specifications
from models import Action, Observation, Reward, EpisodeResult
from environment import NegotiationEnvironment
from tasks import get_task, list_tasks
from curriculum import CurriculumManager
import graders

app = FastAPI(
    title="Negotiation Environment API",
    description="API for social-agent-negotiation-v1",
    version="0.1.0"
)
# CORS enabled for all origins (HuggingFace Spaces requirement)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Global state
app.state.env = None
curriculum_manager = CurriculumManager()

class HealthResponse(BaseModel):
    status: str
    environment_id: str
    version: str
class ResetRequest(BaseModel):
    task_id: str = "single-round-consensus"
class ResetResponse(BaseModel):
    obs_agent_a: Observation
    obs_agent_b: Observation
    task_id: str
class StepRequest(BaseModel):
    action: Action
class StepResponse(BaseModel):
    obs_agent_a: Observation
    obs_agent_b: Observation
    reward: Reward
    done: bool
    episode_result: Optional[EpisodeResult] = None
@app.get("/health", summary="Health Check")
def health():
    """
    Returns the health status, environment identifier, and version of the API.
    """
    return {
        "status": "ok",
        "environment_id": "social-agent-negotiation-v1",
        "version": "0.1.0",
        "tasks_available": len(list_tasks()),
        "reward_range": [0.01, 0.99]
    }
@app.get("/tasks", summary="Get Available Tasks")
def get_tasks() -> Dict[str, Any]:
    """
    Returns all task definitions, excluding internal fields like 
    correct_answer and _bias_metadata.
    """
    try:
        tasks = list_tasks()
        
        # Helper to clean a single task definition
        def clean_task(task: dict) -> dict:
            cleaned = dict(task)
            cleaned.pop("correct_answer", None)
            cleaned.pop("_bias_metadata", None)
            cleaned.pop("agendas", None)
            cleaned.pop("bias_detection_criteria", None)
            if "phases" in cleaned:
                # Deep copy to avoid mutating the original registry
                cleaned["phases"] = [dict(p) for p in cleaned["phases"]]
                for p in cleaned["phases"]:
                    p.pop("correct_answer", None)
                    p.pop("correct_answer_keywords", None)
            return cleaned
        if isinstance(tasks, dict):
            return {k: clean_task(v) for k, v in tasks.items()}
        elif isinstance(tasks, list):
            return {t.get("id", str(i)): clean_task(t) for i, t in enumerate(tasks)}
        else:
            return tasks
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
@app.post("/reset", response_model=ResetResponse, summary="Reset Environment")
def reset(request: ResetRequest = None) -> ResetResponse:
    if request is None:
        request = ResetRequest()
    """
    Creates a new NegotiationEnvironment with the requested task.
    Returns the initial observations for both agents and the task_id.
    """
    try:
        # Check task
        from tasks import get_task
        try:
            task = get_task(request.task_id)
        except KeyError as e:
            raise HTTPException(status_code=400, detail=str(e))
        
        # Apply curriculum difficulty scaling before creating env
        task = curriculum_manager.apply_to_task(task)
        env = NegotiationEnvironment(task)
        
        # Depending on exactly what API env.reset() implements
        obs_a, obs_b = env.reset()
        
        # Store globally
        app.state.env = env
        
        return ResetResponse(
            obs_agent_a=obs_a,
            obs_agent_b=obs_b,
            task_id=request.task_id
        )
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reset environment: {str(e)}")
@app.post("/step", response_model=StepResponse, summary="Step Environment")
def step(request: StepRequest) -> StepResponse:
    """
    Take an action in the environment.
    Returns observations, reward, done status, and if done, the episode result.
    """
    env = app.state.env
    if env is None:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")
    
    try:
        obs_a, obs_b, reward = env.step(request.action)
        done = reward.done
            
        response_kwargs = {
            "obs_agent_a": obs_a,
            "obs_agent_b": obs_b,
            "reward": reward,
            "done": done
        }
        
        if done:
            state = env.state()
            grader_result = graders.grade(state)
            from rewards import compute_episode_reward
            episode_reward, episode_breakdown = compute_episode_reward(grader_result, state)
            total_reward = round(min(0.99, max(0.01, episode_reward)), 4)
            response_kwargs["episode_result"] = EpisodeResult(
                task_id=grader_result.task_id,
                task_difficulty=state["task_difficulty"],
                total_turns=state["current_turn"],
                final_consensus=state["consensus_state"],
                final_joint_decision=state.get("pending_proposal"),
                total_reward=total_reward,
                reward_breakdown=episode_breakdown,
                bias_detected=grader_result.bias_detected,
                bias_flag_quality=round(min(0.95, max(0.05, grader_result.bias_flag_quality)), 4),
                grader_notes=grader_result.notes,
                phase_results=grader_result.phase_results,
                axis_scores=grader_result.axis_scores,
                current_phase=state.get("current_phase", "triage")
            )
            
            # Log failure to curriculum manager
            curriculum_manager.update({
                "episode_id": f"ep_{env._current_turn}",
                "axis_scores": grader_result.axis_scores,
                "failure_categories": grader_result.notes
            })
                
        return StepResponse(**response_kwargs)
        
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except AssertionError as ae:
        raise HTTPException(status_code=400, detail=str(ae))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/validate", summary="OpenEnv Validation")
def validate():
    return {
        "environment_id": "social-agent-negotiation-v1",
        "version": "0.1.0",
        "tasks": [t["id"] for t in list_tasks()],
        "observation_space": "structured/json",
        "action_space": "discrete-structured/json",
        "reward_range": [0.01, 0.95],
        "spec_compliant": True
    }

@app.get("/curriculum", summary="Get Curriculum Report")
def get_curriculum() -> Any:
    """
    Returns the failure trace report and current difficulty parameters.
    Used for hackathon judge presentation.
    """
    return curriculum_manager.get_failure_report()

@app.get("/state", summary="Get Full State")
def get_state() -> Any:
    """
    Returns the full god-view state dict of the environment.
    """
    env = app.state.env
    if env is None:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")
    
    try:
        return env.state()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def landing():
    import os
    html_path = os.path.join(os.path.dirname(__file__), "index.html")
    with open(html_path, "r") as f:
        return f.read()
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)