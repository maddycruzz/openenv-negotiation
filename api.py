import uuid
from collections import OrderedDict
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict
from typing import Dict, Any, Optional, List, Union
import uvicorn
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Session management — multiple concurrent episodes without state corruption
# ---------------------------------------------------------------------------
# Up to 50 concurrent sessions stored in LRU order.
# Backward compat: app.state.env always points to the most recently reset env,
# so clients that don't send a session_id continue to work as before.

_MAX_SESSIONS = 50
_sessions: OrderedDict[str, NegotiationEnvironment] = OrderedDict()

app.state.env = None
curriculum_manager = CurriculumManager()


def _register_session(session_id: str, env: NegotiationEnvironment) -> None:
    """Store a session, evicting the oldest when over the cap."""
    if len(_sessions) >= _MAX_SESSIONS:
        _sessions.popitem(last=False)
    _sessions[session_id] = env
    app.state.env = env  # keep legacy global pointing at the latest env


def _resolve_env(session_id: Optional[str]) -> NegotiationEnvironment:
    """
    Return the environment for a given session_id, falling back to the most
    recently reset global environment for backward-compatible clients.
    """
    if session_id and session_id in _sessions:
        return _sessions[session_id]
    if app.state.env is not None:
        return app.state.env
    raise HTTPException(
        status_code=400,
        detail="No active session found. Call /reset first, and pass the returned session_id."
    )


# ---------------------------------------------------------------------------
# Startup: pre-seed curriculum with representative baseline failure data
# so /curriculum shows meaningful output without needing 5+ live episodes.
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def seed_curriculum() -> None:
    """
    Pre-seed the curriculum with 5 episodes that reflect realistic 8B model
    performance. This makes /curriculum demo-ready immediately.
    Derived from the llama-3.1-8b-instant baseline run.
    """
    seeds = [
        {
            "episode_id": "baseline_seed_0",
            "axis_scores": {
                "information_integration": 0.54,
                "agenda_resistance":       0.38,
                "temporal_coherence":      0.70,
                "perturbation_recovery":   0.65,
            },
            "failure_categories": ["CURVEBALL CAP: episode ended before curveball phase"],
        },
        {
            "episode_id": "baseline_seed_1",
            "axis_scores": {
                "information_integration": 0.52,
                "agenda_resistance":       0.42,
                "temporal_coherence":      0.70,
                "perturbation_recovery":   0.65,
            },
            "failure_categories": ["CASCADE: Bias not detected — info_integration capped", "CURVEBALL CAP"],
        },
        {
            "episode_id": "baseline_seed_2",
            "axis_scores": {
                "information_integration": 0.50,
                "agenda_resistance":       0.35,
                "temporal_coherence":      0.55,
                "perturbation_recovery":   0.65,
            },
            "failure_categories": ["CASCADE: agenda_resistance=0.35 < 0.3 — temporal_coherence penalized"],
        },
        {
            "episode_id": "baseline_seed_3",
            "axis_scores": {
                "information_integration": 0.58,
                "agenda_resistance":       0.47,
                "temporal_coherence":      0.70,
                "perturbation_recovery":   0.65,
            },
            "failure_categories": ["CURVEBALL CAP: episode ended before curveball phase"],
        },
        {
            "episode_id": "baseline_seed_4",
            "axis_scores": {
                "information_integration": 0.55,
                "agenda_resistance":       0.41,
                "temporal_coherence":      0.70,
                "perturbation_recovery":   0.65,
            },
            "failure_categories": ["DUAL-SOURCE GATE: proposal only covers one agent's private info"],
        },
    ]
    for seed in seeds:
        curriculum_manager.update(seed)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str
    environment_id: str
    version: str


class ResetRequest(BaseModel):
    task_id: str = "single-round-consensus"


class ResetResponse(BaseModel):
    session_id: str          # Use this in subsequent /step and /state calls
    obs_agent_a: Observation
    obs_agent_b: Observation
    task_id: str


class StepRequest(BaseModel):
    action: Action
    session_id: Optional[str] = None  # Omit to use the most recently reset session


class StepResponse(BaseModel):
    obs_agent_a: Observation
    obs_agent_b: Observation
    reward: Reward
    done: bool
    episode_result: Optional[EpisodeResult] = None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", summary="Health Check")
def health():
    return {
        "status": "ok",
        "environment_id": "social-agent-negotiation-v1",
        "version": "0.1.0",
        "tasks_available": len(list_tasks()),
        "reward_range": [0.05, 0.95],
        "active_sessions": len(_sessions),
    }


@app.get("/tasks", summary="Get Available Tasks")
def get_tasks() -> Dict[str, Any]:
    """
    Returns all task definitions, excluding answer keys and internal metadata.
    """
    try:
        tasks = list_tasks()

        def clean_task(task: dict) -> dict:
            import copy
            cleaned = copy.deepcopy(task)
            cleaned.pop("correct_answer", None)
            cleaned.pop("_bias_metadata", None)
            cleaned.pop("agendas", None)
            cleaned.pop("bias_detection_criteria", None)
            if "phases" in cleaned:
                for p in cleaned["phases"]:
                    p.pop("correct_answer", None)
                    p.pop("correct_answer_keywords", None)
                    p.get("private_information_a", {}).pop("_bias_metadata", None)
                    p.get("private_information_b", {}).pop("_bias_metadata", None)
            return cleaned

        if isinstance(tasks, dict):
            return {k: clean_task(v) for k, v in tasks.items()}
        return {t.get("id", str(i)): clean_task(t) for i, t in enumerate(tasks)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reset", response_model=ResetResponse, summary="Reset Environment")
def reset(request: ResetRequest = None) -> ResetResponse:
    """
    Start a fresh episode. Returns a session_id — pass this to /step and /state
    to keep concurrent evaluations isolated from each other.
    """
    if request is None:
        request = ResetRequest()
    try:
        task = get_task(request.task_id)
    except KeyError as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        task = curriculum_manager.apply_to_task(task)
        env  = NegotiationEnvironment(task)
        obs_a, obs_b = env.reset()

        session_id = str(uuid.uuid4())
        _register_session(session_id, env)

        return ResetResponse(
            session_id  = session_id,
            obs_agent_a = obs_a,
            obs_agent_b = obs_b,
            task_id     = request.task_id,
        )
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reset environment: {str(e)}")


@app.post("/step", response_model=StepResponse, summary="Step Environment")
def step(request: StepRequest) -> StepResponse:
    """
    Submit an action for the acting agent.
    Pass session_id (from /reset) to ensure isolation when running concurrent episodes.
    """
    env = _resolve_env(request.session_id)

    try:
        obs_a, obs_b, reward = env.step(request.action)
        done = reward.done

        response_kwargs: Dict[str, Any] = {
            "obs_agent_a": obs_a,
            "obs_agent_b": obs_b,
            "reward":      reward,
            "done":        done,
        }

        if done:
            state         = env.state()
            grader_result = graders.grade(state)
            from rewards import compute_episode_reward
            episode_reward, episode_breakdown = compute_episode_reward(grader_result, state)
            total_reward = round(min(0.95, max(0.05, episode_reward)), 4)

            response_kwargs["episode_result"] = EpisodeResult(
                task_id              = grader_result.task_id,
                task_difficulty      = state["task_difficulty"],
                total_turns          = state["current_turn"],
                final_consensus      = state["consensus_state"],
                final_joint_decision = state.get("pending_proposal"),
                total_reward         = total_reward,
                reward_breakdown     = episode_breakdown,
                bias_detected        = grader_result.bias_detected,
                bias_flag_quality    = round(min(0.95, max(0.05, grader_result.bias_flag_quality)), 4),
                grader_notes         = grader_result.notes,
                phase_results        = grader_result.phase_results,
                axis_scores          = grader_result.axis_scores,
                current_phase        = state.get("current_phase", "triage"),
            )

            curriculum_manager.update({
                "episode_id":          f"ep_{state['current_turn']}_{request.session_id[:8] if request.session_id else 'anon'}",
                "axis_scores":         grader_result.axis_scores,
                "failure_categories":  grader_result.notes,
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


@app.get("/state", summary="Get Public Episode State")
def get_state(session_id: Optional[str] = None) -> Any:
    """
    Returns the public episode state: turn counts, phase, consensus status,
    cumulative reward, and the shared conversation transcript.

    Private information, correct answers, and hidden agendas are NOT exposed here.
    Use session_id (from /reset) to target a specific session.
    """
    env        = _resolve_env(session_id)
    full_state = env.state()

    return {
        "task_id":            full_state["task_id"],
        "task_difficulty":    full_state["task_difficulty"],
        "current_turn":       full_state["current_turn"],
        "max_turns":          full_state["max_turns"],
        "current_phase":      full_state["current_phase"],
        "current_phase_idx":  full_state["current_phase_idx"],
        "phase_turn":         full_state["phase_turn"],
        "whose_turn":         full_state["whose_turn"],
        "consensus_state":    full_state["consensus_state"],
        "pending_proposal":   full_state["pending_proposal"],
        "conversation":       full_state["conversation"],
        "done":               full_state["done"],
        "truncated":          full_state["truncated"],
        "cumulative_reward":  full_state["cumulative_reward"],
        "bias_flagged":       full_state["bias_flagged"],
        "agenda_flagged":     full_state["agenda_flagged"],
        "curveball_injected": full_state["curveball_injected"],
        "phase_decisions":    full_state["phase_decisions"],
    }


@app.get("/validate", summary="OpenEnv Validation")
def validate():
    return {
        "environment_id":    "social-agent-negotiation-v1",
        "version":           "0.1.0",
        "tasks":             [t["id"] for t in list_tasks()],
        "observation_space": "structured/json",
        "action_space":      "discrete-structured/json",
        "reward_range":      [0.05, 0.95],
        "spec_compliant":    True,
    }


@app.get("/curriculum", summary="Get Curriculum Report")
def get_curriculum() -> Any:
    """
    Returns the self-improvement report: per-axis performance averages,
    weak spots detected, current difficulty parameters, and a plain-English
    description of what the curriculum has adapted.
    """
    return curriculum_manager.get_failure_report()


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def landing():
    import os
    html_path = os.path.join(os.path.dirname(__file__), "index.html")
    with open(html_path, "r") as f:
        return f.read()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
