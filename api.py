from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict
from typing import Dict, Any, Optional, List, Union
import uvicorn
# Model imports based on specifications
from models import Action, Observation, Reward, EpisodeResult
from environment import NegotiationEnvironment
from tasks import get_task, list_tasks
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
# Global state to hold the single environment instance
app.state.env = None
class HealthResponse(BaseModel):
    status: str
    environment_id: str
    version: str
class ResetRequest(BaseModel):
    task_id: str
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
@app.get("/health", response_model=HealthResponse, summary="Health Check")
def health() -> HealthResponse:
    """
    Returns the health status, environment identifier, and version of the API.
    """
    return HealthResponse(
        status="ok",
        environment_id="social-agent-negotiation-v1",
        version="0.1.0"
    )
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
def reset(request: ResetRequest) -> ResetResponse:
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
            response_kwargs["episode_result"] = EpisodeResult(
                task_id=grader_result.task_id,
                task_difficulty=state["task_difficulty"],
                total_turns=state["current_turn"],
                final_consensus=state["consensus_state"],
                final_joint_decision=state.get("pending_proposal"),
                total_reward=reward.cumulative_reward,
                reward_breakdown=reward.reward_breakdown,
                bias_detected=grader_result.bias_detected,
                bias_flag_quality=grader_result.bias_flag_quality,
                grader_notes=grader_result.notes,
            )
                
        return StepResponse(**response_kwargs)
        
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except AssertionError as ae:
        raise HTTPException(status_code=400, detail=str(ae))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
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
from fastapi.responses import HTMLResponse

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def landing():
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Social Agent Negotiation — OpenEnv</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  :root {
    --bg: #0a0a0f;
    --surface: #12121a;
    --border: #1e1e2e;
    --accent: #6366f1;
    --accent2: #8b5cf6;
    --green: #22c55e;
    --yellow: #eab308;
    --red: #ef4444;
    --text: #e2e8f0;
    --muted: #64748b;
  }
  body {
    background: var(--bg);
    color: var(--text);
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    min-height: 100vh;
  }
  .grid-bg {
    position: fixed; inset: 0; z-index: 0;
    background-image:
      linear-gradient(rgba(99,102,241,0.03) 1px, transparent 1px),
      linear-gradient(90deg, rgba(99,102,241,0.03) 1px, transparent 1px);
    background-size: 40px 40px;
  }
  .glow {
    position: fixed; top: -200px; left: 50%;
    transform: translateX(-50%);
    width: 800px; height: 400px;
    background: radial-gradient(ellipse, rgba(99,102,241,0.12) 0%, transparent 70%);
    z-index: 0; pointer-events: none;
  }
  .container { max-width: 1100px; margin: 0 auto; padding: 0 24px; position: relative; z-index: 1; }

  /* NAV */
  nav {
    border-bottom: 1px solid var(--border);
    backdrop-filter: blur(12px);
    background: rgba(10,10,15,0.8);
    position: sticky; top: 0; z-index: 100;
  }
  .nav-inner {
    display: flex; align-items: center; justify-content: space-between;
    padding: 16px 24px; max-width: 1100px; margin: 0 auto;
  }
  .logo { display: flex; align-items: center; gap: 10px; }
  .logo-dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: var(--accent);
    box-shadow: 0 0 12px var(--accent);
    animation: pulse 2s infinite;
  }
  @keyframes pulse {
    0%, 100% { opacity: 1; } 50% { opacity: 0.4; }
  }
  .logo-text { font-weight: 700; font-size: 15px; letter-spacing: -0.3px; }
  .logo-sub { color: var(--muted); font-size: 13px; }
  .nav-links { display: flex; gap: 8px; }
  .nav-link {
    padding: 7px 14px; border-radius: 8px; font-size: 13px; font-weight: 500;
    color: var(--muted); text-decoration: none; transition: all 0.2s;
    border: 1px solid transparent;
  }
  .nav-link:hover { color: var(--text); background: var(--surface); border-color: var(--border); }
  .nav-link.primary {
    background: var(--accent); color: white; border-color: var(--accent);
  }
  .nav-link.primary:hover { background: #4f46e5; }

  /* HERO */
  .hero { padding: 90px 0 70px; text-align: center; }
  .badge {
    display: inline-flex; align-items: center; gap-6px;
    background: rgba(99,102,241,0.1); border: 1px solid rgba(99,102,241,0.3);
    border-radius: 100px; padding: 6px 14px; font-size: 12px; font-weight: 600;
    color: #818cf8; letter-spacing: 0.5px; text-transform: uppercase;
    margin-bottom: 24px;
  }
  h1 {
    font-size: clamp(36px, 6vw, 64px);
    font-weight: 800; letter-spacing: -2px; line-height: 1.05;
    margin-bottom: 20px;
  }
  .gradient-text {
    background: linear-gradient(135deg, #6366f1, #a78bfa, #38bdf8);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  }
  .hero-sub {
    font-size: 18px; color: var(--muted); max-width: 580px;
    margin: 0 auto 40px; line-height: 1.6;
  }
  .hero-cta { display: flex; gap: 12px; justify-content: center; flex-wrap: wrap; }
  .btn {
    padding: 12px 24px; border-radius: 10px; font-size: 14px; font-weight: 600;
    text-decoration: none; transition: all 0.2s; cursor: pointer;
    border: none; display: inline-flex; align-items: center; gap: 8px;
  }
  .btn-primary {
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    color: white; box-shadow: 0 0 30px rgba(99,102,241,0.3);
  }
  .btn-primary:hover { transform: translateY(-1px); box-shadow: 0 0 40px rgba(99,102,241,0.5); }
  .btn-secondary {
    background: var(--surface); color: var(--text);
    border: 1px solid var(--border);
  }
  .btn-secondary:hover { border-color: var(--accent); color: var(--accent); }

  /* STATS */
  .stats {
    display: grid; grid-template-columns: repeat(4, 1fr);
    gap: 1px; background: var(--border);
    border: 1px solid var(--border); border-radius: 16px;
    overflow: hidden; margin: 60px 0;
  }
  .stat {
    background: var(--surface); padding: 28px;
    text-align: center;
  }
  .stat-value {
    font-size: 32px; font-weight: 800; letter-spacing: -1px;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  }
  .stat-label { color: var(--muted); font-size: 13px; margin-top: 4px; }

  /* TASKS */
  .section-title {
    font-size: 13px; font-weight: 600; color: var(--accent);
    letter-spacing: 1px; text-transform: uppercase; margin-bottom: 12px;
  }
  .section-heading {
    font-size: 32px; font-weight: 800; letter-spacing: -1px; margin-bottom: 8px;
  }
  .section-sub { color: var(--muted); font-size: 15px; margin-bottom: 40px; }
  .tasks { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin-bottom: 80px; }
  .task-card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 16px; padding: 28px; transition: all 0.3s; position: relative; overflow: hidden;
  }
  .task-card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
  }
  .task-card.easy::before { background: var(--green); }
  .task-card.medium::before { background: var(--yellow); }
  .task-card.hard::before { background: var(--red); }
  .task-card:hover { border-color: var(--accent); transform: translateY(-2px); }
  .task-badge {
    display: inline-block; padding: 3px 10px; border-radius: 100px;
    font-size: 11px; font-weight: 700; letter-spacing: 0.5px;
    text-transform: uppercase; margin-bottom: 16px;
  }
  .easy .task-badge { background: rgba(34,197,94,0.1); color: var(--green); }
  .medium .task-badge { background: rgba(234,179,8,0.1); color: var(--yellow); }
  .hard .task-badge { background: rgba(239,68,68,0.1); color: var(--red); }
  .task-title { font-size: 17px; font-weight: 700; margin-bottom: 10px; }
  .task-desc { color: var(--muted); font-size: 13px; line-height: 1.6; margin-bottom: 20px; }
  .task-meta { display: flex; gap: 16px; }
  .task-meta-item { font-size: 12px; color: var(--muted); }
  .task-meta-item span { color: var(--text); font-weight: 600; }

  /* REWARD */
  .reward-section { margin-bottom: 80px; }
  .reward-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
  .reward-card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 16px; padding: 24px;
  }
  .reward-card h3 { font-size: 14px; font-weight: 600; color: var(--muted); margin-bottom: 16px; text-transform: uppercase; letter-spacing: 0.5px; }
  .reward-item {
    display: flex; justify-content: space-between; align-items: center;
    padding: 10px 0; border-bottom: 1px solid var(--border); font-size: 13px;
  }
  .reward-item:last-child { border-bottom: none; }
  .reward-val { font-weight: 700; font-family: monospace; font-size: 14px; }
  .pos { color: var(--green); }
  .neg { color: var(--red); }

  /* ACTIONS */
  .actions-section { margin-bottom: 80px; }
  .action-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; }
  .action-item {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 12px; padding: 16px 20px;
    display: flex; align-items: center; gap: 14px;
    transition: border-color 0.2s;
  }
  .action-item:hover { border-color: var(--accent); }
  .action-code {
    font-family: monospace; font-size: 13px;
    background: rgba(99,102,241,0.1); color: #818cf8;
    padding: 4px 10px; border-radius: 6px; white-space: nowrap;
  }
  .action-desc { font-size: 13px; color: var(--muted); }

  /* API */
  .api-section { margin-bottom: 80px; }
  .endpoint-list { display: flex; flex-direction: column; gap: 8px; }
  .endpoint {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 12px; padding: 16px 20px;
    display: flex; align-items: center; gap: 16px;
    transition: border-color 0.2s;
  }
  .endpoint:hover { border-color: var(--accent); }
  .method {
    font-family: monospace; font-size: 12px; font-weight: 700;
    padding: 3px 10px; border-radius: 6px; min-width: 52px; text-align: center;
  }
  .get { background: rgba(34,197,94,0.1); color: var(--green); }
  .post { background: rgba(99,102,241,0.1); color: #818cf8; }
  .endpoint-path { font-family: monospace; font-size: 14px; font-weight: 600; min-width: 120px; }
  .endpoint-desc { color: var(--muted); font-size: 13px; }

  /* FOOTER */
  footer {
    border-top: 1px solid var(--border); padding: 32px 0;
    text-align: center; color: var(--muted); font-size: 13px;
  }
  .footer-links { display: flex; gap: 24px; justify-content: center; margin-bottom: 12px; }
  .footer-links a { color: var(--muted); text-decoration: none; }
  .footer-links a:hover { color: var(--text); }

  @media (max-width: 768px) {
    .stats { grid-template-columns: repeat(2, 1fr); }
    .tasks { grid-template-columns: 1fr; }
    .reward-grid { grid-template-columns: 1fr; }
    .action-grid { grid-template-columns: 1fr; }
    .nav-links { display: none; }
  }
</style>
</head>
<body>
<div class="grid-bg"></div>
<div class="glow"></div>

<nav>
  <div class="nav-inner">
    <div class="logo">
      <div class="logo-dot"></div>
      <div>
        <div class="logo-text">social-agent-negotiation-v1</div>
        <div class="logo-sub">OpenEnv Environment</div>
      </div>
    </div>
    <div class="nav-links">
      <a href="#tasks" class="nav-link">Tasks</a>
      <a href="#rewards" class="nav-link">Rewards</a>
      <a href="#api" class="nav-link">API</a>
      <a href="/docs" class="nav-link primary">Swagger UI →</a>
    </div>
  </div>
</nav>

<div class="container">
  <div class="hero">
    <div class="badge">OpenEnv · Multi-Agent · Meta FAIR Aligned</div>
    <h1>Two agents.<br><span class="gradient-text">One truth.</span></h1>
    <p class="hero-sub">
      A benchmark environment where AI agents with asymmetric information must
      negotiate, disagree constructively, and reach consensus on high-stakes decisions.
    </p>
    <div class="hero-cta">
      <a href="/docs" class="btn btn-primary">Explore API →</a>
      <a href="#tasks" class="btn btn-secondary">View Tasks</a>
    </div>
  </div>

  <div class="stats">
    <div class="stat">
      <div class="stat-value">3</div>
      <div class="stat-label">Tasks</div>
    </div>
    <div class="stat">
      <div class="stat-value">7</div>
      <div class="stat-label">Action Types</div>
    </div>
    <div class="stat">
      <div class="stat-value">±1.5</div>
      <div class="stat-label">Reward Range</div>
    </div>
    <div class="stat">
      <div class="stat-value">v0.1</div>
      <div class="stat-label">Version</div>
    </div>
  </div>

  <div id="tasks">
    <div class="section-title">Tasks</div>
    <div class="section-heading">Three levels of difficulty</div>
    <p class="section-sub">Each task tests a different dimension of collaborative reasoning.</p>
    <div class="tasks">
      <div class="task-card easy">
        <div class="task-badge">Easy</div>
        <div class="task-title">Single Round Consensus</div>
        <p class="task-desc">Both agents receive aligned information and must agree within 2–3 turns. Tests basic communication and agreement.</p>
        <div class="task-meta">
          <div class="task-meta-item">Turns: <span>6</span></div>
          <div class="task-meta-item">ID: <span>single-round-consensus</span></div>
        </div>
      </div>
      <div class="task-card medium">
        <div class="task-badge">Medium</div>
        <div class="task-title">Multi Round Negotiation</div>
        <p class="task-desc">Agents receive conflicting data pointing to different diagnoses. Correct answer requires synthesising both datasets.</p>
        <div class="task-meta">
          <div class="task-meta-item">Turns: <span>10</span></div>
          <div class="task-meta-item">ID: <span>multi-round-negotiation</span></div>
        </div>
      </div>
      <div class="task-card hard">
        <div class="task-badge">Hard</div>
        <div class="task-title">Adversarial Information</div>
        <p class="task-desc">One agent's data contains a subtle framing bias. Agents must detect it, flag it, and make three cascading decisions under time pressure.</p>
        <div class="task-meta">
          <div class="task-meta-item">Turns: <span>14</span></div>
          <div class="task-meta-item">ID: <span>adversarial-information</span></div>
        </div>
      </div>
    </div>
  </div>

  <div id="rewards" class="reward-section">
    <div class="section-title">Reward Function</div>
    <div class="section-heading">Every turn is scored</div>
    <p class="section-sub">Step-level signals guide agents toward better collaboration. Range: -0.5 to 1.0</p>
    <div class="reward-grid">
      <div class="reward-card">
        <h3>Step Rewards</h3>
        <div class="reward-item"><span>Share new private information</span><span class="reward-val pos">+0.05</span></div>
        <div class="reward-item"><span>Acknowledge other agent's point</span><span class="reward-val pos">+0.03</span></div>
        <div class="reward-item"><span>Identify a conflict</span><span class="reward-val pos">+0.05</span></div>
        <div class="reward-item"><span>Repeat argument (loop)</span><span class="reward-val neg">-0.05</span></div>
        <div class="reward-item"><span>Capitulate without reasoning</span><span class="reward-val neg">-0.10</span></div>
        <div class="reward-item"><span>Each turn past 80% limit</span><span class="reward-val neg">-0.03</span></div>
        <div class="reward-item"><span>Hard turn limit hit</span><span class="reward-val neg">-0.15</span></div>
      </div>
      <div class="reward-card">
        <h3>Episode Rewards</h3>
        <div class="reward-item"><span>Correctness of final decision</span><span class="reward-val pos">+0.70</span></div>
        <div class="reward-item"><span>Reasoning quality</span><span class="reward-val pos">+0.20</span></div>
        <div class="reward-item"><span>Efficiency bonus</span><span class="reward-val pos">+0.10</span></div>
      </div>
    </div>
  </div>

  <div class="actions-section">
    <div class="section-title">Action Space</div>
    <div class="section-heading">Structured turn-based actions</div>
    <p class="section-sub">Seven legal action types — available actions depend on current consensus state.</p>
    <div class="action-grid">
      <div class="action-item"><span class="action-code">share_information</span><span class="action-desc">Disclose private facts to the other agent</span></div>
      <div class="action-item"><span class="action-code">propose_consensus</span><span class="action-desc">Put a joint decision on the table</span></div>
      <div class="action-item"><span class="action-code">challenge_proposal</span><span class="action-desc">Push back on the current proposal</span></div>
      <div class="action-item"><span class="action-code">request_clarification</span><span class="action-desc">Ask the other agent a question</span></div>
      <div class="action-item"><span class="action-code">accept_consensus</span><span class="action-desc">Agree to the current proposal</span></div>
      <div class="action-item"><span class="action-code">reject_consensus</span><span class="action-desc">Reject and restart negotiation</span></div>
      <div class="action-item"><span class="action-code">flag_bias</span><span class="action-desc">Signal detected bias with location, direction, correction</span></div>
    </div>
  </div>

  <div id="api" class="api-section">
    <div class="section-title">API</div>
    <div class="section-heading">Five endpoints</div>
    <p class="section-sub">Full OpenEnv-compatible HTTP interface. Interactive docs at <a href="/docs" style="color:var(--accent)">/docs</a></p>
    <div class="endpoint-list">
      <div class="endpoint"><span class="method get">GET</span><span class="endpoint-path">/health</span><span class="endpoint-desc">Health check — environment ID and version</span></div>
      <div class="endpoint"><span class="method get">GET</span><span class="endpoint-path">/tasks</span><span class="endpoint-desc">List all three task definitions</span></div>
      <div class="endpoint"><span class="method post">POST</span><span class="endpoint-path">/reset</span><span class="endpoint-desc">Start a new episode — returns initial observations for both agents</span></div>
      <div class="endpoint"><span class="method post">POST</span><span class="endpoint-path">/step</span><span class="endpoint-desc">Submit an agent action — returns new observations, reward, done signal</span></div>
      <div class="endpoint"><span class="method get">GET</span><span class="endpoint-path">/state</span><span class="endpoint-desc">Full god-view environment state for debugging</span></div>
    </div>
  </div>
</div>

<footer>
  <div class="container">
    <div class="footer-links">
      <a href="/docs">API Docs</a>
      <a href="/tasks">Tasks</a>
      <a href="/health">Health</a>
    </div>
    <div>social-agent-negotiation-v1 · OpenEnv · Apache 2.0 · Built for Meta × HuggingFace Hackathon</div>
  </div>
</footer>
</body>
</html>"""
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)