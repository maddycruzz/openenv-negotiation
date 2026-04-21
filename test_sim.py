from environment import NegotiationEnvironment
from tasks import list_tasks
from models import Action, ActionType, AgentID
from graders import grade
import api # loads to check if api imports cleanly with curriculum manager

for t in list_tasks():
    print("Testing task:", t["id"])
    env = NegotiationEnvironment(t)
    obs_a, obs_b = env.reset()
    assert obs_a.current_turn == 0
    assert obs_a.agent_id == AgentID.AGENT_A

    # Submit action
    action = Action(
        agent_id=AgentID.AGENT_A,
        action_type=ActionType.PROPOSE_CONSENSUS,
        content="Let's agree on this.",
        reasoning="Because it makes sense."
    )
    obs_a, obs_b, reward = env.step(action)
    print("  Turn:", obs_a.current_turn, "Consensus:", obs_a.current_consensus_state)

    # Let agent B accept
    action2 = Action(
        agent_id=AgentID.AGENT_B,
        action_type=ActionType.ACCEPT_CONSENSUS,
        content="I accept.",
        reasoning="I agree."
    )
    obs_a, obs_b, reward = env.step(action2)
    print("  Phase completed:", obs_a.current_phase.value)
    
print("All tasks initialized and can run phases!")

