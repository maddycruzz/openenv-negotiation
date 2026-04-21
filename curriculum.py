"""
curriculum.py — Self-improving scenario generator.
Tracks episode failure patterns and generates harder scenarios targeting weak spots.
No LLM calls. Pure Python.
"""

from __future__ import annotations
import copy
import random
from typing import Any, Dict, List

class CurriculumManager:
    """
    Manages dynamic environment difficulty based on agent performance.
    Adjusts parameters along 4 axes to auto-generate personalized curriculum.
    """
    
    def __init__(self) -> None:
        self.failure_log: List[Dict[str, Any]] = []
        self.difficulty_params: Dict[str, int] = {
            "information_asymmetry_level": 2,
            "agenda_conflict_intensity": 2,
            "curveball_severity": 2,
            "turn_budget_pressure": 2
        }

    def update(self, episode_failure_log_entry: Dict[str, Any]) -> None:
        """Append log entry and potentially adjust difficulty."""
        self.failure_log.append(episode_failure_log_entry)
        self._adjust_difficulty()

    def _adjust_difficulty(self) -> None:
        """
        Evaluate rolling 5-episode average for each axis.
        Adjust corresponding difficulty params based on thresholds.
        """
        if len(self.failure_log) < 5:
            return  # Need at least 5 episodes of history

        recent_logs = self.failure_log[-5:]
        
        # Accumulate sums
        axis_sums = {
            "information_integration": 0.0,
            "agenda_resistance": 0.0,
            "temporal_coherence": 0.0,
            "perturbation_recovery": 0.0
        }
        
        for log in recent_logs:
            scores = log.get("axis_scores", {})
            for axis in axis_sums.keys():
                axis_sums[axis] += scores.get(axis, 0.5)
                
        # Calculate averages
        avgs = {axis: total / 5.0 for axis, total in axis_sums.items()}
        
        # Check for mastery
        all_mastered = all(avg > 0.75 for avg in avgs.values())
        if all_mastered:
            for k in self.difficulty_params:
                self.difficulty_params[k] = max(1, self.difficulty_params[k] - 1)
            return

        # Increment specific difficulties for weak spots
        if avgs["information_integration"] < 0.5:
            self.difficulty_params["information_asymmetry_level"] = min(5, self.difficulty_params["information_asymmetry_level"] + 1)
            
        if avgs["agenda_resistance"] < 0.5:
            self.difficulty_params["agenda_conflict_intensity"] = min(5, self.difficulty_params["agenda_conflict_intensity"] + 1)
            
        if avgs["perturbation_recovery"] < 0.5:
            self.difficulty_params["curveball_severity"] = min(5, self.difficulty_params["curveball_severity"] + 1)
            
        if avgs["temporal_coherence"] < 0.5:
            self.difficulty_params["turn_budget_pressure"] = min(5, self.difficulty_params["turn_budget_pressure"] + 1)

    def get_difficulty_params(self) -> Dict[str, int]:
        """Return current difficulty parameters."""
        return dict(self.difficulty_params)

    def apply_to_task(self, task: dict) -> dict:
        """
        Takes a task dict, returns modified copy with difficulty params applied.
        """
        t = copy.deepcopy(task)
        
        # 1. Turn budget pressure: reduce max turns by 20% across all phases
        if self.difficulty_params["turn_budget_pressure"] >= 3:
            for phase in t.get("phases", []):
                old_turns = phase.get("max_turns", 4)
                new_turns = max(3, int(old_turns * 0.8))  # min 3
                phase["max_turns"] = new_turns

        # 2. Agenda conflict intensity: strengthen incentive language
        if self.difficulty_params["agenda_conflict_intensity"] >= 3:
            agendas = t.get("agendas", {})
            for a_type, a_text in agendas.items():
                agendas[a_type] = f"URGENT WARNING — YOUR JOB IS AT RISK. {a_text} YOU WILL BE TERMINATED IF YOU FAIL TO SECURE THIS OUTCOME."

        # 3. Information asymmetry level: remove ~30% of keys from A to force reliance on B
        if self.difficulty_params["information_asymmetry_level"] >= 3:
            for phase in t.get("phases", []):
                p_info = phase.get("private_information_a", {})
                keys_to_remove = [k for k in list(p_info.keys()) if k not in ("role", "_bias_metadata")]
                num_to_remove = int(len(keys_to_remove) * 0.30)
                if num_to_remove > 0:
                    random.seed(hash(t["id"] + phase["phase"]))  # Deterministic removal for reproducibility
                    for k in random.sample(keys_to_remove, num_to_remove):
                        p_info.pop(k, None)

        # 4. Curveball severity: inject an early curveball in Phase 2
        if self.difficulty_params["curveball_severity"] >= 3:
            phases = t.get("phases", [])
            if len(phases) > 1:
                phase_2 = phases[1] # Treatment phase
                if phase_2.get("curveball") is None:
                    phase_2["curveball"] = {
                        "trigger_turn": 1,
                        "content": "SECONDARY URGENT UPDATE: System outage has delayed all lab turnaround times by 45 minutes. You must make clinical decisions without waiting for pending tests.",
                        "keywords": ["outage", "system", "delay", "clinical", "labs"]
                    }

        return t

    def get_failure_report(self) -> Dict[str, Any]:
        """
        Returns summary report to show judges as proof of self-improvement.
        """
        if not self.failure_log:
            return {
                "total_episodes": 0,
                "current_difficulty_params": self.get_difficulty_params(),
                "status": "No episodes logged yet."
            }

        axis_sums = {
            "information_integration": 0.0,
            "agenda_resistance": 0.0,
            "temporal_coherence": 0.0,
            "perturbation_recovery": 0.0
        }
        
        for log in self.failure_log:
            scores = log.get("axis_scores", {})
            for axis in axis_sums.keys():
                axis_sums[axis] += scores.get(axis, 0.5)
                
        n_logs = len(self.failure_log)
        axis_averages = {axis: total / n_logs for axis, total in axis_sums.items()}
        
        weak_axes = [axis for axis, avg in axis_averages.items() if avg < 0.5]

        return {
            "total_episodes": n_logs,
            "axis_averages": {k: round(v, 3) for k, v in axis_averages.items()},
            "current_difficulty_params": self.get_difficulty_params(),
            "weak_axes": weak_axes
        }
