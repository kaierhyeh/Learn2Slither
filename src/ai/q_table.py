"""Q-learning Agent"""

import json
import os
import random
from typing import Any, Dict

from config.ai import DEFAULT_ALPHA, DEFAULT_EPSILON, DEFAULT_GAMMA
from config.game import DIRECTIONS


class QTableAgent:
    """Q-learning agent for decision making and persistence."""

    def __init__(
        self,
        alpha: float = DEFAULT_ALPHA,
        gamma: float = DEFAULT_GAMMA,
        epsilon: float = DEFAULT_EPSILON,
        dontlearn: bool = False,
    ):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = 0.0 if dontlearn else epsilon
        self.dontlearn = dontlearn
        self.actions = list(DIRECTIONS.keys())
        self.q_table: Dict[str, Dict[str, float]] = {}

    def get_q_values(self, state: Any) -> Dict[str, float]:
        """Get or initialize Q-values for state."""
        key = str(state)
        if key not in self.q_table:
            self.q_table[key] = {act: 0.0 for act in self.actions}
        return self.q_table[key]

    def choose_action(self, state: Any) -> str:
        """Select action via epsilon-greedy strategy."""
        if not self.dontlearn and random.random() < self.epsilon:
            return random.choice(self.actions)

        q_dict = self.get_q_values(state)
        max_q = max(q_dict.values())
        if self.dontlearn and max_q < 2.0 and random.random() < 0.15:
            safe_actions = [
                act for act, val in q_dict.items() if val > -2.0
            ]
            if safe_actions:
                return random.choice(safe_actions)
        # Break ties randomly to avoid directional bias
        best_actions = [act for act, val in q_dict.items() if val == max_q]
        return random.choice(best_actions)

    def update(
        self,
        state: Any,
        action: str,
        reward: float,
        next_state: Any,
        done: bool,
    ) -> None:
        """Update Q-table entry using Bellman equation."""
        if self.dontlearn:
            return

        key = str(state)
        curr_q = self.get_q_values(state)[action]
        if done:  # 死亡時沒有未來，target = r
            target = reward
        else:
            next_q_vals = self.get_q_values(next_state).values()
            target = reward + self.gamma * max(next_q_vals)

        self.q_table[key][action] = curr_q + self.alpha * (target - curr_q)

    def save(self, filepath: str) -> None:
        """Save Q-table to text file in subject specified format."""
        dir_name = os.path.dirname(filepath)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as file:
            json.dump(self.q_table, file, indent=2)
        print(f"Save learning state in {filepath}")

    def load(self, filepath: str) -> bool:
        """Load Q-table from text file in subject specified format."""
        if not os.path.exists(filepath):
            print(f"Warning: Model file {filepath} not found.")
            return False
        with open(filepath, "r", encoding="utf-8") as file:
            data = json.load(file)
            normalized = {}
            for k, v in data.items():
                if isinstance(v, list):
                    normalized[k] = {
                        act: float(v[idx]) if idx < len(v) else 0.0
                        for idx, act in enumerate(self.actions)
                    }
                elif isinstance(v, dict):
                    normalized[k] = {
                        str(act).upper(): float(val) for act, val in v.items()
                    }
                else:
                    normalized[k] = v
            self.q_table = normalized
        print(f"Load trained model from {filepath}")
        return True
