"""Episode-scoped static context and bounded observation/decision history."""
from collections import deque
from copy import deepcopy
import json


class EpisodeMemory:
    def __init__(self, history_size: int = 5):
        if history_size <= 0:
            raise ValueError("history_size must be positive")
        self.static_context = {}
        self.history = deque(maxlen=history_size)

    def reset(self, static_context: dict) -> None:
        """Replace the static cache and discard observations from older episodes."""
        self.static_context = deepcopy(static_context)
        self.history.clear()

    def render_static_context(self) -> str:
        context = {
            "movements": {movement: {"direction": info["direction"], "lanes": info["number_of_lanes"],
                                     "lane_ids": info.get("lane_ids", [])}
                          for movement, info in self.static_context["layout"].items()},
            "phases": self.static_context["phases"],
            "available_actions": self.static_context["available_actions"],
            "control_constraints": self.static_context["control_constraints"],
        }
        return json.dumps(context, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    def remember(self, snapshot: dict, phase_id: int) -> None:
        self.history.append({
            "sim_time": snapshot["sim_time"],
            "phase_id": phase_id,
            "occupancy": deepcopy(snapshot["occupancy"]),
            "situation": deepcopy(snapshot["situation"]),
        })

    def recent(self, limit: int = 3) -> list[dict]:
        if not 1 <= limit <= self.history.maxlen:
            raise ValueError(f"limit must be between 1 and {self.history.maxlen}")
        return deepcopy(list(self.history)[-limit:])
