"""Use the same occupancy recommendation exposed to the LLM agent."""
from common.runtime import parse_phase_id


class MaxPressureController:
    def __init__(self, phase_num: int):
        self.phase_num = phase_num
        self.phase_id = 0

    def act(self, env) -> int:
        phase_id = parse_phase_id(env.get_traditional_decision().get("recommended_phase"))
        if phase_id is not None:
            self.phase_id = phase_id % self.phase_num
        return self.phase_id
