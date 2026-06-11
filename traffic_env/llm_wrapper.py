'''
@Author: WANG Maonan
@Date: 2026-06-11 17:58:44
@Description: Lightweight LLM-facing environment wrapper.

The old implementation used LangChain prompt/output parser classes.  The
refactored version exposes plain Python data and a plain-text description so it
can be used by any agent implementation.
@LastEditTime: 2026-06-11 23:59:00
'''
from gymnasium.core import Env
from typing import Any, Dict, SupportsFloat, Tuple

from traffic_env.base_tsc_wrapper import BaseTSCEnvWrapper


class LLMTSCEnvWrapper(BaseTSCEnvWrapper):
    def __init__(
        self,
        env: Env,
        tls_id: str,
        phase_num: int,
        max_states: int = 5,
    ) -> None:
        super().__init__(env, tls_id, phase_num, max_states)

    def description_env(self) -> str:
        return "\n".join(
            [
                "Traffic signal control state:",
                f"Current phase: {self.get_current_phase()}",
                f"Intersection layout: {self.get_intersection_layout()}",
                f"Signal phases: {self.get_signal_phase_structure()}",
                f"Current occupancy: {self.get_current_occupancy()}",
                f"Junction situation: {self.get_junction_situation()}",
                f"Available actions: {self.get_available_actions()}",
                'Return JSON like {"decision": 0, "explanation": "..."}',
            ]
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: Dict[str, Any] | None = None,
    ) -> Tuple[Any, Dict[str, Any]]:
        return super().reset(seed=seed, options=options)

    def step(self, action: int) -> Tuple[Any, SupportsFloat, bool, bool, Dict[str, Any]]:
        return super().step(action)
