'''
@Author: WANG Maonan
@Date: 2026-06-11 17:58:54
@Description: Simple max-pressure style controller for single-intersection TSC.
@LastEditTime: 2026-06-12 00:11:52
'''
from __future__ import annotations

from typing import Dict, Mapping


def choose_maxpressure_phase(phase_occ: Mapping[int, float] | Mapping[str, float]) -> int:
    """Choose the phase with the largest observed movement pressure.

    The current lightweight environment exposes ``info["phase_occ"]`` as the
    summed occupancy for each phase.  For the small single-junction scenarios in
    this repository, selecting the largest phase occupancy is the intended
    max-pressure baseline.
    """
    if not phase_occ:
        return 0
    phase_id, _ = max(phase_occ.items(), key=lambda item: float(item[1]))
    return int(str(phase_id).split("-")[-1])


class MaxPressureController:
    def __init__(self, phase_num: int) -> None:
        self.phase_num = phase_num

    def act(self, info: Dict | None = None, fallback_phase: int = 0) -> int:
        if not info or "phase_occ" not in info:
            return fallback_phase % self.phase_num
        return choose_maxpressure_phase(info["phase_occ"]) % self.phase_num
