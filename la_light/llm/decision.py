"""Parse model output and obtain an MP fallback for invalid decisions."""
import json
import re
from typing import Dict

from loguru import logger

from la_light.runtime import parse_phase_id


def parse_agent_response(response: str) -> Dict:
    response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
    data = {}
    try:
        data = json.loads(response)
    except json.JSONDecodeError:
        for match in reversed(list(re.finditer(r"\{", response))):
            try:
                candidate, _ = json.JSONDecoder().raw_decode(response[match.start():])
                if isinstance(candidate, dict) and "decision" in candidate:
                    data = candidate
                    break
            except json.JSONDecodeError:
                continue

    if not isinstance(data, dict):
        data = {}

    decision = data.get("decision")
    phase_id = None
    if isinstance(decision, int) and not isinstance(decision, bool) and decision >= 0:
        phase_id = decision
    elif isinstance(decision, str):
        match = re.fullmatch(r"(?:Phase[- ]?)?(\d+)", decision.strip())
        if match:
            phase_id = int(match.group(1))

    return {
        "phase_id": phase_id,
        "explanation": data.get("explanation", response),
        "has_valid_decision": phase_id is not None,
    }


def get_traditional_fallback_phase(env, fallback_phase: int, phase_num: int) -> int:
    try:
        decision = env.get_traditional_decision()
    except Exception as e:
        logger.warning(
            "Failed to get traditional fallback decision; keeping phase {}: {}",
            fallback_phase,
            e,
        )
        return fallback_phase % phase_num

    phase_id = parse_phase_id(decision.get("recommended_phase"))
    if phase_id is None:
        logger.warning(
            "Traditional fallback returned invalid phase {}; keeping phase {}",
            decision.get("recommended_phase"),
            fallback_phase,
        )
        return fallback_phase % phase_num

    return phase_id % phase_num
