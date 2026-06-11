"""Traffic environment exports.

The package uses lazy exports so event configuration can be imported without
pulling in SUMO, Gymnasium, or TSHub.
"""

from importlib import import_module
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from traffic_env.event_config import (
        AccidentEvent,
        EventConfig,
        EventConfigManager,
        SensorFailureEvent,
        SpecialVehicleEvent,
    )
    from traffic_env.event_wrapper import EnvironmentEventWrapper, create_event_wrapper
    from traffic_env.llm_wrapper import LLMTSCEnvWrapper
    from traffic_env.tsc_env import TrafficSignalEnv


_EXPORTS = {
    "AccidentEvent": ("traffic_env.event_config", "AccidentEvent"),
    "EventConfig": ("traffic_env.event_config", "EventConfig"),
    "EventConfigManager": ("traffic_env.event_config", "EventConfigManager"),
    "SensorFailureEvent": ("traffic_env.event_config", "SensorFailureEvent"),
    "SpecialVehicleEvent": ("traffic_env.event_config", "SpecialVehicleEvent"),
    "EnvironmentEventWrapper": ("traffic_env.event_wrapper", "EnvironmentEventWrapper"),
    "create_event_wrapper": ("traffic_env.event_wrapper", "create_event_wrapper"),
    "LLMTSCEnvWrapper": ("traffic_env.llm_wrapper", "LLMTSCEnvWrapper"),
    "TrafficSignalEnv": ("traffic_env.tsc_env", "TrafficSignalEnv"),
}


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module 'traffic_env' has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


__all__ = sorted(_EXPORTS)
