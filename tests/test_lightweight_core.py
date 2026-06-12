"""Smoke check for the lightweight tool registry."""

from llm_tsc.tools_registry import ToolRegistry, tool
from traffic_env.event_config import EventConfig, EventConfigManager
from llm_tsc.maxpressure import choose_maxpressure_phase
import pytest


class DemoTools:
    @tool(name="demo_tool", description="demo")
    def demo(self, value: str = "ok") -> str:
        return value


def test_tool_registry_discovers_decorated_methods():
    registry = ToolRegistry()
    tools = registry.register_from_object(DemoTools())
    assert "demo_tool" in tools
    assert tools["demo_tool"](value="ready") == "ready"


def test_event_config_loads_scenario_yaml():
    config = EventConfig("scenarios/4way/events/accident_set1.yaml")
    assert len(config.accidents) == 2
    assert len(config.special_vehicles) == 1
    assert config.special_vehicles[0].route == ["E0", "-E2"]
    assert config.special_vehicles[0].depart_lane == "best"


def test_event_config_requires_existing_path():
    with pytest.raises(FileNotFoundError):
        EventConfigManager.get_config_for_scenario("4way", "accident_set1.yaml")


def test_maxpressure_selects_largest_phase_pressure():
    assert choose_maxpressure_phase({0: 0.1, 1: 0.8, 2: 0.2}) == 1
