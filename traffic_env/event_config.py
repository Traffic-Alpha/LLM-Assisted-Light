'''
@Author: WANG Maonan
@Date: 2026-06-12 00:01:05
@Description: Configuration objects for dynamic traffic events.
包含 3 个事件类型：事故、特殊车辆、传感器故障。每个事件都有一个开始时间和持续时间，可以查询在任意仿真时间点哪些事件处于活跃状态。
@LastEditTime: 2026-06-12 00:26:45
'''
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml
from traffic_env.logger import logger


def _as_list(value: Any) -> List[Dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"Event section must be a list, got {type(value).__name__}")
    return value


@dataclass(frozen=True)
class TimedEvent:
    id: str
    depart_time: float
    duration: float = 0

    @property
    def end_time(self) -> float:
        return self.depart_time + self.duration

    def is_active(self, sim_time: float) -> bool:
        if self.duration <= 0:
            return sim_time >= self.depart_time
        return self.depart_time <= sim_time < self.end_time


@dataclass(frozen=True)
class AccidentEvent(TimedEvent):
    edge_id: str = ""
    lane_index: int = 0
    position: float = 0
    type: str = "parked_car"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AccidentEvent":
        return cls(
            id=str(data["id"]),
            depart_time=float(data["depart_time"]),
            edge_id=str(data["edge_id"]),
            lane_index=int(data.get("lane_index", 0)),
            position=float(data["position"]),
            type=str(data.get("type", "parked_car")),
            duration=float(data.get("duration", 60)),
        )


@dataclass(frozen=True)
class SpecialVehicleEvent(TimedEvent):
    route: List[str] = field(default_factory=list)
    vehicle_type: str = "rescue"
    priority: int = 1
    depart_lane: str = "best"
    depart_pos: float = 0
    depart_speed: float = 0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SpecialVehicleEvent":
        vehicle_type = data.get("vehicle_type", data.get("type", "rescue"))
        route = data.get("route", [])
        if not route:
            raise ValueError(f"Special vehicle {data.get('id')} must define route")
        return cls(
            id=str(data["id"]),
            depart_time=float(data["depart_time"]),
            route=[str(edge) for edge in route],
            vehicle_type=str(vehicle_type),
            priority=int(data.get("priority", 1)),
            depart_lane=str(data.get("depart_lane", "best")),
            depart_pos=float(data.get("depart_pos", 0)),
            depart_speed=float(data.get("depart_speed", 0)),
            duration=float(data.get("duration", 0)),
        )


@dataclass(frozen=True)
class SensorFailureEvent(TimedEvent):
    detector_id: str = ""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SensorFailureEvent":
        return cls(
            id=str(data["id"]),
            depart_time=float(data["depart_time"]),
            detector_id=str(data["detector_id"]),
            duration=float(data.get("duration", 60)),
        )


class EventConfig:
    """Parsed event configuration for one scenario run."""

    def __init__(self, config_file: Optional[str | Path] = None):
        self.config_file = Path(config_file) if config_file else None
        self.accidents: List[AccidentEvent] = []
        self.special_vehicles: List[SpecialVehicleEvent] = []
        self.sensor_failures: List[SensorFailureEvent] = []

        if self.config_file:
            self.load_from_file(self.config_file)

    def load_from_file(self, config_file: str | Path) -> None:
        config_path = Path(config_file)
        if not config_path.exists():
            raise FileNotFoundError(f"Event config not found: {config_path}")

        with config_path.open("r", encoding="utf-8") as file:
            data = yaml.safe_load(file) or {}

        try:
            self.accidents = [
                AccidentEvent.from_dict(item)
                for item in _as_list(data.get("ACCIDENTS"))
            ]
            self.special_vehicles = [
                SpecialVehicleEvent.from_dict(item)
                for item in _as_list(data.get("SPECIAL_VEHICLES"))
            ]
            self.sensor_failures = [
                SensorFailureEvent.from_dict(item)
                for item in _as_list(data.get("SENSOR_FAILURES"))
            ]
        except KeyError as exc:
            raise ValueError(f"Missing required event field: {exc}") from exc

        logger.info(f"Loaded event config: {config_path}")
        logger.info(self.summary())

    def has_events(self) -> bool:
        return any(
            (
                self.accidents,
                self.special_vehicles,
                self.sensor_failures,
            )
        )

    def active_events(self, sim_time: float) -> Dict[str, List[TimedEvent]]:
        return {
            "accidents": [event for event in self.accidents if event.is_active(sim_time)],
            "special_vehicles": [
                event for event in self.special_vehicles if event.is_active(sim_time)
            ],
            "sensor_failures": [
                event for event in self.sensor_failures if event.is_active(sim_time)
            ],
        }

    def all_events(self) -> Iterable[TimedEvent]:
        yield from self.accidents
        yield from self.special_vehicles
        yield from self.sensor_failures

    def summary(self) -> str:
        return (
            "Event configuration: "
            f"{len(self.accidents)} accidents, "
            f"{len(self.special_vehicles)} special vehicles, "
            f"{len(self.sensor_failures)} sensor failures"
        )

    get_summary = summary
    get_active_events = active_events


class EventConfigManager:
    CONFIG_DIR = Path(__file__).resolve().parent.parent / "scenarios"

    @classmethod
    def resolve_config_path(cls, env_name: str, config_name: str = "default") -> Path:
        return Path(config_name).expanduser()

    @classmethod
    def get_config_for_scenario(
        cls, env_name: str, config_name: str = "default"
    ) -> EventConfig:
        config_path = cls.resolve_config_path(env_name, config_name)
        if not config_path.exists():
            raise FileNotFoundError(f"Event config not found: {config_path}")
        return EventConfig(config_path)

    @classmethod
    def list_available_configs(cls, env_name: str) -> List[str]:
        events_dir = cls.CONFIG_DIR / env_name / "events"
        if not events_dir.exists():
            return []
        return sorted(path.stem for path in events_dir.glob("*.y*ml"))
