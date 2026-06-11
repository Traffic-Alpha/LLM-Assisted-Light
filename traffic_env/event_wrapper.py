'''
@Author: WANG Maonan
@Date: 2026-06-12 00:01:05
@Description: Dynamic event wrapper for SUMO-based TSC environments.
@LastEditTime: 2026-06-12 00:35:33
'''
from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Set, cast

import gymnasium as gym
from traffic_env.logger import logger

from traffic_env.event_config import (
    AccidentEvent,
    EventConfig,
    EventConfigManager,
    SensorFailureEvent,
    SpecialVehicleEvent,
)


class EnvironmentEventWrapper(gym.Wrapper):
    """Apply configured dynamic events to a 2D TSC environment."""

    def __init__(
        self,
        env: gym.Env,
        event_config: Optional[EventConfig] = None,
        enable_accidents: bool = True,
        enable_special_vehicles: bool = True,
        enable_sensor_failures: bool = True,
    ) -> None:
        super().__init__(env)
        self.event_config = event_config or EventConfig()
        self.enable_accidents = enable_accidents
        self.enable_special_vehicles = enable_special_vehicles
        self.enable_sensor_failures = enable_sensor_failures

        self.conn: Any = None
        self.sim_time = 0.0
        self.pending_accidents: Dict[str, Dict[str, Any]] = {}
        self.inserted_accidents: Set[str] = set()
        self.scheduled_special_vehicles: Set[str] = set()
        self.active_sensor_failures: Set[str] = set()

        if self.event_config.has_events():
            logger.info(f"Event wrapper enabled: {self.event_config.summary()}")

    # ------------------------------------------------------------------
    # Gym lifecycle
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: Dict[str, Any] | None = None,
    ) -> Any:
        state = cast(Any, self.env).reset(seed=seed, options=options)
        self.conn = self._find_sumo_connection()
        self.sim_time = self._get_sumo_time(default=0.0)
        self._clear_runtime_state()
        self._prepare_events()
        return state

    def step(self, action: Any) -> Any:
        self.sim_time = self._get_sumo_time(default=self.sim_time)
        self._apply_events(self.sim_time)

        result = cast(Any, self.env).step(action)
        info = self._extract_info(result)
        if "step_time" in info:
            self.sim_time = float(info["step_time"])
        else:
            self.sim_time = self._get_sumo_time(default=self.sim_time)
        self._apply_events(self.sim_time)
        return result

    def close(self) -> None:
        cast(Any, self.env).close()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)

    # ------------------------------------------------------------------
    # Event lifecycle
    # ------------------------------------------------------------------

    def _clear_runtime_state(self) -> None:
        """每次 reset 的时候清空 events 信息
        """
        self.pending_accidents.clear()
        self.inserted_accidents.clear()
        self.scheduled_special_vehicles.clear()
        self.active_sensor_failures.clear()

    def _prepare_events(self) -> None:
        if self.conn is None:
            logger.warning("SUMO connection is unavailable; event injection is disabled")
            return

        if self.enable_accidents:
            for accident in self.event_config.accidents:
                self._prepare_accident(accident)

        if self.enable_special_vehicles:
            for vehicle in self.event_config.special_vehicles:
                self._schedule_special_vehicle(vehicle)

    def _apply_events(self, sim_time: float) -> None:
        if not self.event_config.has_events() or self.conn is None:
            return

        if self.enable_accidents:
            for accident in self.event_config.accidents:
                if sim_time >= accident.depart_time:
                    self._insert_accident(accident)

        if self.enable_sensor_failures:
            self._update_sensor_failures(sim_time)

    # ------------------------------------------------------------------
    # Accident / obstacle vehicles
    # ------------------------------------------------------------------

    def _prepare_accident(self, accident: AccidentEvent) -> None:
        """Register route and metadata for an accident obstacle."""
        route_id = f"route_{accident.id}"
        self._add_route(route_id, [accident.edge_id])
        veh_type = self._ensure_vehicle_type(accident.type, fallback="background")
        self.pending_accidents[accident.id] = {
            "route_id": route_id,
            "edge_id": accident.edge_id,
            "lane_index": accident.lane_index,
            "position": accident.position,
            "duration": accident.duration,
            "depart_time": accident.depart_time,
            "veh_type": veh_type,
        }
        logger.info(
            "Accident vehicle {} will appear at {:.1f}s on {}_{} @ {:.1f}m",
            accident.id,
            accident.depart_time,
            accident.edge_id,
            accident.lane_index,
            accident.position,
        )

    def _insert_accident(self, accident: AccidentEvent) -> None:
        """Insert one stopped obstacle vehicle into SUMO."""
        if accident.id in self.inserted_accidents:
            return

        veh_info = self.pending_accidents.get(accident.id)
        if not veh_info:
            return

        lane_id = f"{veh_info['edge_id']}_{veh_info['lane_index']}"
        self._validate_vehicle_lane(lane_id, str(veh_info["veh_type"]))
        self._remove_conflicting_vehicles(lane_id, float(veh_info["position"]))

        self.conn.vehicle.add(
            vehID=accident.id,
            routeID=veh_info["route_id"],
            typeID=veh_info["veh_type"],
            depart="now",
            departLane=int(veh_info["lane_index"]),
            departPos=float(veh_info["position"]),
            departSpeed=0,
        )
        self.conn.vehicle.moveTo(
            vehID=accident.id,
            laneID=lane_id,
            pos=float(veh_info["position"]),
        )
        self.conn.vehicle.setStop(
            vehID=accident.id,
            edgeID=veh_info["edge_id"],
            pos=float(veh_info["position"]),
            laneIndex=int(veh_info["lane_index"]),
            duration=float(veh_info["duration"]),
        )
        self.conn.vehicle.setSpeed(accident.id, 0)
        self.inserted_accidents.add(accident.id)
        logger.info("Inserted accident obstacle {} at {:.1f}s", accident.id, self.sim_time)

    # ------------------------------------------------------------------
    # Special vehicles
    # ------------------------------------------------------------------

    def _schedule_special_vehicle(self, vehicle: SpecialVehicleEvent) -> None:
        """Schedule an emergency or priority vehicle on its configured route."""
        if vehicle.id in self.scheduled_special_vehicles:
            return

        route_id = f"route_{vehicle.id}"
        self._add_route(route_id, vehicle.route)
        vehicle_type = self._ensure_vehicle_type(vehicle.vehicle_type, fallback="rescue")
        self.conn.vehicle.add(
            vehID=vehicle.id,
            routeID=route_id,
            typeID=vehicle_type,
            depart=str(vehicle.depart_time),
            departLane=str(vehicle.depart_lane),
            departPos=float(vehicle.depart_pos),
            departSpeed=float(vehicle.depart_speed),
        )
        self.scheduled_special_vehicles.add(vehicle.id)
        logger.info(
            "Scheduled special vehicle {} ({}) at {:.1f}s on {}",
            vehicle.id,
            vehicle_type,
            vehicle.depart_time,
            " ".join(vehicle.route),
        )

    # ------------------------------------------------------------------
    # Sensor failures
    # ------------------------------------------------------------------

    def _update_sensor_failures(self, sim_time: float) -> None:
        """Apply or clear detector masks based on active failure events."""
        active = [
            failure
            for failure in self.event_config.sensor_failures
            if failure.is_active(sim_time)
        ]
        active_ids = {failure.id for failure in active}
        had_active_failures = bool(self.active_sensor_failures)

        for failure in active:
            if failure.id not in self.active_sensor_failures:
                self._set_detector_failure(failure)

        for failure_id in list(self.active_sensor_failures - active_ids):
            self.active_sensor_failures.discard(failure_id)

        if active:
            self._call_env_method("set_occ_missing", not_work_element=active[-1].detector_id)
        elif had_active_failures:
            self._call_env_method("set_occ_missing", not_work_element="")

        if not active and had_active_failures:
            self.active_sensor_failures.clear()

    def _set_detector_failure(self, failure: SensorFailureEvent) -> None:
        self._call_env_method("set_occ_missing", not_work_element=failure.detector_id)
        self.active_sensor_failures.add(failure.id)
        logger.info("Applied sensor failure {} ({})", failure.id, failure.detector_id)

    # ------------------------------------------------------------------
    # SUMO / TraCI helpers
    # ------------------------------------------------------------------

    def _add_route(self, route_id: str, edges: Iterable[str]) -> None:
        try:
            if route_id not in set(self.conn.route.getIDList()):
                self.conn.route.add(route_id, list(edges))
        except Exception as exc:
            logger.debug(f"Could not add route {route_id}: {exc}")

    def _ensure_vehicle_type(self, type_id: str, fallback: str = "background") -> str:
        type_domain = getattr(self.conn, "vehicletype", None)
        if type_domain is None:
            return type_id

        existing_types = set(type_domain.getIDList())
        if type_id in existing_types:
            return type_id

        base_type = fallback if fallback in existing_types else "DEFAULT_VEHTYPE"
        try:
            type_domain.copy(base_type, type_id)
            if "rescue" in type_id or "ambulance" in type_id or "police" in type_id:
                type_domain.setColor(type_id, (255, 40, 40, 255))
                type_domain.setMaxSpeed(type_id, 20.0)
            elif "barrier" in type_id or "parked" in type_id:
                type_domain.setColor(type_id, (80, 80, 80, 255))
                type_domain.setMaxSpeed(type_id, 0.0)
            return type_id
        except Exception as exc:
            logger.debug(f"Could not create vehicle type {type_id}: {exc}")
            return fallback if fallback in existing_types else type_id

    def _validate_vehicle_lane(self, lane_id: str, type_id: str) -> None:
        """Fail early when an obstacle is configured on a pedestrian-only lane."""
        try:
            self.conn.lane.getLength(lane_id)
            allowed = set(self.conn.lane.getAllowed(lane_id))
            disallowed = set(self.conn.lane.getDisallowed(lane_id))
            vehicle_class = self.conn.vehicletype.getVehicleClass(type_id)
            if allowed and vehicle_class not in allowed:
                raise ValueError(
                    f"Vehicle type {type_id!r} ({vehicle_class}) cannot depart on lane {lane_id}; "
                    f"allowed classes are {sorted(allowed)}"
                )
            if vehicle_class in disallowed:
                raise ValueError(
                    f"Vehicle type {type_id!r} ({vehicle_class}) cannot depart on lane {lane_id}; "
                    f"disallowed classes are {sorted(disallowed)}"
                )
        except AttributeError:
            return

    def _remove_conflicting_vehicles(self, lane_id: str, target_pos: float) -> None:
        try:
            for veh_id in self.conn.lane.getLastStepVehicleIDs(lane_id):
                pos = float(self.conn.vehicle.getLanePosition(veh_id))
                if abs(target_pos - pos) < 1.0:
                    self.conn.vehicle.remove(veh_id)
                    logger.info("Removed vehicle {} before inserting obstacle", veh_id)
        except Exception as exc:
            logger.debug(f"Could not clear lane {lane_id}: {exc}")

    def _call_env_method(self, method_name: str, **kwargs: Any) -> bool:
        target: Any = self.env
        seen: Set[int] = set()
        while target is not None and id(target) not in seen:
            seen.add(id(target))
            method = getattr(target, method_name, None)
            if callable(method):
                method(**kwargs)
                return True
            target = getattr(target, "env", None)
        return False

    def _find_sumo_connection(self) -> Any | None:
        target: Any = self.env
        seen: Set[int] = set()
        while target is not None and id(target) not in seen:
            seen.add(id(target))
            tsc_env = getattr(target, "tsc_env", None)
            if tsc_env is not None:
                if hasattr(tsc_env, "sumo"):
                    return tsc_env.sumo
                nested = getattr(tsc_env, "tshub_env", None)
                if nested is not None and hasattr(nested, "sumo"):
                    return nested.sumo
            target = getattr(target, "env", None)
        return None

    def _get_sumo_time(self, default: float) -> float:
        if self.conn is None:
            return default
        try:
            return float(self.conn.simulation.getTime())
        except Exception:
            return default

    # ------------------------------------------------------------------
    # Reporting / LLM-facing helpers
    # ------------------------------------------------------------------

    def _extract_info(self, step_result: Any) -> Dict[str, Any]:
        if isinstance(step_result, tuple) and len(step_result) >= 5:
            return step_result[4] or {}
        if isinstance(step_result, tuple) and len(step_result) >= 3:
            return step_result[-1] or {}
        return {}

    def get_active_events(self) -> Dict[str, Any]:
        active = self.event_config.active_events(self.sim_time)
        return {
            "time": self.sim_time,
            "accidents": [event.id for event in active["accidents"]],
            "inserted_accidents": sorted(self.inserted_accidents),
            "special_vehicles": [event.id for event in active["special_vehicles"]],
            "scheduled_special_vehicles": sorted(self.scheduled_special_vehicles),
            "sensor_failures": sorted(self.active_sensor_failures),
        }

    def get_junction_situation(self) -> Dict[str, Any]:
        base_situation: Dict[str, Any] = {}
        base_method = getattr(self.env, "get_junction_situation", None)
        if callable(base_method):
            base_situation = cast(Dict[str, Any], base_method())
        return {
            **base_situation,
            "configured_events": self.event_config.summary(),
            "active_events": self.get_active_events(),
        }


# ----------------------------------------------------------------------
# Factory
# ----------------------------------------------------------------------

def create_event_wrapper(
    env: gym.Env,
    env_name: str,
    event_config_name: str,
    enable_accidents: bool = True,
    enable_special_vehicles: bool = True,
    enable_sensor_failures: bool = True,
) -> EnvironmentEventWrapper:
    event_config = EventConfigManager.get_config_for_scenario(env_name, event_config_name)
    return EnvironmentEventWrapper(
        env=env,
        event_config=event_config,
        enable_accidents=enable_accidents,
        enable_special_vehicles=enable_special_vehicles,
        enable_sensor_failures=enable_sensor_failures,
    )
