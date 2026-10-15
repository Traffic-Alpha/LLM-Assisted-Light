'''
@Author: WANG Maonan
@Date: 2023-09-04 20:43:53
@Description: 信号灯控制环境
@LastEditTime: 2026-06-11 23:58:34
'''
import json

from pathlib import Path
from typing import Any, Callable, Dict
from tshub.tshub_env.tshub_env import TshubEnvironment

class TrafficSignalEnv:
    def __init__(self,
                 sumo_cfg:str,
                 net_file:str,
                 trip_info: str,
                 num_seconds:int | None,
                 tls_id:str,
                 tls_action_type:str,
                 use_gui:bool=False,
                 delta_time:int=5,
        ) -> None:
        self.tls_id = tls_id
        self.num_seconds = num_seconds
        self.trip_info = trip_info
        self.pending_events_callback: Callable[[], bool] = lambda: False
        self.collision_events: list[dict[str, Any]] = []
        self.teleport_events: list[dict[str, Any]] = []
        self.episode_status: Dict[str, Any] = {}

        self.tsc_env = TshubEnvironment(
            sumo_cfg=sumo_cfg,
            net_file=net_file,
            is_map_builder_initialized=True, # For scenario render
            is_aircraft_builder_initialized=False,
            is_vehicle_builder_initialized=True, # calculate waiting time for vehicle
            is_traffic_light_builder_initialized=True, # control traffic light
            is_person_builder_initialized=False,
            trip_info=trip_info, # 记录仿真中所有 object 的指标
            tls_ids=[tls_id],
            num_seconds=num_seconds if num_seconds is not None else float('inf'),
            time_to_teleport=-1,
            collision_action="warn",
            tls_action_type=tls_action_type,
            delta_time=delta_time,
            use_gui=use_gui,
            is_libsumo=(not use_gui),
        )

    @property
    def unwrapped(self) -> "TrafficSignalEnv":
        return self

    def reset(
            self,
            *,
            seed: int | None = None,
            options: Dict[str, Any] | None = None,
        ) -> Any:
        self.collision_events.clear()
        self.teleport_events.clear()
        self.episode_status = {}
        state_infos = self.tsc_env.reset()
        return state_infos

    def set_pending_events_callback(self, callback: Callable[[], bool]) -> None:
        self.pending_events_callback = callback

    def step(self, action: Dict[str, int]) -> tuple[Dict[str, Any], bool, bool, Dict[str, Any]]:
        tsc_action = {'tls': action}
        states, _, infos, _ = self.tsc_env.step(tsc_action)
        simulation = self.tsc_env.sumo.simulation
        sim_time = infos['step_time']
        remaining = simulation.getMinExpectedNumber()
        pending_events = self.pending_events_callback()
        terminated = remaining == 0 and not pending_events
        truncated = not terminated and self.num_seconds is not None and sim_time >= self.num_seconds
        reason = "vehicles_cleared" if terminated else "time_limit" if truncated else None
        colliding = list(simulation.getCollidingVehiclesIDList())
        if colliding:
            self.collision_events.append({"sim_time": sim_time, "vehicle_ids": colliding})
        teleporting = list(simulation.getStartingTeleportIDList())
        if teleporting:
            self.teleport_events.append({"sim_time": sim_time, "vehicle_ids": teleporting})
        self.episode_status = {
            "final_sim_time": sim_time, "termination_reason": reason,
            "remaining_vehicles": remaining, "pending_events": pending_events,
            "collision_steps": len(self.collision_events),
            "teleport_steps": len(self.teleport_events),
        }
        infos.update(self.episode_status)

        return states, terminated, truncated, infos

    def close(self) -> None:
        if self.tsc_env.sumo is not None:
            path = Path(self.trip_info).with_suffix(".simulation.json")
            path.write_text(json.dumps({
                **self.episode_status,
                "time_limit_seconds": self.num_seconds,
                "time_to_teleport": -1, "collision_action": "warn",
                "collision_events": self.collision_events,
                "teleport_events": self.teleport_events,
            }, ensure_ascii=False, indent=2) + "\n")
        self.tsc_env._close_simulation()
