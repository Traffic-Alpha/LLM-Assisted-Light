"""Single-intersection observations and decision-step control shared by all methods."""
import numpy as np

from typing import Any, Callable, Dict, List, TypeAlias, cast
from numpy.typing import NDArray

from la_light.env.tsc_env import TrafficSignalEnv
from la_light.env.wrapper_utils import convert_state_to_static_information

Float32Array: TypeAlias = NDArray[np.float32]
EnvStepResult: TypeAlias = tuple[Dict[str, Any], bool, bool, Dict[str, Any]]

class OccupancyList:
    def __init__(self) -> None:
        self.elements: List[List[float]] = []

    def add_element(self, element) -> None:
        if isinstance(element, list):
            if all(isinstance(e, float) for e in element):
                self.elements.append(element)
            else:
                raise ValueError("列表中的元素必须是浮点数类型")
        else:
            raise TypeError("添加的元素必须是列表类型")

    def clear_elements(self) -> None:
        self.elements = []

    def calculate_average(self) -> Float32Array:
        """计算一段时间的平均 occupancy
        """
        arr: Float32Array = np.asarray(self.elements, dtype=np.float32)
        averages = cast(Float32Array, np.mean(arr, axis=0, dtype=np.float32))
        averages = cast(Float32Array, averages / np.float32(100))
        self.clear_elements() # 清空列表
        return averages


class BaseTSCEnvWrapper:
    """TSC Env Wrapper for single junction with tls_id."""
    def __init__(
            self, env: TrafficSignalEnv,
            tls_id:str, 
            phase_num:int, 
        ) -> None:
        self.env = env
        self.tls_id = tls_id # 单路口的 id
        self.phase_num = phase_num
        self.movement_ids: List[str] = []
        self.phase2movements: Dict[Any, List[str]] = {}
        self.occupancy = OccupancyList()
        self.current_tls_state: Dict[str, Any] = {}
        self.llm_static_information: Dict[str, Dict[str, Any]] = {
            "movement_infos": {},
            "phase_infos": {},
        }
        self.current_avg_occ: Float32Array = cast(Float32Array, np.zeros(0, dtype=np.float32))
        self.previous_avg_occ: Float32Array | None = None
        self.ls_elements: List[str] = []
        self.mask: List[str] = []
        self.not_work_element: str | None = None
        self.simulation_step_callback: Callable[[], None] | None = None

    @property
    def unwrapped(self) -> TrafficSignalEnv:
        return self.env

    def info_wrapper(self, infos: Dict[str, Any], occupancy: Float32Array) -> Dict[str, Any]:
        """在 info 中加入一些信息:
        1. 当前的仿真时间
        2. 每个 phase 的占有率
        3. 当前的 phase index
        """
        movement_occ = {key: float(value) for key, value in zip(self.movement_ids, occupancy)}
        if self.not_work_element is not None and self.not_work_element in movement_occ:
            movement_occ[self.not_work_element] = -1 # 如果存在无法工作的传感器

        # 计算每个 phase 的占有率
        phase_occ: Dict[Any, float] = {}
        for phase_index, phase_movements in self.phase2movements.items():
            phase_occ[phase_index] = sum([movement_occ[phase] for phase in phase_movements])
        
        infos['phase_occ'] = phase_occ
        return infos


    # ##############
    # Reset & Step
    # ##############
    def reset(
            self,
            *,
            seed: int | None = None,
            options: Dict[str, Any] | None = None,
        ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """reset 时初始化 (1) 静态信息; (2) 动态信息
        """
        # 环境的 reset
        state = self.env.reset(seed=seed, options=options)
        self.current_tls_state = state['tls'][self.tls_id].copy() # 当前的状态
        
        # 初始化路口静态信息
        self.llm_static_information = convert_state_to_static_information(state['tls'][self.tls_id]) # 路口的静态信息
        self.movement_ids = state['tls'][self.tls_id]['movement_ids']
        self.phase2movements = state['tls'][self.tls_id]['phase2movements']
        self._cache_signal_phase_information()
        self.current_avg_occ = np.zeros(len(self.movement_ids), dtype=np.float32)
        self.previous_avg_occ = None
        self.occupancy.clear_elements()

        # For Detector State
        self.ls_elements = [element for element in self.movement_ids if element.endswith('--l') or element.endswith('--s')]
        self.mask = ['Work'] * len(self.ls_elements)
        self.not_work_element = None # 初始化时候没有传感器是损坏的

        return self.current_tls_state.copy(), {'step_time': 0}

    def _step_env(self, tsc_action: Dict[str, int]) -> EnvStepResult:
        if self.simulation_step_callback is not None:
            self.simulation_step_callback()
        result = self.env.step(tsc_action)
        if self.simulation_step_callback is not None:
            self.simulation_step_callback()
        return result

    def set_simulation_step_callback(self, callback: Callable[[], None]) -> None:
        """Run event updates at each SUMO step, including between decisions."""
        self.simulation_step_callback = callback

    def step(self, action: int) -> EnvStepResult:
        """Advance SUMO to the next decision point and average interval occupancy."""
        tsc_action = {self.tls_id: action}
        while True:
            states, terminated, truncated, infos = self._step_env(tsc_action)
            self.current_tls_state = states['tls'][self.tls_id].copy()
            self.occupancy.add_element(self.current_tls_state['last_step_occupancy'])
            if self.current_tls_state['can_perform_action'] or terminated or truncated:
                break

        avg_occupancy = self.occupancy.calculate_average()
        self.previous_avg_occ = self.current_avg_occ.copy()
        self.current_avg_occ = avg_occupancy.copy()
        infos['this_phase_index'] = self.current_tls_state['this_phase_index']
        infos = self.info_wrapper(infos, occupancy=avg_occupancy)
        return self.current_tls_state.copy(), terminated, truncated, infos

    def close(self) -> None:
        self.env.close()

    # ##############
    # Tools for LLM
    # ##############
    def transform_occ_data(self) -> Dict[str, str]:
        """将 avg_occupancy 与每一个 movement id 对应起来
        Note: 这里和 mask 对应起来, 如果传感器损坏, 则对应的 movement 的占有率变为 -1

        Returns:
            Dict[str, float]: 每一个 movement 的占有率, 如果损坏, 则是 -1
                {
                    '-E1--s': '-1', 
                    'E0--l': '35.928142070770264%', 
                    'E0--s': '58.3832323551178%', 
                    'E2--l': '15.152384340763092%'
                }
        """
        return self._format_occupancy(self.current_avg_occ)

    def _format_occupancy(self, avg_occ: Float32Array) -> Dict[str, str]:
        """Return occupancy for every movement, including signal-controlled right turns."""
        output_dict: Dict[str, str] = {} # 每个 movement 的占有率
        detector_work = self.get_detector_state() # 传感器的状态
        for movement_id, value in zip(self.movement_ids, avg_occ):
            if detector_work.get(movement_id, 'Work') == 'Not Work':
                output_dict[movement_id] = "-1"
            else:
                output_dict[movement_id] = f"{value*100}%" # 转换为占有率
        return output_dict


    # ###########################
    # Custom Tools for TSC Agent
    # ###########################
    def get_available_actions(self) -> List[str]:
        """获得控制信号灯可以做的动作
        """
        tls_available_actions = [f'Phase-{i}' for i in range(self.phase_num)]
        return tls_available_actions
    
    def get_current_phase(self) -> str:
        """获得当前路口的相位
        """
        phase_index = self.current_tls_state['this_phase_index']
        return f'Phase-{phase_index}'
    
    def get_intersection_layout(self) -> Dict[str, Any]:
        """路口的静态路网结构信息
        """
        layout = {}
        for movement_id, info in self.llm_static_information["movement_infos"].items():
            layout[movement_id] = {
                **info,
                "lane_ids": list(self.current_tls_state["movement_lane_ids"].get(movement_id, [])),
                "connection": self.current_tls_state["fromEdge_toEdge"].get(movement_id),
            }
        return layout

    def get_control_constraints(self) -> Dict[str, Any]:
        """Static timing and action semantics supplied to the controller."""
        tls = cast(Any, self.env).tsc_env.scene_objects["tls"].traffic_lights[self.tls_id]
        return {
            "junction_id": self.tls_id,
            "action_type": "choose_next_phase",
            "decision_interval_seconds": tls.tls_action.delta_time,
            "yellow_time_seconds": tls.tls_action.yellow_time,
            "phase_ids": list(self.phase2movements),
        }

    def get_signal_phase_structure(self) -> Dict[str, Any]:
        """路口的信号灯信息
        """
        return self.llm_static_information["phase_infos"]

    def _cache_signal_phase_information(self) -> None:
        """Map actual green signals, including permissive turns, to agent actions."""
        env = cast(Any, self.env).tsc_env
        conn = env.sumo
        phases = env.scene_objects['tls'].traffic_lights[self.tls_id].tls_action.green_phases
        links = conn.trafficlight.getControlledLinks(self.tls_id)
        edge_movements = {(info[0], info[1]): movement for movement, info in
                          self.current_tls_state['fromEdge_toEdge'].items()}
        phase_infos = {}
        for phase_id, phase in enumerate(phases):
            movements, permissive = set(), set()
            for entries, signal in zip(links, phase.state):
                if signal not in 'Gg':
                    continue
                for in_lane, out_lane, _ in entries:
                    movement = edge_movements.get((conn.lane.getEdgeID(in_lane), conn.lane.getEdgeID(out_lane)))
                    if movement is not None:
                        movements.add(movement)
                        if signal == 'g':
                            permissive.add(movement)
            phase_infos[f'Phase {phase_id}'] = {
                'phase_id': phase_id, 'movements': sorted(movements),
                'permissive_movements': sorted(permissive),
            }
        self.llm_static_information['phase_infos'] = phase_infos

    def _get_serving_phases(self, movement_id: str | None) -> List[int]:
        return [info['phase_id'] for info in self.llm_static_information['phase_infos'].values()
                if movement_id in info['movements']]
    
    def get_current_occupancy(self) -> Dict[str, str]:
        """获得当前时刻的路口状态(每一个 movement 的占有率)
        """
        return self.transform_occ_data()

    def get_previous_occupancy(self) -> Dict[str, str]:
        """获得上一个决策周期的路口占有率。"""
        if self.previous_avg_occ is None:
            return {}
        return self._format_occupancy(self.previous_avg_occ)
    
    def get_rescue_movement_ids(self) -> List[str] | None:
        """获得当前 Emergency Vehicle 在什么车道上
        """
        last_step_vehicle_id_list = self.current_tls_state['last_step_vehicle_id_list']
        rescue_movement_ids: List[str] = []
        for vehicle_ids, movement_id in zip(last_step_vehicle_id_list, self.movement_ids):
            for vehicle_id in vehicle_ids:
                vehicle_id_lower = vehicle_id.lower()
                if any(
                    key in vehicle_id_lower
                    for key in ('rescue', 'ambulance', 'police', 'fire')
                ):
                    rescue_movement_ids.append(movement_id)
        
        if len(rescue_movement_ids) == 0:
            return None
        else:
            return rescue_movement_ids

    def get_traditional_decision(self) -> Dict[str, Any]:
        """用当前 phase occupancy 做一个轻量 max-pressure baseline。"""
        movement_occ: Dict[str, float] = {}
        for movement_id, value in self.get_current_occupancy().items():
            if value == "-1":
                continue
            movement_occ[movement_id] = float(str(value).strip('%'))

        phase_scores: Dict[str, float] = {}
        for phase_index, phase_movements in self.phase2movements.items():
            valid_values = [
                movement_occ[movement_id]
                for movement_id in phase_movements
                if movement_id in movement_occ
            ]
            phase_scores[f"Phase-{phase_index}"] = (
                sum(valid_values) if valid_values else 0.0
            )

        best_phase = max(phase_scores, key=lambda phase: phase_scores[phase]) if phase_scores else "Phase-0"
        return {
            "phase_scores": phase_scores,
            "recommended_phase": best_phase,
        }

    def get_junction_situation(self) -> Dict[str, Any]:
        """汇总特殊车辆、道路可通行性和传感器状态。"""
        return {
            "rescue_movements": self.get_rescue_movement_ids(),
            "movement_access": self.get_movement_state(),
            "detector_state": self.get_detector_state(),
        }

    def get_emergency_vehicles(self, vehicle_ids: set[str] | None = None) -> List[Dict[str, Any]]:
        """Read actual approaching vehicles, including those outside the E2 detectors."""
        conn = cast(Any, self.env).tsc_env.sumo
        result = []
        connections = self.current_tls_state['fromEdge_toEdge']
        links = conn.trafficlight.getControlledLinks(self.tls_id)
        for vehicle_id in conn.vehicle.getIDList():
            vehicle_type = conn.vehicle.getTypeID(vehicle_id)
            label = f'{vehicle_id} {vehicle_type}'.lower()
            if vehicle_id not in (vehicle_ids or set()) and not any(
                word in label for word in ('ambulance', 'rescue', 'police', 'fire')
            ):
                continue
            next_tls = next((entry for entry in conn.vehicle.getNextTLS(vehicle_id)
                             if entry[0] == self.tls_id), None)
            if next_tls is None:
                continue
            _, link_index, distance, signal = next_tls
            in_lane, out_lane, _ = links[link_index][0]
            in_edge = conn.lane.getEdgeID(in_lane)
            out_edge = conn.lane.getEdgeID(out_lane)
            movement_id = next((movement for movement, info in connections.items()
                                if info[0] == in_edge and info[1] == out_edge), None)
            speed = float(conn.vehicle.getSpeed(vehicle_id))
            result.append({
                'id': vehicle_id, 'vehicle_type': vehicle_type,
                'route': list(conn.vehicle.getRoute(vehicle_id)),
                'junction_route': [in_edge, out_edge],
                'lane_id': conn.vehicle.getLaneID(vehicle_id),
                'movement_id': movement_id,
                'serving_phases': self._get_serving_phases(movement_id),
                'permissive_phases': [info['phase_id'] for info in self.llm_static_information['phase_infos'].values()
                                      if movement_id in info['permissive_movements']],
                'distance_to_junction_m': round(float(distance), 2),
                'speed_mps': round(speed, 2),
                'eta_at_current_speed_s': round(float(distance) / speed, 2) if speed > 0.1 else None,
                'waiting_time_s': float(conn.vehicle.getWaitingTime(vehicle_id)),
                'signal_state': signal,
            })
        return sorted(result, key=lambda item: item['distance_to_junction_m'])

    def get_movement_details(self, movement_id: str) -> Dict[str, Any]:
        """Read lane demand; detector failures leave the readings unavailable."""
        lane_ids = self.current_tls_state['movement_lane_ids'].get(movement_id)
        if not lane_ids:
            raise ValueError(f'Unknown movement: {movement_id}')
        conn = cast(Any, self.env).tsc_env.sumo
        working = self.get_detector_state().get(movement_id, 'Work') == 'Work'
        return {
            'movement_id': movement_id,
            'serving_phases': self._get_serving_phases(movement_id),
            'detector_state': 'Work' if working else 'Not Work',
            'lanes': [{
                'lane_id': lane,
                'vehicle_count': conn.lane.getLastStepVehicleNumber(lane) if working else None,
                'halting_vehicle_count': conn.lane.getLastStepHaltingNumber(lane) if working else None,
                'mean_speed_mps': round(float(conn.lane.getLastStepMeanSpeed(lane)), 2) if working else None,
            } for lane in lane_ids],
        }

    def get_movement_state(self) -> Dict[str, bool]:
        """Report connected vehicle lanes; congestion alone is not a closure."""
        conn = cast(Any, self.env).tsc_env.sumo
        links = conn.trafficlight.getControlledLinks(self.tls_id)
        connected = {(conn.lane.getEdgeID(in_lane), conn.lane.getEdgeID(out_lane))
                     for entries in links for in_lane, out_lane, _ in entries
                     if conn.lane.getMaxSpeed(in_lane) > 0 and conn.lane.getMaxSpeed(out_lane) > 0}
        return {movement: (info[0], info[1]) in connected
                for movement, info in self.current_tls_state['fromEdge_toEdge'].items()}

    def get_detector_state(self) -> Dict[str, str]:
        """获得探测器的工作状态
        """
        detector_state: Dict[str, str] = {}
        for _movement_id, _mask in zip(self.ls_elements, self.mask):
            detector_state[_movement_id] = _mask
        return detector_state
        
    def set_occ_missing(self, not_work_element: str | None = None) -> None:
        """设置某个 movement 对应的传感器发生损坏, 生成一个 mask
        """
        # Generate the mask
        self.mask = ['Not Work' if element == not_work_element else 'Work' for element in self.ls_elements]
        self.not_work_element = not_work_element
