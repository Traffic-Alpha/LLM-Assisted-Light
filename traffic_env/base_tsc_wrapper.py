'''
@Author: WANG Maonan
@Date: 2023-09-08 15:49:30
@Description: Base wrapper for single-intersection traffic signal control
+ state: 5 个时刻的每一个 movement 的 queue length
+ reward: 路口总的 waiting time
@LastEditTime: 2026-06-12 00:21:33
'''
import numpy as np
import gymnasium as gym

from gymnasium.core import Env
from collections import deque
from typing import Any, Deque, SupportsFloat, Tuple, Dict, List, TypeAlias, cast
from numpy.typing import NDArray

from traffic_env.wrapper_utils import convert_state_to_static_information, find_index

Float32Array: TypeAlias = NDArray[np.float32]
EnvStepResult: TypeAlias = Tuple[Dict[str, Any], SupportsFloat, bool, bool, Dict[str, Any]]
DecisionStepResult: TypeAlias = Tuple[
    Dict[str, Any],
    SupportsFloat,
    bool,
    bool,
    Dict[str, Any],
    Float32Array,
    Float32Array,
]

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


class BaseTSCEnvWrapper(gym.Wrapper):
    """TSC Env Wrapper for single junction with tls_id."""
    def __init__(
            self, env: Env, 
            tls_id:str, 
            phase_num:int, 
            max_states:int=5,
        ) -> None:
        super().__init__(env)
        self.tls_id = tls_id # 单路口的 id
        self.phase_num = phase_num
        self.states: Deque[Float32Array] = deque(
            [self._get_initial_state()] * max_states,
            maxlen=max_states,
        ) # state 队列
        self.movement_ids: List[str] = []
        self.phase2movements: Dict[Any, List[str]] = {}
        self.occupancy = OccupancyList()
        self.current_tls_state: Dict[str, Any] = {}
        self.llm_static_information: Dict[str, Dict[str, Any]] = {
            "movement_infos": {},
            "phase_infos": {},
        }
        self.current_avg_occ: Float32Array = cast(Float32Array, np.zeros(12, dtype=np.float32))
        self.previous_avg_occ: Float32Array | None = None
        self.ls_elements: List[str] = []
        self.mask: List[str] = []
        self.not_work_element: str | None = None

    def _get_initial_state(self) -> Float32Array:
        # 返回初始状态，这里假设所有状态都为 0
        return cast(Float32Array, np.zeros(12, dtype=np.float32))
    
    def get_state(self) -> Float32Array:
        """返回 np.array, 同时根据 mask 遮住部分 movement 的信息
        """
        not_working_index = find_index(self.movement_ids, self.not_work_element) # 无法工作的探测器的 index
        tsc_state = cast(Float32Array, np.asarray(self.states, dtype=np.float32))
        if not_working_index is not None: # 当存在传感器损坏的时候
            tsc_state[:,not_working_index] = -1 # 故障的检测器对应的 movement 的占有率设置为 -1
        return tsc_state
    
    @property
    def action_space(self) -> gym.spaces.Discrete:
        return gym.spaces.Discrete(self.phase_num)
    
    @property
    def observation_space(self) -> gym.spaces.Box:
        obs_space = gym.spaces.Box(
            low=np.zeros((7,12)), # (5,12) 是 occ, (2,12) 是 this phase 和 next phase -> (7, 12)
            high=np.ones((7,12)),
            shape=(7,12)
        ) # self.states 是一个时间序列
        return obs_space
    
    # ########
    # Wrapper
    # ########
    def state_wrapper(self, state: Dict[str, Any]) -> tuple[List[float], bool, Float32Array, Float32Array]:
        """返回当前每个 movement 的 occupancy
        """
        tls_state = state['tls'][self.tls_id]
        occupancy = cast(List[float], tls_state['last_step_occupancy'])
        can_perform_action = bool(tls_state['can_perform_action'])
        one_hot_this_phase = cast(
            Float32Array,
            np.asarray([int(value) for value in tls_state['this_phase']], dtype=np.float32),
        ) # 当前状态
        one_hot_next_phase = cast(
            Float32Array,
            np.asarray([int(value) for value in tls_state['next_phase']], dtype=np.float32),
        ) # 下一个状态
        return occupancy, can_perform_action, one_hot_this_phase, one_hot_next_phase
    
    def reward_wrapper(self, states: Dict[str, Any]) -> float:
        """返回整个路口的排队长度的平均值
        """
        total_waiting_time = 0.0
        for _, veh_info in states['vehicle'].items():
            total_waiting_time += veh_info['waiting_time']
        return -total_waiting_time
    
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
        ) -> Tuple[Any, Dict[str, Any]]:
        """reset 时初始化 (1) 静态信息; (2) 动态信息
        """
        # 环境的 reset
        state = cast(Dict[str, Any], cast(Any, self.env).reset(seed=seed, options=options))
        self.current_tls_state = state['tls'][self.tls_id].copy() # 当前的状态
        
        # 初始化路口静态信息
        self.llm_static_information = convert_state_to_static_information(state['tls'][self.tls_id]) # 路口的静态信息
        self.movement_ids = state['tls'][self.tls_id]['movement_ids']
        self.phase2movements = state['tls'][self.tls_id]['phase2movements']
        self.current_avg_occ = cast(Float32Array, np.zeros(12, dtype=np.float32)) # 初始化每一个 movement 的占有率
        self.previous_avg_occ = None

        # For Detector State
        self.ls_elements = [element for element in self.movement_ids if element.endswith('--l') or element.endswith('--s')]
        self.mask = ['Work'] * len(self.ls_elements)
        self.not_work_element = None # 初始化时候没有传感器是损坏的

        # 处理路口动态信息
        occupancy, _, one_hot_this_phase, one_hot_next_phase = self.state_wrapper(state=state)
        self.states.append(cast(Float32Array, np.asarray(occupancy, dtype=np.float32)))
        wrapped_state = self.get_state()
        # 合并路口的 state, (5,12) + (2,12) -> (7,12)
        wrapped_state = cast(
            Float32Array,
            np.vstack((wrapped_state, one_hot_this_phase[np.newaxis, :], one_hot_next_phase[np.newaxis, :])).astype(np.float32),
        )
        return wrapped_state, {'step_time':0}
    
    def _step_env(self, tsc_action: Dict[str, int]) -> EnvStepResult:
        return cast(EnvStepResult, cast(Any, self.env).step(tsc_action))

    def _advance_to_decision_point(self, action: int) -> DecisionStepResult:
        tsc_action = {self.tls_id: action}

        while True:
            states, rewards, truncated, dones, infos = self._step_env(tsc_action) # 与环境交互
            self.current_tls_state = states['tls'][self.tls_id].copy() # 当前的状态
            occupancy, can_perform_action, one_hot_this_phase, one_hot_next_phase = self.state_wrapper(state=states) # 处理每一帧的数据
            # 记录每一时刻的数据
            self.occupancy.add_element(occupancy)
            if can_perform_action or truncated or dones:
                return states, rewards, truncated, dones, infos, one_hot_this_phase, one_hot_next_phase

    def step(self, action: int) -> Tuple[Any, SupportsFloat, bool, bool, Dict[str, Any]]:
        (
            states,
            rewards,
            truncated,
            dones,
            infos,
            one_hot_this_phase,
            one_hot_next_phase,
        ) = self._advance_to_decision_point(action)
        
        # 处理好的时序的 state
        avg_occupancy = self.occupancy.calculate_average() # 计算平均占有率
        self.previous_avg_occ = cast(Float32Array, self.current_avg_occ.copy())
        self.current_avg_occ = cast(Float32Array, avg_occupancy.copy())
        rewards = self.reward_wrapper(states=states) # 计算 vehicle waiting time
        # Update info
        infos['this_phase_index'] = states['tls'][self.tls_id]['this_phase_index']
        infos = self.info_wrapper(infos, occupancy=avg_occupancy) # info 里面包含每个 phase 的排队
        
        self.states.append(avg_occupancy) # 获得时间序列
        state = self.get_state() # 得到 state, 需要根据 mask 进行处理
        # 合并路口的 state, (5,12) + (2,12) -> (7,12)
        state = cast(
            Float32Array,
            np.vstack((state, one_hot_this_phase[np.newaxis, :], one_hot_next_phase[np.newaxis, :])).astype(np.float32),
        )
        return state, rewards, truncated, dones, infos
    
    def close(self) -> None:
        super().close()
    
    
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
        output_dict: Dict[str, str] = {} # 每个 movement 的占有率
        detector_work = self.get_detector_state() # 传感器的状态
        for movement_id, value in zip(self.movement_ids, self.current_avg_occ):
            if 'r' in movement_id:
                continue
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
        return self.llm_static_information["movement_infos"]

    def get_signal_phase_structure(self) -> Dict[str, Any]:
        """路口的信号灯信息
        """
        return self.llm_static_information["phase_infos"]
    
    def get_current_occupancy(self) -> Dict[str, str]:
        """获得当前时刻的路口状态(每一个 movement 的占有率)
        """
        return self.transform_occ_data()

    def get_previous_occupancy(self) -> Dict[str, str]:
        """获得上一个决策周期的路口占有率。"""
        if self.previous_avg_occ is None:
            return {}

        output_dict: Dict[str, str] = {}
        detector_work = self.get_detector_state()
        for movement_id, value in zip(self.movement_ids, self.previous_avg_occ):
            if 'r' in movement_id:
                continue
            if detector_work.get(movement_id, 'Work') == 'Not Work':
                output_dict[movement_id] = "-1"
            else:
                output_dict[movement_id] = f"{value*100}%"
        return output_dict
    
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
        """用当前 phase occupancy 做一个轻量 SOTL baseline。"""
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
                max(valid_values) if valid_values else 0.0
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

    def get_movement_state(self) -> Dict[str, bool]:
        """得到每一个 movement 是否可以正常通行
        """
        _sumo = cast(Any, self.env).tsc_env.sumo

        # 当 movement 的速度小于 5 的时候, 认为是不可通行的
        movement_state: Dict[str, bool] = {}
        for _movement_id, (_in_edge, _out_edge, _, _) in self.current_tls_state['fromEdge_toEdge'].items():
            _, direction = _movement_id.split('--')
            if direction != 'r': # 忽略右转
                movement_state[_movement_id] = _sumo.lane.getMaxSpeed(f'{_out_edge}_0') > 5 
        
        return movement_state
    
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
