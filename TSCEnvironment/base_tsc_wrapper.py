'''
@Author: WANG Maonan
@Date: 2023-09-08 15:49:30
@Description: The Wrapper for RL
+ state: 5 个时刻的每一个 movement 的 queue length
+ reward: 路口总的 waiting time
@LastEditTime: 2024-01-05 20:41:34
'''
import numpy as np
import gymnasium as gym

from pathlib import Path
from gymnasium.core import Env
from collections import deque
from typing import Any, SupportsFloat, Tuple, Dict, List

from TSCEnvironment.wrapper_utils import convert_state_to_static_information, find_index

class OccupancyList:
    def __init__(self) -> None:
        self.elements = []

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

    def calculate_average(self) -> float:
        """计算一段时间的平均 occupancy
        """
        arr = np.array(self.elements)
        averages = np.mean(arr, axis=0, dtype=np.float32)/100
        self.clear_elements() # 清空列表
        return averages


class BaseTSCEnvWrapper(gym.Wrapper):
    """TSC Env Wrapper for single junction with tls_id for RL-based model
    """
    def __init__(
            self, env: Env, 
            tls_id:str, 
            phase_num:int, 
            max_states:int=5,
            copy_files:List[str]=[],
        ) -> None:
        super().__init__(env)
        self.tls_id = tls_id # 单路口的 id
        self.phase_num = phase_num
        self.states = deque([self._get_initial_state()] * max_states, maxlen=max_states) # 队列
        self.movement_ids = None
        self.phase2movements = None
        self.occupancy = OccupancyList()
        self.episode = 0
        self.copy_files = copy_files # 需要保留的文件

    def _get_initial_state(self) -> List[int]:
        # 返回初始状态，这里假设所有状态都为 0
        return [0]*12
    
    def get_state(self):
        """返回 np.array, 同时根据 mask 遮住部分 movement 的信息
        """
        not_working_index = find_index(self.movement_ids, self.not_work_element) # 无法工作的探测器的 index
        tsc_state = np.array(self.states, dtype=np.float32)
        if not_working_index is not None: # 当存在传感器损坏的时候
            tsc_state[:,not_working_index] = -1
        return tsc_state
    
    @property
    def action_space(self):
        return gym.spaces.Discrete(self.phase_num)
    
    @property
    def observation_space(self):
        obs_space = gym.spaces.Box(
            low=np.zeros((7,12)), # (5,12) 是 occ, (2,12) 是 this phase 和 next phase -> (7, 12)
            high=np.ones((7,12)),
            shape=(7,12)
        ) # self.states 是一个时间序列
        return obs_space
    
    # Wrapper
    def state_wrapper(self, state):
        """返回当前每个 movement 的 occupancy
        """
        occupancy = state['tls'][self.tls_id]['last_step_occupancy']
        can_perform_action = state['tls'][self.tls_id]['can_perform_action']
        one_hot_this_phase = np.array([int(value) for value in state['tls'][self.tls_id]['this_phase']]) # 当前状态
        one_hot_next_phase = np.array([int(value) for value in state['tls'][self.tls_id]['next_phase']]) # 下一个状态
        return occupancy, can_perform_action, one_hot_this_phase, one_hot_next_phase
    
    def reward_wrapper(self, states) -> float:
        """返回整个路口的排队长度的平均值
        """
        total_waiting_time = 0
        for _, veh_info in states['vehicle'].items():
            total_waiting_time += veh_info['waiting_time']
        return -total_waiting_time
    
    def info_wrapper(self, infos, occupancy):
        """在 info 中加入一些信息:
        1. 当前的仿真时间
        2. 每个 phase 的占有率
        3. 当前的 phase index
        """
        movement_occ = {key: value for key, value in zip(self.movement_ids, occupancy)}
        if self.not_work_element in movement_occ:
            movement_occ[self.not_work_element] = -1 # 如果存在无法工作的传感器
        phase_occ = {}
        for phase_index, phase_movements in self.phase2movements.items():
            phase_occ[phase_index] = sum([movement_occ[phase] for phase in phase_movements])
        
        infos['phase_occ'] = phase_occ
        return infos

    def reset(self, seed=1) -> Tuple[Any, Dict[str, Any]]:
        """reset 时初始化 (1) 静态信息; (2) 动态信息
        """
        # reset 之前进行文件的复制
        if self.episode > 0:
            for file_path in self.copy_files:
                file = Path(file_path)
                new_file = file.with_stem(f"{file.stem}_{self.episode}")
                file.rename(new_file)

        self.episode += 1
        
        # 环境的 reset
        state =  self.env.reset()
        self.current_tls_state = state['tls'][self.tls_id].copy() # 当前的状态
        
        # 初始化路口静态信息
        self.llm_static_information = convert_state_to_static_information(state['tls'][self.tls_id]) # 路口的静态信息
        self.movement_ids = state['tls'][self.tls_id]['movement_ids']
        self.phase2movements = state['tls'][self.tls_id]['phase2movements']
        self.current_avg_occ = [0]*12 # 初始化每一个 movement 的占有率

        # For Detector State
        self.ls_elements = [element for element in self.movement_ids if element.endswith('--l') or element.endswith('--s')]
        self.mask = ['Work'] * len(self.ls_elements)
        self.not_work_element = None # 初始化时候没有传感器是损坏的

        # 处理路口动态信息
        occupancy, _, one_hot_this_phase, one_hot_next_phase = self.state_wrapper(state=state)
        self.states.append(occupancy)
        state = self.get_state()
        # 合并路口的 state, (5,12) + (2,12) -> (7,12)
        state = np.vstack((state, one_hot_this_phase[np.newaxis, :], one_hot_next_phase[np.newaxis, :])).astype(np.float32)
        return state, {'step_time':0}
    
    def step(self, action: int) -> Tuple[Any, SupportsFloat, bool, bool, Dict[str, Any]]:
        can_perform_action = False
        while not can_perform_action:
            action = {self.tls_id: action} # 构建单路口 action 的动作
            states, rewards, truncated, dones, infos = super().step(action) # 与环境交互
            self.current_tls_state = states['tls'][self.tls_id].copy() # 当前的状态
            occupancy, can_perform_action, one_hot_this_phase, one_hot_next_phase = self.state_wrapper(state=states) # 处理每一帧的数据
            # 记录每一时刻的数据
            self.occupancy.add_element(occupancy)
        
        # 处理好的时序的 state
        avg_occupancy = self.occupancy.calculate_average() # 计算平均占有率
        self.current_avg_occ = avg_occupancy.copy()
        rewards = self.reward_wrapper(states=states) # 计算 vehicle waiting time
        # Update info
        infos['this_phase_index'] = states['tls'][self.tls_id]['this_phase_index']
        infos = self.info_wrapper(infos, occupancy=avg_occupancy) # info 里面包含每个 phase 的排队
        
        self.states.append(avg_occupancy) # 获得时间序列
        state = self.get_state() # 得到 state, 需要根据 mask 进行处理
        # 合并路口的 state, (5,12) + (2,12) -> (7,12)
        state = np.vstack((state, one_hot_this_phase[np.newaxis, :], one_hot_next_phase[np.newaxis, :])).astype(np.float32)
        return state, rewards, truncated, dones, infos
    
    def close(self) -> None:
        return super().close()
    
    # ##############
    # Tools for LLM
    # ##############
    def transform_occ_data(self) -> Dict[str, float]:
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
        output_dict = {} # 每个 movement 的占有率
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
    def get_available_actions(self) -> List[int]:
        """获得控制信号灯可以做的动作
        """
        tls_available_actions = [f'Phase-{i}' for i in range(self.phase_num)]
        return tls_available_actions
    
    def get_current_phase(self):
        """获得当前路口的相位
        """
        phase_index = self.current_tls_state['this_phase_index']
        return f'Phase-{phase_index}'
    
    def get_intersection_layout(self) -> Dict[str, str|float|int]:
        """路口的静态路网结构信息
        """
        return self.llm_static_information["movement_infos"]

    def get_signal_phase_structure(self) -> Dict[str, List[str]]:
        """路口的信号灯信息
        """
        return self.llm_static_information["phase_infos"]
    
    def get_current_occupancy(self) -> Dict[str, float]:
        """获得当前时刻的路口状态(每一个 movement 的占有率)
        """
        return self.transform_occ_data()
    
    def get_rescue_movement_ids(self):
        """获得当前 Emergency Vehicle 在什么车道上
        """
        last_step_vehicle_id_list = self.current_tls_state['last_step_vehicle_id_list']
        rescue_movement_ids = []
        for vehicle_ids, movement_id in zip(last_step_vehicle_id_list, self.movement_ids):
            for vehicle_id in vehicle_ids:
                if 'rescue' in vehicle_id:
                    rescue_movement_ids.append(movement_id)
        
        if len(rescue_movement_ids) == 0:
            return None
        else:
            return rescue_movement_ids

    def get_movement_state(self):
        """得到每一个 movement 是否可以正常通行
        """
        _sumo = self.env.tsc_env.sumo

        # 当 movement 的速度小于 5 的时候, 认为是不可通行的
        movement_state = dict()
        for _movement_id, (_in_edge, _out_edge, _, _) in self.current_tls_state['fromEdge_toEdge'].items():
            _, direction = _movement_id.split('--')
            if direction != 'r': # 忽略右转
                movement_state[_movement_id] = _sumo.lane.getMaxSpeed(f'{_out_edge}_0') > 5 
        
        return movement_state
    
    def get_detector_state(self):
        """获得探测器的工作状态
        """
        detector_state = {}
        for _movement_id, _mask in zip(self.ls_elements, self.mask):
            detector_state[_movement_id] = _mask
        return detector_state
        

    # #################################
    # Custom Tools for Change Env State
    # ##################################
    def set_edge_speed(self, edge_id:str, speed:float=3) -> None:
        """设置 edge 速度, 模拟道路是否发生事故等
        + 出现事故则车速降低
        + 前一个路口排队溢出, 进入的车辆车速较低
        """
        _sumo = self.env.tsc_env.sumo
        _sumo.edge.setMaxSpeed(edge_id, speed)
    
    def set_occ_missing(self, not_work_element:str=None) -> None:
        """设置某个 movement 对应的传感器发生损坏, 生成一个 mask
        """
        # Generate the mask
        self.mask = ['Not Work' if element == not_work_element else 'Work' for element in self.ls_elements]
        self.not_work_element = not_work_element