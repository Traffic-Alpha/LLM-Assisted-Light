'''
@Author: WANG Maonan
@Date: 2023-09-05 11:27:05
@Description: 处理 Traffic Signal Control Env 的 State
1. 处理路口的信息
    - 路口的静态信息, 道路拓扑信息 (这个是用于相似度的比较)
    - 道路的动态信息, 会随着时间进行变化（每个 phase 的排队长度和当前所在的 phase）
2. 实现一些查询的接口
    - （a）所有可能的动作, 这里就是 `change phase 0 green` or `change phase k green`, 切换到某个 phase 是绿灯
    - （b）作出动作后，成为的新的 phase
    - （c）作出某个动作后, phase 对应排队长度的变化的预测（这里预测可以直接使用 MCT 来进行预测，或者服从某个分布，这里需要做一个预测）
    - （d）比较前后两次 phase 之间的排队的增加
    - （e）分析路口的性能（就是根据 c 的结果做进一步的计算）
@LastEditTime: 2023-09-18 17:10:46
'''
import gymnasium as gym
from gymnasium.core import Env
from typing import Any, SupportsFloat, Tuple, Dict, List
from tshub.utils.nested_dict_conversion import create_nested_defaultdict, defaultdict2dict

from TSCEnvironment.wrapper_utils import (
    convert_state_to_static_information, 
    predict_queue_length, 
    OccupancyList
)

    
class TSCEnvWrapper(gym.Wrapper):
    def __init__(self, env: Env) -> None:
        super().__init__(env)
        # Static Information
        self.movement_ids = None
        self.phase_num = None # phase 数量
        self.llm_static_information = None # Static information, (1). Intersection Geometry; (2). Signal Phases Structure

        # Dynamic Information
        self.state = None # 当前的 state
        self.rescue_movement = None
        self.last_state = None # 上一时刻的 state
        self.occupancy = OccupancyList()

    def transform_occ_data(self, occ:List[float]) -> Dict[str, float]:
        """将 avg_occupancy 与每一个 movement id 对应起来

        Args:
            occ (List[float]): _description_

        Returns:
            Dict[str, float]: _description_
        """
        output_dict = {}
        for movement_id, value in zip(self.movement_ids, occ):
            if 'r' in movement_id:
                continue
            output_dict[movement_id] = f"{value*100}%" # 转换为占有率
        return output_dict

    def get_rescue_movement_ids(self, last_step_vehicle_id_list):
        """获得当前 emergy vehicle 在什么车道上
        """
        rescue_movement_ids = []
        for vehicle_ids, movement_id in zip(last_step_vehicle_id_list, self.movement_ids):
            for vehicle_id in vehicle_ids:
                if 'rescue' in vehicle_id:
                    rescue_movement_ids.append(movement_id)
        return rescue_movement_ids

    def state_wrapper(self, state):
        """从 state 中返回 occupancy
        """
        occupancy = state['last_step_occupancy']
        last_step_vehicle_id_list = state['last_step_vehicle_id_list']
        return occupancy, last_step_vehicle_id_list
    

    def reset(self) -> Tuple[Any, Dict[str, Any]]:
        state =  self.env.reset()
        # Initialize static information
        self.phase_num = len(state['phase2movements'])
        self.movement_ids = state['movement_ids']
        self.llm_static_information = convert_state_to_static_information(state)

        # Dynamic junction information
        occupancy, last_step_vehicle_id_list = self.state_wrapper(state=state)
        self.state = self.transform_occ_data(occupancy)
        self.rescue_movement = self.get_rescue_movement_ids(last_step_vehicle_id_list)
        return self.state
    

    def step(self, action: Any) -> Tuple[Any, SupportsFloat, bool, bool, Dict[str, Any]]:
        """更新路口的 state
        """
        can_perform_action = False
        while not can_perform_action:
            states, rewards, truncated, dones, infos = super().step(action) # 与环境交互
            occupancy, last_step_vehicle_id_list = self.state_wrapper(state=states) # 处理每一帧的数据
            self.occupancy.add_element(occupancy)
            can_perform_action = states['can_perform_action']

        avg_occupancy = self.occupancy.calculate_average() # 计算平均的 average occupancy
        self.last_state = self.state
        self.state = self.transform_occ_data(avg_occupancy) # 计算每一个 phase 的 occupancy
        self.rescue_movement = self.get_rescue_movement_ids(last_step_vehicle_id_list)

        return self.state, dones, infos
    

    def close(self) -> None:
        return super().close()
    

    # ###########################
    # Custom Tools for TSC Agent
    # ###########################
    def get_available_actions(self) -> List[int]:
        """获得控制信号灯可以做的动作
        """
        tls_available_actions = list(range(self.phase_num))
        return tls_available_actions
    
    def get_intersection_layout(self) -> Dict[str, str|float|int]:
        """路口的静态路网结构信息
        """
        return self.llm_static_information["movement_infos"]

    def get_signal_phase_structure(self) -> Dict[str, List[str]]:
        """路口的信号灯信息
        """
        return self.llm_static_information["phase_infos"]
    
    def get_current_occupancy(self) -> Dict[str, float]:
        """获得当前时刻的路口状态
        """
        return self.state
    
    def get_previous_occupancy(self) -> Dict[str, float]:
        """获得上一个时刻的路口状态
        """
        return self.last_state
    
    def get_rescue_movement(self) -> List[str]:
        """返回车道上是否有急救车
        """
        return self.rescue_movement
    
    def predict_future_scene(self, phase_index):
        """预测将 phase index 设置为绿灯后, 路口排队长度的变化
        """
        try:
            phase_index = int(phase_index)
        except:
            raise ValueError(f"phase_index need to be a number, rather than {phase_index}.")
        predict_state = create_nested_defaultdict()
        for _tls_id, _tls_info in self.state.items():
            _tls_phase_queue_info = _tls_info['phase_queue_lengths']
            for _phase_index, (_phase_name, _queue_info) in enumerate(_tls_phase_queue_info.items()):
                if _phase_index == phase_index: # green light
                    _p_state = predict_queue_length(_queue_info, is_green=True)
                else: # red light
                    _p_state = predict_queue_length(_queue_info, is_green=False)
                predict_state[_tls_id][_phase_name] = _p_state
        return defaultdict2dict(predict_state)