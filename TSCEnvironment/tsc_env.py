'''
@Author: WANG Maonan
@Date: 2023-09-04 20:43:53
@Description: 信号灯控制环境
@LastEditTime: 2023-11-25 15:17:53
'''
import gymnasium as gym

from typing import Dict
from tshub.tshub_env.tshub_env import TshubEnvironment

class TSCEnvironment(gym.Env):
    def __init__(self, 
                 sumo_cfg:str,
                 net_file:str,
                 trip_info: str, 
                 num_seconds:int, 
                 tls_id:str, 
                 tls_action_type:str, 
                 use_gui:bool=False
        ) -> None:
        super().__init__()

        self.tls_id = tls_id

        self.tsc_env = TshubEnvironment(
            sumo_cfg=sumo_cfg,
            net_file=net_file,
            is_map_builder_initialized=True, # For scenario render
            is_aircraft_builder_initialized=False, 
            is_vehicle_builder_initialized=True, # calculate waiting time for vehicle
            is_traffic_light_builder_initialized=True,
            is_person_builder_initialized=False,
            trip_info=trip_info, # 记录仿真中所有 object 的指标
            tls_ids=[tls_id], 
            num_seconds=num_seconds,
            tls_action_type=tls_action_type,
            use_gui=use_gui
        )

    def reset(self):
        state_infos = self.tsc_env.reset()
        return state_infos
        
    def step(self, action: Dict[str, Dict[str, int]]):
        action = {'tls': action}
        states, rewards, infos, dones = self.tsc_env.step(action)
        truncated = dones

        return states, rewards, truncated, dones, infos
    
    def close(self) -> None:
        self.tsc_env._close_simulation()