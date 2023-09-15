'''
@Author: WANG Maonan
@Date: 2023-09-04 20:43:53
@Description: 信号灯控制环境
@LastEditTime: 2023-09-15 20:19:45
'''
import gymnasium as gym

from typing import List, Dict
from tshub.tshub_env.tshub_env import TshubEnvironment


class TSCEnvironment(gym.Env):
    def __init__(self, sumo_cfg:str, num_seconds:int, tls_id:str, tls_action_type:str, use_gui:bool=False) -> None:
        super().__init__()

        self.tls_id = tls_id

        self.tsc_env = TshubEnvironment(
            sumo_cfg=sumo_cfg,
            is_aircraft_builder_initialized=False, 
            is_vehicle_builder_initialized=False, 
            is_traffic_light_builder_initialized=True,
            tls_ids=[tls_id], 
            num_seconds=num_seconds,
            tls_action_type=tls_action_type,
            use_gui=use_gui
        )

    def reset(self):
        state_infos = self.tsc_env.reset()
        return state_infos['tls'][self.tls_id]
        
    def step(self, action:Dict[str, Dict[str, int]]):
        action = {'tls': {self.tls_id: action}}
        states, rewards, infos, dones = self.tsc_env.step(action)
        truncated = dones

        return states['tls'][self.tls_id], rewards, truncated, dones, infos
    
    def close(self) -> None:
        self.tsc_env._close_simulation()