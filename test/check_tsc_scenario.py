'''
@Author: WANG Maonan
@Date: 2023-11-30 16:10:31
@Description: 检查环境是否正确
@LastEditTime: 2023-11-30 16:12:43
'''
import sys
from pathlib import Path

parent_directory = Path(__file__).resolve().parent.parent
if str(parent_directory) not in sys.path:
    sys.path.insert(0, str(parent_directory))

import numpy as np
from loguru import logger
from tshub.utils.format_dict import dict_to_str
from tshub.utils.get_abs_path import get_abs_path
from tshub.utils.init_log import set_logger

from TSCEnvironment.tsc_env import TSCEnvironment
from TSCEnvironment.tsc_env_wrapper import TSCEnvWrapper

path_convert = get_abs_path(__file__)
set_logger(path_convert('./'))


if __name__ == '__main__':
    env_name = '4way'
    route_type = 'vehicle' # vehicle_pedestrian
    tls_action_type = 'next_or_not' # next_or_not, choose_next_phase
    sumo_cfg = path_convert("../TSCScenario/4way/env/vehicle_pedestrian.sumocfg")
    net_file = path_convert("../TSCScenario/4way/env/4way.net.xml")
    trip_info = path_convert("./trip.out.xml")
    database_path = path_convert("./junction.db")

    tsc_scenario = TSCEnvironment(
        sumo_cfg=sumo_cfg, 
        net_file=net_file,
        trip_info=trip_info,
        num_seconds=300,
        tls_id='J4', 
        tls_action_type='choose_next_phase',
        use_gui=True
    )
    tsc_wrapper = TSCEnvWrapper(
        env=tsc_scenario, 
        database=database_path
    )

    # Simulation with environment
    dones = False
    tsc_wrapper.reset()
    tsc_wrapper.set_edge_speed(edge_id='E2', speed=1) # 设置 edge speed, 模拟修路等情况
    while not dones:
        action = np.random.randint(4)
        states, dones, infos = tsc_wrapper.step(action=action, explanation="")
        phase_max_occupancy, preliminary_decision = tsc_wrapper.get_traditional_decision() # 获得传统方法的结果
        tsc_wrapper.get_movement_state() # 获得当前的 movement 是否是可以通行的
        logger.info(f"SIM: {infos['step_time']} \n{dict_to_str(states)}")

        if infos['step_time'] > 200:
            tsc_wrapper.set_edge_speed(edge_id='E2', speed=15)

    tsc_wrapper.close()