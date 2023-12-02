'''
@Author: WANG Maonan
@Date: 2023-12-01 20:58:40
@Description: 感应控制算法
Command: 
@ Scenario-1
-> python sotl.py --env_name '3way' --phase_num 3
-> python sotl.py --env_name '4way' --phase_num 4
@ Scenario-2
-> python sotl.py --env_name '3way' --phase_num 3 --edge_block 'E1'
-> python sotl.py --env_name '4way' --phase_num 4 --edge_block 'E1'
@ Scenario-3
-> python sotl.py --env_name '3way' --phase_num 3 --detector_break 'E0--s'
-> python sotl.py --env_name '4way' --phase_num 4 --detector_break 'E0--s'
@LastEditTime: 2023-12-02 18:05:05
'''
import sys
from pathlib import Path

parent_directory = Path(__file__).resolve().parent.parent
if str(parent_directory) not in sys.path:
    sys.path.insert(0, str(parent_directory))

import argparse
from loguru import logger
from tshub.utils.get_abs_path import get_abs_path
from tshub.utils.init_log import set_logger

from TSCEnvironment.tsc_env import TSCEnvironment
from TSCEnvironment.llm_wrapper import LLMTSCEnvWrapper

path_convert = get_abs_path(__file__)
set_logger(path_convert('./'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--env_name', type=str, default='4way', help='Environment name')
    parser.add_argument('--phase_num', type=int, default=4, help='Phase number')
    parser.add_argument('--edge_block', type=str, default=None, help='Edge block')
    parser.add_argument('--detector_break', type=str, default=None, help='Detector break')

    args = parser.parse_args()
    env_name = args.env_name
    phase_num = args.phase_num
    edge_block = args.edge_block
    detector_break = args.detector_break

    route_type = 'vehicle' # vehicle_pedestrian
    sumo_cfg = path_convert(f"../TSCScenario/{env_name}/env/{route_type}.sumocfg")
    net_file = path_convert(f"../TSCScenario/{env_name}/env/{env_name}.net.xml")
    log_path = path_convert(f'./')
    trip_info = path_convert(f'./{env_name}_SOTL.tripinfo.xml')

    tsc_scenario = TSCEnvironment(
        sumo_cfg=sumo_cfg, 
        net_file=net_file,
        trip_info=trip_info,
        num_seconds=1000,
        tls_id='J1', 
        tls_action_type='next_or_not',
        use_gui=True,
    ) # 初始化环境
    
    llm_env = LLMTSCEnvWrapper(
        env=tsc_scenario,
        tls_id='J1',
        phase_num=phase_num,
        copy_files=[trip_info]
    ) # wrapper for llm

    # Simulation with environment
    dones = False
    action = 1 # 初始的动作, 保持不变
    sim_step = 0 # 初始仿真时间
    llm_env.reset()
    while not dones:
        # 设置 edge block
        if edge_block is not None:
            if sim_step>50 and sim_step<100:
                llm_env.set_edge_speed(edge_block, speed=1)
            else:
                llm_env.set_edge_speed(edge_block, speed=13)
        
        # 设置 detector break
        if detector_break is not None:
            if sim_step>200 and sim_step<400:
                llm_env.set_occ_missing(not_work_element=detector_break) # 存在这 400s 的时候传感器数据是损坏的
            else:
                llm_env.set_occ_missing(not_work_element='') # 其余时间传感器数据是好的

        states, rewards, truncated, dones, infos = llm_env.step(action=action)
        this_phase_index = infos['this_phase_index']
        next_phase_index = (this_phase_index + 1)%phase_num

        if infos['phase_occ'][next_phase_index] > infos['phase_occ'][this_phase_index]:
            # 如果下一个相位的 occ 比现在的大, 则切换到下一个相位
            action = 0 # 切换到下一个相位
        else:
            action = 0 # 保持不变
            last_change_light_time = infos['step_time']
        sim_step = infos['step_time']
        logger.info(f'SIM: {infos}')
    llm_env.close()