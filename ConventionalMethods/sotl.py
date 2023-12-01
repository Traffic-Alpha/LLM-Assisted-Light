'''
@Author: WANG Maonan
@Date: 2023-12-01 20:58:40
@Description: 感应控制算法
@LastEditTime: 2023-12-01 22:43:39
'''
import sys
from pathlib import Path

parent_directory = Path(__file__).resolve().parent.parent
if str(parent_directory) not in sys.path:
    sys.path.insert(0, str(parent_directory))

from loguru import logger
from tshub.utils.get_abs_path import get_abs_path
from tshub.utils.init_log import set_logger

from TSCEnvironment.tsc_env import TSCEnvironment
from TSCEnvironment.llm_tsc_wrapper import LLMTSCEnvWrapper

path_convert = get_abs_path(__file__)
set_logger(path_convert('./'))


if __name__ == '__main__':
    env_name = '4way'
    route_type = 'vehicle' # vehicle_pedestrian
    tls_action_type = 'next_or_not' # next_or_not, choose_next_phase
    phase_num = 4
    sumo_cfg = path_convert(f"../TSCScenario/{env_name}/env/{route_type}.sumocfg")
    net_file = path_convert(f"../TSCScenario/{env_name}/env/{env_name}.net.xml")
    log_path = path_convert(f'./')
    trip_info = path_convert(f'./{env_name}_SOTL.tripinfo.xml')

    tsc_scenario = TSCEnvironment(
        sumo_cfg=sumo_cfg, 
        net_file=net_file,
        trip_info=trip_info,
        num_seconds=500,
        tls_id='J1', 
        tls_action_type=tls_action_type,
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
    llm_env.reset()
    while not dones:
        states, rewards, truncated, dones, infos = llm_env.step(action=action)
        this_phase_index = infos['this_phase_index']
        next_phase_index = (this_phase_index + 1)%phase_num

        if infos['phase_occ'][next_phase_index] > infos['phase_occ'][this_phase_index]:
            # 如果下一个相位的 occ 比现在的大, 则切换到下一个相位
            action = 0 # 切换到下一个相位
        else:
            action = 0 # 保持不变
            last_change_light_time = infos['step_time']
    llm_env.close()