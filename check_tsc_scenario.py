'''
@Author: WANG Maonan
@Date: 2023-09-05 11:38:20
@Description: Traffic Signal Control Scenario
@LastEditTime: 2023-10-15 18:03:29
'''
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
    sumo_cfg = path_convert("./TSCScenario/J1/env/J1.sumocfg")
    database_path = path_convert("./junction.db")
    tsc_scenario = TSCEnvironment(
        sumo_cfg=sumo_cfg, 
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
    while not dones:
        action = np.random.randint(4)
        states, dones, infos = tsc_wrapper.step(action=action, explanation="")
        traditional_decision = tsc_wrapper.get_traditional_decision() # 获得传统方法的结果
        logger.info(f"SIM: {infos['step_time']} \n{dict_to_str(states)}")

    tsc_wrapper.close()