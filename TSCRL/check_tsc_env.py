'''
@Author: WANG Maonan
@Date: 2023-09-08 15:57:34
@Description: 测试 TSC Env 环境
@LastEditTime: 2023-11-25 15:39:55
'''
import numpy as np
from loguru import logger
from tshub.utils.get_abs_path import get_abs_path
from tshub.utils.init_log import set_logger
from stable_baselines3.common.env_checker import check_env
from rl_utils.make_tsc_env import make_env

path_convert = get_abs_path(__file__)
set_logger(path_convert('./'))


if __name__ == '__main__':
    sumo_cfg = path_convert("../TSCScenario/3way/env/vehicle.sumocfg")
    net_file = path_convert("../TSCScenario/3way/env/3way.net.xml")
    log_path = path_convert('./3way_log/')
    trip_info = path_convert('./rl 3way.xml')
    tsc_env_generate = make_env(
        tls_id='J1',
        num_seconds=500,
        phase_num=3,
        sumo_cfg=sumo_cfg, 
        net_file=net_file,
        trip_info=trip_info,
        use_gui=True,
        log_file=log_path,
        env_index=0,
    )
    tsc_env = tsc_env_generate()

    # Check Env
    print(tsc_env.observation_space.sample())
    print(tsc_env.action_space.n)
    check_env(tsc_env)

    # Simulation with environment
    dones = False
    tsc_env.reset()
    while not dones:
        action = np.random.randint(3)
        states, rewards, truncated, dones, infos = tsc_env.step(action=action)
        logger.info(f"SIM: {infos['step_time']} \n+State:{states}; \n+Reward:{rewards}.")
    tsc_env.close()