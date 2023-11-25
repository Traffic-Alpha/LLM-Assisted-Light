'''
@Author: WANG Maonan
@Date: 2023-09-08 18:57:35
@Description: 使用训练好的 RL Agent 进行测试
@LastEditTime: 2023-11-25 20:50:36
'''
import torch
from loguru import logger
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv
from rl_utils.make_tsc_env import make_env
from tshub.utils.get_abs_path import get_abs_path

path_convert = get_abs_path(__file__)
logger.remove()

if __name__ == '__main__':
    # #########
    # Init Env
    # #########
    env_name = '3way'
    sumo_cfg = path_convert(f"../TSCScenario/{env_name}/env/vehicle_pedestrian.sumocfg")
    net_file = path_convert(f"../TSCScenario/{env_name}/env/{env_name}.net.xml")
    log_path = path_convert(f'./')
    trip_info = path_convert(f'./rl_{env_name}.tripinfo.xml')
    params = {
        'tls_id':'J1',
        'num_seconds':500,
        'phase_num':3,
        'sumo_cfg':sumo_cfg,
        'net_file':net_file,
        'use_gui':True,
        'log_file':log_path,
        'trip_info':trip_info,
        'copy_files': [trip_info]
    }
    env = SubprocVecEnv([make_env(env_index=f'{i}', **params) for i in range(1)])
    env = VecNormalize.load(
        load_path=path_convert(f'./{env_name}/models/last_vec_normalize.pkl'), 
        venv=env
    )
    env.training = False
    env.norm_reward = False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = path_convert(f'./{env_name}/models/last_rl_model.zip')
    model = PPO.load(model_path, env=env, device=device)

    # 使用模型进行测试
    obs = env.reset()
    dones = False # 默认是 False
    total_reward = 0

    while not dones:
        action, _state = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(action)
        total_reward += rewards
        
    env.close() # 需要把自动的 reset 关闭
    print(f'累积奖励为, {total_reward}.')