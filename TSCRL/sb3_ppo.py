'''
@Author: WANG Maonan
@Date: 2023-09-08 15:48:26
@Description: PPO-based TSC control
@ Choose Next Phase
+ State Design: Last step occupancy for each movement
+ Action Design: Choose Next Phase 
+ Reward Design: Total Waiting Time

@ Next or Not
+ State Design: Last step occupancy for each movement + green phase id
+ Action Design: Next or Not
+ Reward Design: Total Waiting Time
@LastEditTime: 2023-11-28 14:04:07
'''
import os
import torch
from loguru import logger
from tshub.utils.get_abs_path import get_abs_path
from tshub.utils.init_log import set_logger

from rl_utils.make_tsc_env import make_env
from TSCRL.rl_utils.custom_model import CustomTSCModel
from rl_utils.sb3_utils import VecNormalizeCallback, linear_schedule

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback

path_convert = get_abs_path(__file__)
logger.remove()
set_logger(path_convert('./'), log_level="INFO")

if __name__ == '__main__':
    env_name = '3way'
    tls_action_type = 'choose_next_phase' # next_or_not, choose_next_phase
    phase_num = 2 if tls_action_type=='next_or_not' else 3
    log_path = path_convert(f'./{env_name}/log/')
    model_path = path_convert(f'./{env_name}/models/')
    tensorboard_path = path_convert(f'./{env_name}/tensorboard/')
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    if not os.path.exists(tensorboard_path):
        os.makedirs(tensorboard_path)
    
    # #########
    # Init Env
    # #########
    sumo_cfg = path_convert(f"../TSCScenario/{env_name}/env/vehicle.sumocfg")
    net_file = path_convert(f"../TSCScenario/{env_name}/env/{env_name}.net.xml")
    params = {
        'tls_id':'J1',
        'num_seconds':500,
        'tls_action_type': tls_action_type,
        'phase_num':phase_num, # 用于初始化 action space
        'sumo_cfg':sumo_cfg,
        'net_file':net_file,
        'use_gui':False,
        'log_file':log_path,
    }
    env = SubprocVecEnv([make_env(env_index=f'{i}', **params) for i in range(6)])
    env = VecNormalize(env, norm_obs=False, norm_reward=True)

    # #########
    # Callback
    # #########
    checkpoint_callback = CheckpointCallback(
        save_freq=10000, # 多少个 step, 需要根据与环境的交互来决定
        save_path=model_path,
    )
    vec_normalize_callback = VecNormalizeCallback(
        save_freq=10000,
        save_path=model_path,
    ) # 保存环境参数
    callback_list = CallbackList([checkpoint_callback, vec_normalize_callback])

    # #########
    # Training
    # #########
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    policy_kwargs = dict(
        features_extractor_class=CustomTSCModel,
        features_extractor_kwargs=dict(features_dim=16),
    )
    model = PPO(
                "MlpPolicy", 
                env, 
                batch_size=64,
                n_steps=300, n_epochs=5, # 每次间隔 n_epoch 去评估一次
                learning_rate=linear_schedule(1e-3),
                verbose=True, 
                policy_kwargs=policy_kwargs, 
                tensorboard_log=tensorboard_path, 
                device=device
            )
    model.learn(total_timesteps=1e5, tb_log_name='J1', callback=callback_list)
    
    # #################
    # 保存 model 和 env
    # #################
    env.save(f'{model_path}/last_vec_normalize.pkl')
    model.save(f'{model_path}/last_rl_model.zip')
    print('训练结束, 达到最大步数.')

    env.close()