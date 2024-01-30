'''
@Author: WANG Maonan
@Date: 2023-09-08 18:57:35
@Description: 使用训练好的 RL Agent 进行测试
Command: 
---> Choose Next Phase <---
@ Scenario-1
-> python eval_rl_agent.py --tls_action_type 'choose_next_phase' --env_name '3way' --phase_num 3
-> python eval_rl_agent.py --tls_action_type 'choose_next_phase' --env_name '4way' --phase_num 4
@ Scenario-2
-> python eval_rl_agent.py --tls_action_type 'choose_next_phase' --env_name '3way' --phase_num 3 --edge_block 'E1'
-> python eval_rl_agent.py --tls_action_type 'choose_next_phase' --env_name '4way' --phase_num 4 --edge_block 'E1'
@ Scenario-3
-> python eval_rl_agent.py --tls_action_type 'choose_next_phase' --env_name '3way' --phase_num 3 --detector_break 'E0--s'
-> python eval_rl_agent.py --tls_action_type 'choose_next_phase' --env_name '4way' --phase_num 4 --detector_break 'E2--s'

---> Next or Not <---
@ Scenario-1
-> python eval_rl_agent.py --tls_action_type 'next_or_not' --env_name '3way' --phase_num 3
-> python eval_rl_agent.py --tls_action_type 'next_or_not' --env_name '4way' --phase_num 4
@ Scenario-2
-> python eval_rl_agent.py --tls_action_type 'next_or_not' --env_name '3way' --phase_num 3 --edge_block 'E1'
-> python eval_rl_agent.py --tls_action_type 'next_or_not' --env_name '4way' --phase_num 4 --edge_block 'E1'
@ Scenario-3
-> python eval_rl_agent.py --tls_action_type 'next_or_not' --env_name '3way' --phase_num 3 --detector_break 'E0--s'
-> python eval_rl_agent.py --tls_action_type 'next_or_not' --env_name '4way' --phase_num 4 --detector_break 'E2--s'
@LastEditTime: 2023-12-02 23:01:23
'''
import torch
import argparse
from loguru import logger
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv
from rl_utils.make_tsc_env import make_env
from tshub.utils.get_abs_path import get_abs_path

path_convert = get_abs_path(__file__)
logger.remove()

if __name__ == '__main__':
    # Parser
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--env_name', type=str, default='4way', help='Environment name')
    parser.add_argument('--phase_num', type=int, default=4, help='Phase number')
    parser.add_argument('--tls_action_type', type=str, default='choose_next_phase', help='TLS Action Type')
    parser.add_argument('--edge_block', type=str, default=None, help='Edge block')
    parser.add_argument('--detector_break', type=str, default=None, help='Detector break')

    args = parser.parse_args()
    env_name = args.env_name # 3way, 4way
    phase_num = args.phase_num # 3, 4
    edge_block = args.edge_block
    detector_break = args.detector_break
    tls_action_type = args.tls_action_type

    # #########
    # Init Env
    # #########
    route_type = 'vehicle' # vehicle_pedestrian
    sumo_cfg = path_convert(f"../TSCScenario/{env_name}/env/{route_type}.sumocfg")
    net_file = path_convert(f"../TSCScenario/{env_name}/env/{env_name}.net.xml")
    log_path = path_convert(f'./')
    trip_info = path_convert(f'./{env_name}_{tls_action_type}.tripinfo.xml')
    params = {
        'tls_id':'J1',
        'num_seconds':1000,
        'tls_action_type': tls_action_type,
        'phase_num':phase_num,
        'sumo_cfg':sumo_cfg,
        'net_file':net_file,
        'use_gui':True,
        'log_file':log_path,
        'trip_info':trip_info,
        'copy_files': [trip_info] # 将对应的 trip 文件进行复制, 方便后续进行分析
    }
    env = SubprocVecEnv([make_env(env_index=f'{i}', **params) for i in range(1)])
    env = VecNormalize.load(
        load_path=path_convert(f'./result/{env_name}/{tls_action_type}/models/last_vec_normalize.pkl'), 
        venv=env
    )
    env.training = False
    env.norm_reward = False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = path_convert(f'./result/{env_name}/{tls_action_type}/models/last_rl_model.zip')
    model = PPO.load(model_path, env=env, device=device)

    # 使用模型进行测试
    obs = env.reset()
    dones = False # 默认是 False
    total_reward = 0
    sim_step = 0 # 初始的仿真时间
    while not dones:
        # 设置 edge block
        if edge_block is not None:
            if sim_step>50 and sim_step<100:
                env.env_method('set_edge_speed', edge_id=edge_block, speed=1)
            else:
                env.env_method('set_edge_speed', edge_id=edge_block, speed=13)
        
        # 设置 detector break
        if detector_break is not None:
            if sim_step>200 and sim_step<400:
                env.env_method('set_occ_missing', not_work_element=detector_break)
            else:
                env.env_method('set_occ_missing', not_work_element='')
        
        action, _state = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(action)
        total_reward += rewards
        sim_step = infos[0]['step_time'] # 取出第 0 个仿真
        
    env.close() # 需要把自动的 reset 关闭
    print(f'累积奖励为, {total_reward}.')