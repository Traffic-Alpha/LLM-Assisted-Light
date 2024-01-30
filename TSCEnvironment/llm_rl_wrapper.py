'''
@Author: WANG Maonan
@Date: 2024-01-06 15:56:19
@Description: Env Wrapper for LLM RL
@LastEditTime: 2024-01-06 16:31:39
'''
import torch
from gymnasium.core import Env
from stable_baselines3 import PPO
from tshub.utils.get_abs_path import get_abs_path
from typing import List, Tuple, Any, Dict, SupportsFloat
from TSCEnvironment.base_tsc_wrapper import BaseTSCEnvWrapper

path_convert = get_abs_path(__file__)

class LLMRLTSCWrapper(BaseTSCEnvWrapper):
    def __init__(
            self, 
            env: Env, 
            tls_id: str, phase_num: int, max_states: int = 5, 
            copy_files: List[str] = []
        ) -> None:
        super().__init__(env, tls_id, phase_num, max_states, copy_files)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_path = path_convert(f'../TSCRL/result/{phase_num}way/choose_next_phase/models/last_rl_model.zip')
        self.model = PPO.load(model_path, device=device) # 导入的模型


    def reset(self, seed=1) -> Tuple[Any, Dict[str, Any]]:
        state, info =  super().reset(seed)
        self.rl_state = state.copy()
        return state, info
    
    def step(self, action: int) -> Tuple[Any, SupportsFloat, bool, bool, Dict[str, Any]]:
        state, rewards, truncated, dones, infos = super().step(action)
        self.rl_state = state.copy() # 用于 RL 算法
        return state, rewards, truncated, dones, infos
    
    # #######################
    # Tools: Get RL Decision
    # #######################
    def get_rl_decision(self):
        """获得基于强化学习的结果
        """
        action, _ = self.model.predict(self.rl_state, deterministic=True)
        return action