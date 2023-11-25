'''
@Author: WANG Maonan
@Date: 2023-09-08 17:39:53
@Description: Stable Baseline3 Utils
@LastEditTime: 2023-09-08 17:39:54
'''
import os
from typing import Callable
from stable_baselines3.common.callbacks import BaseCallback

class VecNormalizeCallback(BaseCallback):
    """保存环境标准化之后的值
    """
    def __init__(self, save_freq: int, save_path: str, name_prefix: str = "vec_normalize", verbose: int = 0):
        super(VecNormalizeCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            path = os.path.join(self.save_path, f"{self.name_prefix}_{self.num_timesteps}_steps.pkl")
            self.model.get_vec_normalize_env().save(path)
            if self.verbose > 1:
                print(f"Saving VecNormalize to {path}")
        return True


class BestVecNormalizeCallback(BaseCallback):
    """保存最优的环境
    """
    def __init__(self, save_path: str, verbose: int = 0):
        super(BestVecNormalizeCallback, self).__init__(verbose)
        self.save_path = save_path

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        path = os.path.join(self.save_path, f"best_vec_normalize.pkl")
        self.model.get_vec_normalize_env().save(path)
        if self.verbose > 1:
            print(f"Saving Best VecNormalize to {path}")
        return True


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func