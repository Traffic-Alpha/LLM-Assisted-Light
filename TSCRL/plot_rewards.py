'''
@Author: WANG Maonan
@Date: 2023-11-01 23:46:03
@Description: Reward Curve for Single TSC
@LastEditTime: 2024-01-30 09:58:40
'''
from tshub.utils.plot_reward_curves import plot_multi_reward_curves
from tshub.utils.get_abs_path import get_abs_path
path_convert = get_abs_path(__file__)


if __name__ == '__main__':
    env_name = '3way'
    dirs_and_labels = {
        'Choose Next Phase': [
            path_convert(f'./result/{env_name}/choose_next_phase/log/{i}.monitor.csv')
            for i in range(6)
        ],
        'Next or Not': [
            path_convert(f'./result/{env_name}/next_or_not/log/{i}.monitor.csv')
            for i in range(6)
        ]
    }

    plot_multi_reward_curves(dirs_and_labels)