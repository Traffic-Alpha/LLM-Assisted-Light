'''
@Author: WANG Maonan
@Date: 2023-11-01 23:46:03
@Description: Reward Curve for Single TSC
@LastEditTime: 2023-11-25 20:05:49
'''
from tshub.utils.plot_reward_curves import plot_reward_curve
from tshub.utils.get_abs_path import get_abs_path
path_convert = get_abs_path(__file__)


if __name__ == '__main__':
    log_files = [
        path_convert(f'./4way/log/{i}.monitor.csv')
        for i in range(6)
    ]
    plot_reward_curve(log_files)