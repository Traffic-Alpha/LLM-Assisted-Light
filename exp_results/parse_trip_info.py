'''
@Author: WANG Maonan
@Date: 2023-12-01 22:36:26
@Description: 解析 TripInfo 文件
@LastEditTime: 2023-12-01 22:48:53
'''
from tshub.utils.parse_trip_info import TripInfoStats
from tshub.utils.get_abs_path import get_abs_path

path_convert = get_abs_path(__file__)

if __name__ == '__main__':
    env_names = ['3way', '4way']
    method_names = ['choose_next_phase', 'next_or_not', 'FT', 'SOTL']
    for env_name in env_names:
        for method_name in method_names:
            stats = TripInfoStats(path_convert(f'./{env_name}/{method_name}.tripinfo.xml'))
            stats.to_csv(path_convert(f'./{env_name}/{method_name}.csv'))