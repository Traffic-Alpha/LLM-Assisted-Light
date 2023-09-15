'''
@Author: WANG Maonan
@Date: 2023-09-05 15:26:11
@Description: 处理 State 的特征
@LastEditTime: 2023-09-15 20:15:02
'''
import numpy as np
from typing import List, Dict, Any
from tshub.utils.nested_dict_conversion import create_nested_defaultdict, defaultdict2dict

class OccupancyList:
    def __init__(self) -> None:
        self.elements = []

    def add_element(self, element) -> None:
        if isinstance(element, list):
            if all(isinstance(e, float) for e in element):
                self.elements.append(element)
            else:
                raise ValueError("列表中的元素必须是浮点数类型")
        else:
            raise TypeError("添加的元素必须是列表类型")

    def clear_elements(self) -> None:
        self.elements = []

    def calculate_average(self) -> float:
        """计算一段时间的平均 occupancy
        """
        arr = np.array(self.elements)
        averages = np.mean(arr, axis=0, dtype=np.float32)/100
        self.clear_elements() # 清空列表
        return averages






def calculate_queue_lengths(movement_ids:List[str], jam_length_meters:List[float], phase2movements:Dict[str, List[str]]) -> Dict[str, Dict[str, float]]:
    """计算每个相位的平均和最大排队长度

    Args:
        movement_ids (List[str]): 路口 movement 的顺序
            movement_ids = [
                "161701303#7.248_l", "161701303#7.248_r", "161701303#7.248_s",
                "29257863#2_l", "29257863#2_r", "29257863#2_s",
                "gsndj_n7_l", "gsndj_n7_r", "gsndj_n7_s",
                "gsndj_s4_l", "gsndj_s4_r", "gsndj_s4_s"
            ]
        jam_length_meters (List[float]): 每个 movement 对应的排队长度, 与上面的顺序相同
            jam_length_meters = [
                0.0, 0.0, 0.0,
                0.0, 0.0, 0.0,
                0.0, 0.0, 60.83249079171935,
                0.0, 0.0, 68.70503137164724
            ]
        phase2movements (Dict[str, List[str]]): 每个 phase 包含的 movement id
            phase2movements = {
                "0": [
                    "gsndj_s4--r",
                    "gsndj_s4--s",
                    "gsndj_n7--s",
                    "gsndj_n7--r"
                ],
                "1": [
                    "gsndj_s4--l",
                    "gsndj_n7--l"
                ],
                "2": [
                    "29257863#2--s",
                    "29257863#2--r",
                    "161701303#7.248--r",
                    "161701303#7.248--s"
                ],
                "3": [
                    "161701303#7.248--l",
                    "29257863#2--l"
                ]
            }
    Returns:
        Dict[str, Dict[str, float]]: 计算每一个 phase 的最大和平均排队长度
            {
                0: {'total_length': 0.0, 'count': 4, 'max_length': 0.0, 'average_length': 0.0}, 
                1: {'total_length': 0.0, 'count': 2, 'max_length': 0.0, 'average_length': 0.0}, 
                2: {'total_length': 0.0, 'count': 4, 'max_length': 0.0, 'average_length': 0.0}, 
                3: {'total_length': 0.0, 'count': 2, 'max_length': 0.0, 'average_length': 0.0}
            }
    """
    phase_queue_lengths = {}

    # 初始化每个 phase 的总排队长度和计数器
    for phase in phase2movements:
        phase_queue_lengths[phase] = {
            'total_length': 0.0,
            'count': 0,
            'max_length': 0.0,
            'average_length': 0.0
        }

    # 遍历每个 phase，累加每个 movement 的排队长度
    for phase, movements in phase2movements.items():
        for movement in movements:
            movement = '_'.join(movement.split('--'))
            index = movement_ids.index(movement)
            length = jam_length_meters[index]
            phase_queue_lengths[phase]['total_length'] += length
            phase_queue_lengths[phase]['count'] += 1
            phase_queue_lengths[phase]['max_length'] = max(phase_queue_lengths[phase]['max_length'], length)

    # 计算每个 phase 的平均排队长度
    for phase, data in phase_queue_lengths.items():
        if data['count'] > 0:
            data['average_length'] = data['total_length'] / data['count']

    return phase_queue_lengths


def predict_queue_length(queue_info:Dict[str, float], is_green:bool=False, num_samples = 10):
    leaving_rate_lambda = 4 # 离开率的参数 λ
    predict_queue_info = {} # 预测的排队长度
    for _id, _queue_length in queue_info.items():
        if _id == 'max_length':
            arrival_rate_lambda = 3 # 到达率的参数 λ
        elif _id == 'average_length':
            arrival_rate_lambda = 2 # 到达率的参数 λ
        else:
            continue
            
        if is_green:
            sample_sum = 0
            for _ in range(num_samples):
                sample_sum += np.random.poisson(arrival_rate_lambda) - np.random.poisson(leaving_rate_lambda)
            sample_sum *= 6 # 车辆数 --> 排队长度
            predicted_length = max(_queue_length + sample_sum / num_samples, 0)
            predict_queue_info[_id] = predicted_length
        else:
            sample_sum = 0
            for _ in range(num_samples):
                sample_sum += np.random.poisson(arrival_rate_lambda)
            sample_sum *= 6 # 车辆数 --> 排队长度
            predicted_length = max(_queue_length + sample_sum / num_samples, 0)
            predict_queue_info[_id] = predicted_length
    return predict_queue_info


def convert_state_to_static_information(input_data) -> Dict[str, Dict[str, Any]]:
    """将 state 输出为路网的静态信息

    Args:
        input_data: 单个 Traffic Light 的 state. 
        {
            'movement_directions': {'E2_r': 'r', 'E2_s': 's', ...},
            'movement_ids': ['E2_l', 'E2_r', 'E2_s', 'E4_l', ...],
            'phase2movements': {0: ['E2--s', 'E1--s'], 1: ['E1--l', 'E2--l'], ...},
            'movement_lane_numbers': {'-E2_r': 1, '-E2_s': 1, '-E2_l': 1, ...}
        }

    Returns:
        Dict[str, Dict[str, Any]]: 将其转换为路口的静态信息
        {
            "movement_infos": {
                "E2_l": {
                    "direction": "Left Turn",
                    "number_of_lanes": 1
                },
                "E2_s": {
                    "direction": "Through",
                    "number_of_lanes": 1
                },
                ...
            },
            "phase_infos": {
                "phase 0": {
                    "movements": ["E2_s", "E1_s"]
                },
                "phase 1": {
                    "movements": ["E1_l", "E2_l"]
                },
                ...
            }
        }
    """
    output_data = {
        "movement_infos": {},
        "phase_infos": {}
    }

    # 处理 movement_directions
    for movement_id, direction in input_data["movement_directions"].items():
        if direction == "l":
            direction_text = "Left Turn"
        elif direction == "s":
            direction_text = "Through"
        else:
            continue

        number_of_lanes = input_data["movement_lane_numbers"].get(movement_id, 0)

        output_data["movement_infos"][movement_id] = {
            "direction": direction_text,
            "number_of_lanes": number_of_lanes
        }

    # 处理 phase2movements
    for phase, movements in input_data["phase2movements"].items():
        phase_key = f"Phase {phase}"
        output_data["phase_infos"][phase_key] = {
            "movements": ["_".join(_movement.split('--')) for _movement in movements]
        }

    return output_data