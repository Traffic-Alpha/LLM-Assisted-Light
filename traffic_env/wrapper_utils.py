'''
@Author: WANG Maonan
@Date: 2023-09-05 15:26:11
@Description: 处理 State 的特征
@LastEditTime: 2023-12-02 17:35:36
'''
from typing import Dict, Any


def find_index(lst, element):
    try:
        return lst.index(element)
    except ValueError:
        return None


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
                    "movements": ["E2--s", "E1--s"]
                },
                "phase 1": {
                    "movements": ["E1--l", "E2--l"]
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
            "movements": movements
        }

    return output_data
