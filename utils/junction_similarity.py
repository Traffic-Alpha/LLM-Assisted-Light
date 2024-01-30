'''
@Author: WANG Maonan
@Date: 2023-09-19 17:52:38
@Description: 计算两个路口的相似度
+ 1. 计算静态相似度 (欧式距离)
+ 2. 计算动态相似度 (余弦相似度)
@LastEditTime: 2023-09-19 20:35:27
'''
from typing import Tuple, Dict, List, Any, Literal
from utils.euclidean_distance import euclidean_distance

def find_min_indices(lst, n):
    # 使用enumerate函数获取每个元素的索引和值
    indices_and_values = list(enumerate(lst))
    
    # 根据值进行排序
    sorted_indices_and_values = sorted(indices_and_values, key=lambda x: x[1])
    
    # 获取最小的n个值的索引
    min_indices = [index for index, _ in sorted_indices_and_values[:n]]
    
    return min_indices

def get_int_info(raw_int) -> Tuple[Any,Any,Any,Any,Any,Any]:
    intersection_layout = eval(raw_int[2])
    phase_structure = eval(raw_int[3])
    emergency_vehicle = eval(raw_int[4])
    current_occupancy = eval(raw_int[5])
    previous_occupancy = eval(raw_int[6])
    return (intersection_layout, phase_structure, emergency_vehicle, current_occupancy, previous_occupancy)


def get_statistic_info(intersection_layout:Dict[str, Dict[str, str]], phase_structure:Dict[str, Dict[str, List[str]]]) -> List[float]:
    """提取路口静态信息矩阵, 每一个 phase 控制的总的车道数

    Args:
        intersection_layout (Dict[str, Dict[str, str]]): 包含每一个 movement 的信息
            {
                '-E2_s': {'direction': 'Through', 'number_of_lanes': 2}, 
                '-E2_l': {'direction': 'Left Turn', 'number_of_lanes': 1}, 
                '-E4_s': {'direction': 'Through', 'number_of_lanes': 2}, 
                '-E4_l': {'direction': 'Left Turn', 'number_of_lanes': 1}, 
                'E1_s': {'direction': 'Through', 'number_of_lanes': 2}, 
                'E1_l': {'direction': 'Left Turn', 'number_of_lanes': 1}, 
                'E3_s': {'direction': 'Through', 'number_of_lanes': 2}, 
                'E3_l': {'direction': 'Left Turn', 'number_of_lanes': 1}
            }
        phase_structure (Dict[str, Dict[str, list[str]]]): 包含每一个 phase 的信息
            {
                'Phase 0': {'movements': ['-E2_s', 'E1_s']}, 
                'Phase 1': {'movements': ['E1_l', '-E2_l']}, 
                'Phase 2': {'movements': ['E3_s', '-E4_s']}, 
                'Phase 3': {'movements': ['-E4_l', 'E3_l']}
            }

    Returns:
        List[float]: 每个 phase 控制的总的车道数, 上面的例子中结果是 [4,2,4,2]
    """
    lane_counts = []
    for phase in phase_structure.values():
        lanes = 0
        for movement in phase['movements']:
            direction = intersection_layout[movement]['direction']
            number_of_lanes = intersection_layout[movement]['number_of_lanes']
            lanes += number_of_lanes
        lane_counts.append(lanes)

    return lane_counts


def get_occupancy_info(movement_occupancy:Dict[str, Dict[str, str]], phase_structure:Dict[str, Dict[str, List[str]]]) -> List[float]:
    """提取路口动态信息矩阵, 每一个 phase 的 occupancy

    Args:
        movement_occupancy (Dict[str, Dict[str, str]]): 包含每一个 movement 的信息
            {
                '-E2_l': '6.99300691485405%', 
                '-E2_s': '27.9720276594162%', 
                '-E4_l': '0.0%', 
                '-E4_s': '0.0%', 
                'E1_l': '6.99300691485405%', 
                'E1_s': '40.81219136714935%', 
                'E3_l': '20.97902148962021%', 
                'E3_s': '0.0%'
            }
        phase_structure (Dict[str, Dict[str, list[str]]]): 包含每一个 phase 的信息
            {
                'Phase 0': {'movements': ['-E2_s', 'E1_s']}, 
                'Phase 1': {'movements': ['E1_l', '-E2_l']}, 
                'Phase 2': {'movements': ['E3_s', '-E4_s']}, 
                'Phase 3': {'movements': ['-E4_l', 'E3_l']}
            }

    Returns:
        List[float]: 每个 phase 控制的总的车道数, 上面的例子中结果是 [4,2,4,2]
    """
    def convert_percentage_to_float(percentage_str):
        # 去除字符串中的百分号字符 '%'
        percentage_str = percentage_str.replace('%', '')
        
        # 将字符串转换为浮点数类型，并除以 100 得到小数形式
        float_value = float(percentage_str) / 100
        
        return float_value

    phase_occupancy = []
    for phase in phase_structure.values():
        occupancy = 0
        for movement in phase['movements']:
            occupancy += convert_percentage_to_float(movement_occupancy[movement])
            
        phase_occupancy.append(occupancy)

    return phase_occupancy


def calculate_similarity(int_anchor):
    """计算 anchor INT 和 compared INT 之间的相似度
    + 数值越小, 越接近
    + 包含静态相似度和动态相似度

    Args:
        int_anchor (_type_): _description_

    Returns:
        _type_: _description_
    """
    int_anchor = get_int_info(int_anchor)
    anchor_intersection_layout, anchor_phase_structure, anchor_emergency_vehicle, anchor_current_occupancy, anchor_previous_occupancy = int_anchor
    anchor_statistic_info = get_statistic_info(anchor_intersection_layout, anchor_phase_structure) # 静态信息
    anchor_current_occupancy_info = get_occupancy_info(anchor_current_occupancy, anchor_phase_structure)
    anchor_previous_occupancy_info = get_occupancy_info(anchor_previous_occupancy, anchor_phase_structure)

    def similarity_score(int_compare) -> float:
        int_compare = get_int_info(int_compare)
        compare_intersection_layout, compare_phase_structure, compare_emergency_vehicle, compare_current_occupancy, compare_previous_occupancy = int_compare
        compare_statistic_info = get_statistic_info(compare_intersection_layout, compare_phase_structure) # 静态信息
        compare_current_occupancy_info = get_occupancy_info(compare_current_occupancy, compare_phase_structure)
        compare_previous_occupancy_info = get_occupancy_info(compare_previous_occupancy, compare_phase_structure)

        static_similarity = euclidean_distance(anchor_statistic_info, compare_statistic_info)  
        dynamic_similarity = calculate_dynamic_similarity(
            anchor_emergency_vehicle, anchor_current_occupancy_info, anchor_previous_occupancy_info,
            compare_emergency_vehicle, compare_current_occupancy_info, compare_previous_occupancy_info
        )
    
        return (static_similarity + dynamic_similarity) / 2
    return similarity_score


def calculate_dynamic_similarity(anchor_emergency_vehicle, anchor_current_occupancy, anchor_previous_occupancy, compare_emergency_vehicle, compare_current_occupancy, compare_previous_occupancy) -> float:
    """计算动态的相似度, 动态相似度由三个部分组成
    + 是否包含 emergency vehicle
    + current_occupancy
    + previous_occupancy
    """
    emergency_vehicle_score = check_emergency(anchor_emergency_vehicle, compare_emergency_vehicle)
    current_occupancy_score = euclidean_distance(anchor_current_occupancy, compare_current_occupancy)
    previous_occupancy_score = euclidean_distance(anchor_previous_occupancy, compare_previous_occupancy)
    
    return (emergency_vehicle_score+current_occupancy_score+previous_occupancy_score)/3

def check_emergency(a, b) -> Literal[0, 1]:
    """比较场景中是否都有 emergency:
    + 如果都有或没有 emergency, 则返回 0 (相似)
    + 如果一个有一个没有, 则返回 1 (不相似)
    """
    if (not a and not b) or (a and b):
        return 0
    else:
        return 1