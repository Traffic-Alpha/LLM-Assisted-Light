'''
@Author: WANG Maonan
@Date: 2023-09-19 20:33:58
@Description: 将 SQLite 的场景转换为文字描述
+ 将 dict 转换为文字
+ 使用 llm 对文字进行概括
@LastEditTime: 2023-09-19 20:54:06
'''
from TSCEnvironment.wrapper_utils import convert_state_to_static_information
from utils.junction_similarity import get_int_info

def convert_description(raw_info):
    """将 SQLite 提取的信息转换为文字描述
    路口的拓扑结构是;
    路口的信号灯结构是;
    其中 - 有急救车
    每个 movement 的占有率是 xxx
    上一步的
    此时上次作出的动作是 xx, 理由是 xx

    Args:
        raw_info (_type_): _description_
    """
    sql_description = """"""
    raw_info = get_int_info(raw_info)
    intersection_layout, phase_structure, emergency_vehicle, current_occupancy, previous_occupancy = raw_info
    convert_state_to_static_information()
    print(1)
    return sql_description