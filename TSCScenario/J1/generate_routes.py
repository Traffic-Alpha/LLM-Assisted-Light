'''
@Author: WANG Maonan
@Date: 2023-09-18 16:25:24
@Description: 
@LastEditTime: 2023-09-18 16:31:56
'''
'''
@Author: WANG Maonan
@Date: 2023-09-01 13:45:26
@Description: 给场景生成路网
@LastEditTime: 2023-09-01 14:32:04
'''
from tshub.utils.get_abs_path import get_abs_path
from tshub.utils.init_log import set_logger
from tshub.sumo_tools.generate_routes import generate_route

# 初始化日志
current_file_path = get_abs_path(__file__)
set_logger(current_file_path('./'))

# 开启仿真 --> 指定 net 文件
sumo_net = current_file_path("./env/J1.net.xml")

# 指定要生成的路口 id 和探测器保存的位置
generate_route(
    sumo_net=sumo_net,
    interval=[1,1,1,1,1], 
    edge_flow_per_minute={
        'E3': [7, 7, 8, 8, 5],
        '-E4': [5, 5, 3, 9, 2],
        'E1': [10, 12, 15, 17, 10],
        '-E2': [10, 12, 15, 17, 10],
    }, # 每分钟每个 edge 有多少车
    edge_turndef={
        'E1__E2': [0.75, 0.75, 0.75, 0.75, 0.75],
        '-E2__-E1': [0.75, 0.75, 0.75, 0.75, 0.75],
    },
    veh_type={
        'rescue': {'probability':0.03},
        'ego': {'color':'26, 188, 156', 'probability':0.27},
        'background': {'color':'155, 89, 182', 'speed':15, 'probability':0.7},
    },
    output_trip=current_file_path('./testflow.trip.xml'),
    output_turndef=current_file_path('./testflow.turndefs.xml'),
    output_route=current_file_path('./testflow.rou.xml'),
    interpolate_flow=False,
    interpolate_turndef=False,
)