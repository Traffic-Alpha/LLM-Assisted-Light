'''
@Author: WANG Maonan
@Date: 2023-11-23 22:56:22
@Description: 同时生成行人和车辆的 route
@LastEditTime: 2023-11-24 22:23:06
'''
from tshub.utils.get_abs_path import get_abs_path
from tshub.utils.init_log import set_logger
from tshub.sumo_tools.generate_routes import generate_route

# 初始化日志
current_file_path = get_abs_path(__file__)
set_logger(current_file_path('./'))

# 开启仿真 --> 指定 net 文件
sumo_net = current_file_path("./env/3way.net.xml")

# 指定要生成的路口 id 和探测器保存的位置
generate_route(
    sumo_net=sumo_net,
    interval=[1,1,1,1,1,1], # 共 6 min 
    edge_flow_per_minute={
        'E0': [19, 10, 19, 10, 19, 10],
        'E2': [10, 19, 10, 19, 10, 19],
        '-E1': [19, 10, 19, 10, 19, 10],
    }, # 每分钟每个 edge 有多少车
    edge_turndef={
        'E0__E1': [0.7, 0.7, 0.8, 0.7, 0.8, 0.7],
        '-E1__-E0': [0.7, 0.7, 0.8, 0.7, 0.8, 0.7],
    },
    veh_type={
        'ego': {'color':'26, 188, 156', 'length':4.5, 'probability':0.3},
        'background': {'color':'155, 89, 182', 'speed':15, 'length':4.5, 'probability':0.7},
    },
    walk_flow_per_minute={
        '-E1__-E0': [0, 50, 50, 0, 0, 0],
    },
    walkfactor=0.5,
    output_trip=current_file_path('./_testflow.trip.xml'),
    output_turndef=current_file_path('./_testflow.turndefs.xml'),
    person_trip_file=current_file_path('./_pedestrian.trip.xml'),
    output_route=current_file_path('./routes/vehicle.rou.xml'),
    output_person_file=current_file_path('./routes/pedestrian.rou.xml'),
    interpolate_flow=False,
    interpolate_turndef=False,
    interpolate_walkflow=False
)