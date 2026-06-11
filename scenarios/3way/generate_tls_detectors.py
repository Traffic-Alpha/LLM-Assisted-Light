'''
@Author: WANG Maonan
@Date: 2021-11-18 12:56:54
@Description: 测试在指定路网和 tls_id, 生成探测器
@LastEditTime: 2023-11-24 20:35:48
'''
import libsumo as traci
from tshub.utils.get_abs_path import get_abs_path
from tshub.utils.init_log import set_logger
from tshub.sumo_tools.generate_detectors import generate_detector

# 初始化日志
current_file_path = get_abs_path(__file__)
set_logger(current_file_path('./'))

# 开启仿真 --> 指定 net 文件
netfile_path = current_file_path("./env/3way.net.xml")
traci.start(["sumo", "-n", netfile_path])

# 指定要生成的路口 id 和探测器保存的位置
g_detectors = generate_detector(traci)
g_detectors.generate_multiple_detectors(
    tls_list=['J1'], 
    result_folder=current_file_path("./detectors"),
    detectors_dict={
        'e2':{'detector_length':50}
    }
)