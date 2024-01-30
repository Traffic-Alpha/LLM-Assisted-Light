'''
@Author: WANG Maonan
@Date: 2023-12-01 23:14:14
@Description: 直接把完整的场景信息给 LLM
@ Scenario-1
-> python llm.py --env_name '3way' --phase_num 3
-> python llm.py --env_name '4way' --phase_num 4
@ Scenario-2, Blocked
-> python llm.py --env_name '3way' --phase_num 3 --edge_block 'E1'
-> python llm.py --env_name '4way' --phase_num 4 --edge_block 'E1'
@ Scenario-3, Detector Break
-> python llm.py --env_name '3way' --phase_num 3 --detector_break 'E0--s'
-> python llm.py --env_name '4way' --phase_num 4 --detector_break 'E2--s'
@LastEditTime: 2024-01-16 16:51:46
'''
import argparse
import langchain
from loguru import logger
from langchain.chat_models import ChatOpenAI

from tshub.utils.get_abs_path import get_abs_path
from tshub.utils.init_log import set_logger

from TSCEnvironment.tsc_env import TSCEnvironment
from TSCEnvironment.llm_wrapper import LLMTSCEnvWrapper
from utils.readConfig import read_config

langchain.debug = False # 开启详细的显示
path_convert = get_abs_path(__file__)
set_logger(path_convert('./'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--env_name', type=str, default='4way', help='Environment name')
    parser.add_argument('--phase_num', type=int, default=4, help='Phase number')
    parser.add_argument('--edge_block', type=str, default=None, help='Edge block')
    parser.add_argument('--detector_break', type=str, default=None, help='Detector break')

    args = parser.parse_args()
    env_name = args.env_name # 3way, 4way
    phase_num = args.phase_num # 3, 4
    edge_block = args.edge_block # 是否 block 堵塞
    detector_break = args.detector_break # 检测器损坏, 导致 state 无法获得好的

    # Init LLM Model
    config = read_config()
    openai_proxy = config['OPENAI_PROXY']
    openai_api_key = config['OPENAI_API_KEY']
    openai_api_base = config['OPENAI_API_BASE']
    chat = ChatOpenAI(
        model=config['OPENAI_API_MODEL'], 
        temperature=0.0,
        openai_api_key=openai_api_key, 
        openai_proxy=openai_proxy,
        openai_api_base=openai_api_base,
    )

    # Init Scenario
    route_type = 'vehicle' # vehicle_pedestrian
    sumo_cfg = path_convert(f"./TSCScenario/{env_name}/env/{route_type}.sumocfg")
    net_file = path_convert(f"./TSCScenario/{env_name}/env/{env_name}.net.xml")
    log_path = path_convert(f'./')
    trip_info = path_convert(f'./{env_name}_LLM.tripinfo.xml')

    tsc_scenario = TSCEnvironment(
        sumo_cfg=sumo_cfg, 
        net_file=net_file,
        trip_info=trip_info,
        num_seconds=500,
        tls_id='J1', 
        tls_action_type='choose_next_phase',
        use_gui=True,
    ) # 初始化环境

    tsc_wrapper = LLMTSCEnvWrapper(
        env=tsc_scenario, 
        tls_id='J1',
        phase_num=phase_num
    )
    
    # Simulate with ENV
    dones = False
    tsc_wrapper.reset()
    action = 0 # 初始动作
    sim_step = 0
    while not dones:
        # 设置 edge block
        if edge_block is not None:
            if sim_step>50 and sim_step<100:
                tsc_wrapper.set_edge_speed(edge_id=edge_block, speed=1)
            else:
                tsc_wrapper.set_edge_speed(edge_id=edge_block, speed=13)
        
        # 设置 detector break
        if detector_break is not None:
            if sim_step>200 and sim_step<400:
                tsc_wrapper.set_occ_missing(not_work_element=detector_break)
            else:
                tsc_wrapper.set_occ_missing(not_work_element='')
                
        states, rewards, truncated, dones, infos = tsc_wrapper.step(action=action)
        tsc_message = tsc_wrapper.description_env() # 描述环境
        llm_decision = chat(tsc_message) # chat 作出决策
        logger.info(f'SIM: {llm_decision.content}')
        final_action = tsc_wrapper.output_parser.parse(llm_decision.content)
        try:
            action = int(final_action['decision'][-1])
            logger.info(f'SIM: {action}.')
            assert action in list(range(phase_num))
        except:
            action = 1 # 如果输出的动作不对

        sim_step = infos['step_time']
        
    tsc_wrapper.close()