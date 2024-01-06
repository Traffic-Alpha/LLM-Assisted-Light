'''
@Author: WANG Maonan
@Date: 2023-12-01 22:59:52
@Description: 结合 RL+LLM 的方式来进行决策, 调用工具
运行 script 3way_agent.log 可以开始记录日志, 将所有输出的终端上面的内容都保存下来
运行 exit 可以推出日志的记录

@3way
-> python llm_rl.py --env_name '3way' --phase_num 3 --edge_block 'E1' --detector_break 'E0--s'

@4way
-> python llm_rl.py --env_name '4way' --phase_num 4 --edge_block 'E1' --detector_break 'E2--s'
@LastEditTime: 2024-01-06 20:45:24
'''
import argparse
import langchain
import numpy as np
from langchain.chat_models import ChatOpenAI

from tshub.utils.get_abs_path import get_abs_path
from tshub.utils.init_log import set_logger

from TSCEnvironment.tsc_env import TSCEnvironment
from TSCEnvironment.llm_rl_wrapper import LLMRLTSCWrapper
from TSCAgent.tsc_agent import TSCAgent
from TSCAgent.output_parse import OutputParse
from TSCAgent.custom_tools import (
    GetAvailableActions, 
    GetCurrentOccupancy,
    GetPreviousOccupancy,
    GetIntersectionLayout,
    GetSignalPhaseStructure,
    GetTraditionalDecision,
    GetJunctionSituation
)
from utils.readConfig import read_config

langchain.debug = False # 开启详细的显示
path_convert = get_abs_path(__file__)
set_logger(path_convert('./'))

if __name__ == '__main__':
    # Init Parser
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

    # Init Chat
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

    # Init scenario
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

    tsc_wrapper = LLMRLTSCWrapper(
        env=tsc_scenario, 
        tls_id='J1',
        phase_num=phase_num # 相位数量
    )

    # Init Agent
    o_parse = OutputParse(env=None, llm=chat)
    tools = [
        GetIntersectionLayout(env=tsc_wrapper),
        GetSignalPhaseStructure(env=tsc_wrapper),
        GetCurrentOccupancy(env=tsc_wrapper),
        GetPreviousOccupancy(env=tsc_wrapper),
        GetTraditionalDecision(env=tsc_wrapper),
        GetAvailableActions(env=tsc_wrapper),
        GetJunctionSituation(env=tsc_wrapper),
    ]
    tsc_agent = TSCAgent(env=tsc_wrapper, llm=chat, tools=tools, verbose=True)

    # Start Simulation
    dones = False
    sim_step = 0
    phase_id = 0 # 当前动作 id
    last_step_explanation = "" # 作出决策的原因
    states = tsc_wrapper.reset()
    while not dones:
        if (sim_step > 150) and (sim_step < 300):
            # 设置 edge block
            if edge_block is not None:
                if sim_step>170 and sim_step<200:
                    tsc_wrapper.set_edge_speed(edge_id=edge_block, speed=1)
                else:
                    tsc_wrapper.set_edge_speed(edge_id=edge_block, speed=13)
            
            # 设置 detector break
            if detector_break is not None:
                if sim_step>220 and sim_step<250:
                    tsc_wrapper.set_occ_missing(not_work_element=detector_break)
                else:
                    tsc_wrapper.set_occ_missing(not_work_element='')
            
            agent_response = tsc_agent.agent_run(
                sim_step=sim_step, 
                last_step_action=phase_id, # 上一步的动作
                last_step_explanation=last_step_explanation # 上一步的解释
            ) # 让 LLM Agent 来回答问题
            print(f'Parser Output, {agent_response}')
            agent_action = o_parse.parser_output(agent_response)
            phase_id = agent_action['phase_id']
            last_step_explanation = agent_action['explanation']
        elif sim_step < 150:
            phase_id = np.random.randint(phase_num) # 随机选择相位
        else:
            action = tsc_wrapper.get_rl_decision() # 获得强化学习的动作
            phase_id = int(action)

        state, rewards, truncated, dones, infos = tsc_wrapper.step(action=phase_id)
        sim_step = infos['step_time']
        print(f'---\nSim Time, {sim_step}\n---')
    
    tsc_wrapper.close()
