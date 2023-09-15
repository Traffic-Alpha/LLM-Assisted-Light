'''
@Author: WANG Maonan
@Date: 2023-09-04 20:46:09
@Description: 基于 LLM 的 Traffic Light Control
1. 会有数据库, 我们会搜索最相似的场景 (如何定义场景的相似程度), 然后可以存储在 memory 里面, 或者放在 query 里面
2. 不同的 action 检查
    - getAvailableActions, 获得当前所有的动作
    - get queue length of all phases
    - get emergency vehicle
    - check possible queue length of all actions
    - 执行每个动作后面的相位是什么
    - 如果执行这个动作, 对未来场景的预测
    - 当前场景总的排队长度
    - 考虑 bus 或是救护车
3. 提取场景的数据, 不同的 phase 由几个 movement 组成, 不同 movement 在此时的排队情况, 这里需要存储数据
4. 这里我们先做出单路口的 LLM 的控制
@LastEditTime: 2023-09-15 17:29:45
'''
import langchain
import numpy as np
from langchain.chat_models import ChatOpenAI

from loguru import logger
from tshub.utils.format_dict import dict_to_str
from tshub.utils.get_abs_path import get_abs_path
from tshub.utils.init_log import set_logger

from TSCEnvironment.tsc_env import TSCEnvironment
from TSCEnvironment.tsc_env_wrapper import TSCEnvWrapper
from TSCAgent.tsc_agent import TSCAgent
from TSCAgent.custom_tools import GetAvailableActions, GetCurrentOccupancy
from utils.readConfig import read_config

langchain.debug = False # 开启详细的显示
path_convert = get_abs_path(__file__)
set_logger(path_convert('./'))

if __name__ == '__main__':
    # Init Chat
    config = read_config()
    openai_proxy = config['OPENAI_PROXY']
    openai_api_key = config['OPENAI_API_KEY']
    chat = ChatOpenAI(
        model='gpt-3.5-turbo-16k', temperature=0.0,
        openai_api_key=openai_api_key, 
        openai_proxy=openai_proxy
    )

    # Init scenario
    sumo_cfg = path_convert("./TSCScenario/J1/env/J1.sumocfg")
    tsc_scenario = TSCEnvironment(
        sumo_cfg=sumo_cfg, 
        num_seconds=1200,
        tls_id='J4', 
        tls_action_type='choose_next_phase',
        use_gui=True
    )
    tsc_wrapper = TSCEnvWrapper(tsc_scenario)

    # Init Agent
    tools = [
        GetAvailableActions(env=tsc_wrapper),
        GetCurrentOccupancy(env=tsc_wrapper), # 查看当前时刻的拥堵情况
    ]
    tsc_agent = TSCAgent(llm=chat, tools=tools, verbose=True)

    # Start Simulation
    dones = False
    sim_step = 0
    states = tsc_wrapper.reset()
    while not dones:
        action = np.random.randint(4)
        tsc_agent.agent_run(sim_step=sim_step)
        states, dones, infos = tsc_wrapper.step(action=action)
        sim_step = infos['step_time']
