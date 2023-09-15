'''
@Author: WANG Maonan
@Date: 2023-09-04 20:51:49
@Description: traffic light control LLM Agent
@LastEditTime: 2023-09-15 20:53:32
'''
from typing import List
from loguru import logger

from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.agents.tools import Tool
from langchain.memory import ConversationTokenBufferMemory

from TSCAgent.tsc_agent_prompt import SYSTEM_MESSAGE_SUFFIX

class TSCAgent:
    def __init__(self, 
                 llm:ChatOpenAI, 
                 tools:List[Tool],
                 verbose:bool=True
                ) -> None:
        self.llm = llm # ChatGPT Model
        self.tools = [] # agent 可以使用的 tools
        for ins in tools:
            func = getattr(ins, 'inference')
            self.tools.append(
                Tool(name=func.name, description=func.description, func=func)
            )
        
        self.memory = ConversationTokenBufferMemory(llm=self.llm, max_token_limit=2048)
        self.agent = initialize_agent(
            tools=self.tools, # 这里是所有可以使用的工具
            llm=self.llm,
            agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            verbose=verbose,
            memory=self.memory,
            agent_kwargs={
                # 'system_message_prefix': SYSTEM_MESSAGE_PREFIX,
                'syetem_message_suffix': SYSTEM_MESSAGE_SUFFIX,
                # 'human_message': HUMAN_MESSAGE,
                # 'format_instructions': FORMAT_INSTRUCTIONS,
            },
            handle_parsing_errors="Check your output and make sure it conforms the format instructions!",
            max_iterations=12,
            early_stopping_method="generate",
        )
    
    def agent_run(self, sim_step:float):
        """_summary_

        Args:
            tls_id (str): _description_
            sim_step (float): _description_
        """
        logger.info(f"SIM: Decision at step {sim_step} is running:")
        # r = self.agent.run(
        #     f'Now you are a traffic signal light with id {tls_id}. Please analysis the efficient of the available actions you can make one by one.'
        # )
        # 需要返回上一步每个 movement 的排队长度
        r = self.agent.run(
            f'Please summary the current state of the intersections, and tell me which movment id is the most congestion.'
        )
        print(r)
        print('-'*10)