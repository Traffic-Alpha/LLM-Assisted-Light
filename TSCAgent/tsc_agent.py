'''
@Author: WANG Maonan
@Date: 2023-09-04 20:51:49
@Description: traffic light control LLM Agent
@LastEditTime: 2024-01-06 17:02:26
'''
from typing import List
from loguru import logger

from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.agents.tools import Tool
from langchain.memory import ConversationSummaryMemory
from tshub.utils.get_abs_path import get_abs_path
from TSCAgent.callback_handler import create_file_callback
from langchain.prompts import ChatPromptTemplate

from TSCPrompt.llm_rl_prompt import (
    SYSTEM_MESSAGE_PREFIX,
    SYSTEM_MESSAGE_SUFFIX,
    HUMAN_MESSAGE,
    HANDLE_PARSING_ERROR,
    AGENT_MESSAGE
)

class TSCAgent:
    def __init__(self, 
                 env,
                 llm:ChatOpenAI, 
                 tools:List[Tool],
                 verbose:bool=True
                ) -> None:
        self.env = env
        self.llm = llm # ChatGPT Model

        # callback
        path_convert = get_abs_path(__file__)
        self.file_callback = create_file_callback(path_convert('../agent.log'))

        self.tools = [] # agent 可以使用的 tools
        for ins in tools:
            func = getattr(ins, 'inference')
            self.tools.append(
                Tool(name=func.name, description=func.description, func=func)
            )
        
        self.memory = ConversationSummaryMemory(
            llm=self.llm,
        )
        self.agent = initialize_agent(
            tools=self.tools, # 这里是所有可以使用的工具
            llm=self.llm,
            agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            verbose=verbose,
            memory=self.memory,
            agent_kwargs={
                'system_message_prefix': SYSTEM_MESSAGE_PREFIX,
                'syetem_message_suffix': SYSTEM_MESSAGE_SUFFIX,
                'human_message': HUMAN_MESSAGE,
                # 'format_instructions': FORMAT_INSTRUCTIONS,
            },
            handle_parsing_errors=HANDLE_PARSING_ERROR,
            max_iterations=8,
            early_stopping_method="generate",
        )
    
    def agent_run(self, sim_step:float, last_step_action, last_step_explanation):
        """Agent Run
        """
        logger.info(f"SIM: Decision at step {sim_step} is running:")
        prompt_templete = ChatPromptTemplate.from_template(AGENT_MESSAGE)
        custom_message = prompt_templete.format_messages(
            sim_step=sim_step,
            last_step_action=last_step_action,
            last_step_explanation=last_step_explanation
        )
        # 找出接近的场景, 动作和解释
        llm_response = self.agent.run(
            custom_message,
            callbacks=[self.file_callback]
        )
        self.memory.clear()
        return llm_response