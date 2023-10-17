'''
@Author: WANG Maonan
@Date: 2023-09-04 20:51:49
@Description: traffic light control LLM Agent
@LastEditTime: 2023-10-16 00:01:59
'''
from typing import List
from loguru import logger

from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.agents.tools import Tool
from langchain.memory import ConversationSummaryMemory
from tshub.utils.get_abs_path import get_abs_path
from TSCAgent.callback_handler import create_file_callback

from TSCAgent.tsc_agent_prompt import (
    SYSTEM_MESSAGE_SUFFIX,
    SYSTEM_MESSAGE_PREFIX,
    HUMAN_MESSAGE,
    FORMAT_INSTRUCTIONS,
    TRAFFIC_RULES,
    DECISION_CAUTIONS,
    HANDLE_PARSING_ERROR
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
        # 找出接近的场景, 动作和解释
        llm_response = self.agent.run(
            f"""
            You, the 'traffic signal light', are now controlling the traffic signal in the junction with ID `{self.env.env.tls_id}`. You have already control for {sim_step} seconds.
            The decision you made LAST time step was `{last_step_action}`. Your explanation was `{last_step_explanation}`. 
            Please make decision for the traffic signal light. You have to describe the **Static State** and **Dynamic State** of the `traffic light`, including **Intersection Layout**, **Signal Phase Structure** and **Current Occupancy**. Then you need to determine whether the environment is a long-tail problem. If it's not a long-tail problem, you can refer to the Traditional Decision and provide an explanation based on the scene you observed. If it's a long-tail scenario, you need to analyze the possible actions and make a judgment on your own, and finally output your decision. 
            
            There are several rules you need to follow when you control the traffic lights:
            {TRAFFIC_RULES}

            Here are your attentions points:
            {DECISION_CAUTIONS}
            
            Let's take a deep breath and think step by step. Once you made a final decision, output it in the following format: \n
            ```
            Final Answer: 
                "decision":{{"traffic signal light decision, ONE of the available actions"}},
                "expalanations":{{"your explaination about your decision, described your suggestions to the Crossing Guard"}}
            ``` \n
            """,
            callbacks=[self.file_callback]
        )
        self.memory.clear()
        return llm_response