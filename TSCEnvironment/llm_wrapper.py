'''
@Author: WANG Maonan
@Date: 2023-12-01 21:02:38
@Description: The Wrapper for LLM. It also suit for FT and SOTL methods
- 包含对环境的描述
1. 路口的动态和静态信息
2. 每个 movement 的占有率
3. 每个 movement 是否有急救车
4. 每个 movement 是否可以通行
5. 每个 movement 对应的探测器是否可以正常工作
6. 所有的可以执行的动作
@LastEditTime: 2024-01-05 20:05:38
'''
from gymnasium.core import Env
from loguru import logger
from typing import Any, Dict, List, SupportsFloat, Tuple
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser

from TSCPrompt.llm_prompt import LLM_TSC_PROMPT
from TSCEnvironment.base_tsc_wrapper import BaseTSCEnvWrapper


class LLMTSCEnvWrapper(BaseTSCEnvWrapper):
    def __init__(
            self, 
            env: Env, 
            tls_id: str, phase_num: int, max_states: int = 5, 
            copy_files: List[str] = []
        ) -> None:
        super().__init__(env, tls_id, phase_num, max_states, copy_files)
        # Output 模板
        decision_schema = ResponseSchema(
            name="decision",
            description="Traffic signal light decision, ONE of the available actions"
        )
        explanation_schema = ResponseSchema(
            name="explanation",
            description="Your explanation about your decision, described your suggestions to the Crossing Guard"
        )
        self.output_parser = StructuredOutputParser.from_response_schemas([decision_schema, explanation_schema])

    def description_env(self) -> str:
        """Traffic Signal Control Prompt
        """
        format_instructions = self.output_parser.get_format_instructions() # 转换为提示词

        # Input 模板
        prompt_templete = ChatPromptTemplate.from_template(LLM_TSC_PROMPT)
        custom_message = prompt_templete.format_messages(
            movement_info=self.get_intersection_layout(),
            phase_info=self.get_signal_phase_structure(),
            current_phase=self.get_current_phase(),
            occ=self.get_current_occupancy(),
            rescue_state=self.get_rescue_movement_ids(),
            movement_access=self.get_movement_state(),
            detector_work=self.get_detector_state(),
            available_actions=self.get_available_actions(),
            format_instructions=format_instructions
        )

        logger.info(f'SIM: {custom_message[0].content}')
        return custom_message
    
    def reset(self, seed=1) -> Tuple[Any, Dict[str, Any]]:
        state, info =  super().reset(seed)
        return state, info
    
    def step(self, action: int) -> Tuple[Any, SupportsFloat, bool, bool, Dict[str, Any]]:
        return super().step(action)