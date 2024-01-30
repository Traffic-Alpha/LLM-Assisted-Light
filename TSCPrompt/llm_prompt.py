'''
@Author: WANG Maonan
@Date: 2023-12-02 12:07:17
@Description: Prompt for Simple LLM
@LastEditTime: 2024-01-05 20:24:14
'''
LLM_TSC_PROMPT = """As an AI controlling the traffic signal light at a busy intersection, make a decision based on the following information and rules: 
- The intersection's structure and signal phase: {movement_info}, {phase_info}
- The current phase is {current_phase}
- Congestion levels for each traffic movement: {occ}
- Presence of an emergency vehicle on a specific traffic movement: {rescue_state}
- Accessibility status of each movement: {movement_access}
- Functionality of each movement's detector (if not functioning, occupancy rate is -1): {detector_work}

You can choose only one of the following actions to control the traffic light to alleviate congestion: {available_actions}. Analyze the situation, consider your options, and make a reasoned decision. Once you have made your final decision, output it in the following format:
{format_instructions}

Suggestions:

- Make sure to take into account all the provided information before making a decision.
- Remember to consider the presence of emergency vehicles, which may require priority.
- If the Accessibility status of a movement is False, it means that the movement is currently blocked and cannot be passed, you don't need to give green light to this movements.
- Check the functionality of each detector. If a detector isn't working, that may influence your decision.
"""