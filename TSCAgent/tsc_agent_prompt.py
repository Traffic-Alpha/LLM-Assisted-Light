'''
@Author: WANG Maonan
@Date: 2023-09-04 20:50:31
@Description: Traffic Signal Control Agent Prompt
@LastEditTime: 2023-09-15 17:24:46
'''
TSC_INSTRUCTIONS = """Now suppose you are an expert in traffic signal control, your goal is to reduce congestion at the intersection. The traffic signal at this intersection has **{phase_num}** phases. In the current environment, the average queue length and maximum queue length for each phase are as follows, measured in meters:

```json
{phase_info}
```

Currently, Phase {phase_id} is a green light. You have two actions to choose from:

```
- keep_current_phase: The current Phase {phase_id} keep the green light for another 5s, and the other phases are red lights.
- change_to_next_phase: The next Phase {next_phase_id} changes to the green light and keep it for 5s, and the other phases are red lights.
```

Here are your attentions points:
- Your goal is to reduce congestion at the intersection;
- When the phase of a signal light is **green**, it typically leads to a decrease in the queue length for that phase. The queue length reduces per second is 7m/s;
- To the opposite, the queue length for the **red** phase typically tends to increase. The queue length growth per second is 7m/s

Please make decision for the traffic light. Let's think step by step. 

- Analyze the impact of executing `keep_current_phase`` on the queuing of each phase
- Analyze the impact of executing `change_to_next_phase`` on the queue of each phase
- Based on the above analysis, calculate the congestion situation at the intersection under the two actions 
"""

SYSTEM_MESSAGE_SUFFIX = """
The driving task usually invovles many steps. You can break this task down into subtasks and complete them one by one. 
There is no rush to give a final answer unless you are confident that the answer is correct.
Answer the following questions as best you can. Begin! 

Donot use multiple tools at one time.
Take a deep breath and work on this problem step-by-step.
Reminder you MUST use the EXACT characters `Final Answer` when responding the final answer of the original input question.
"""

TSC_SUMMARY = """
{decision_result}
{format_instructions}
"""