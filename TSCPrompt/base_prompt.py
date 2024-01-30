'''
@Author: WANG Maonan
@Date: 2023-12-02 12:30:02
@Description: 基础规则的 Prompt
@LastEditTime: 2023-12-02 12:30:03
'''
TRAFFIC_RULES = """
1. Give priority to public transportation vehicles and Emergency Vehicles.
2. Traffic signals typically have multiple phases, and each phase is assigned a specific duration during which the associated movements are permitted or prohibited. 
3. If a movement is impassable, we do not need to give it the green light. For example, `M1` is impassable, and `M1` is in Phase `P1`, then you cannot set `P1` to the green light.
"""