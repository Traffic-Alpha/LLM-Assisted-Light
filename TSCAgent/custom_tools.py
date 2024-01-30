'''
@Author: WANG Maonan
@Date: 2023-09-06 14:57:39
@Description: Agent Tools
@LastEditTime: 2024-01-06 20:26:51
'''
from typing import Any, Optional
from tshub.utils.format_dict import dict_to_str

def prompts(name, description):
    def decorator(func) -> Any:
        func.name = name
        func.description = description
        return func

    return decorator

# #############################
# Get Intersection Information
# #############################
class GetIntersectionLayout:
    def __init__(self, env) -> None:
        self.env = env
    
    @prompts(name="Get Intersection Layout",
            description="Useful when you want to know the structure of the intersection. This tool provides the description of the intersection layout. The input to this tool should always be the str, 'None'.")
    def inference(self, junction_id):
        intersection_layout = "The description of this intersection layout"
        intersection_layout += dict_to_str(self.env.get_intersection_layout())
        return intersection_layout


class GetSignalPhaseStructure:
    def __init__(self, env) -> None:
        self.env = env
    
    @prompts(name="Get Signal Phase Structure",
            description="Useful when you want to know the structure of the signal phase. This tool provides the description of the signal phase structure, including the `Phase ID` in this intersection and `Movement ID` in each signal phase. The input to this tool should always be the str, 'None'.")
    def inference(self, junction_id):
        signal_phase_structure = "The description of this Signal Phase Structure"
        signal_phase_structure += dict_to_str(self.env.get_signal_phase_structure())
        return signal_phase_structure


class GetCurrentOccupancy:
    def __init__(self, env: Any) -> None:
        self.env = env

    @prompts(name='Get Current Occupancy',
             description="""Useful when you want to get the congestion situation of each traffic movement at the **current** moment. The input to this tool should be a string, `junction_id`.""")
    def inference(self, *arg, **kwargs) -> str:
        current_occupancy = self.env.get_current_occupancy()
        current_occupancy_string = f"""Now you get the current occupancy for this intersection. At the current moment {self.env.tsc_env.sim_step}, the congestion situation of each movement is:\n{dict_to_str(current_occupancy)}
        \n,where `key` is the `movement id`, and `value` represents the proportion of the queue length on this `traffic movement` to the total length. Based on the information you have obtained so far, the structure of the intersection, the structure of the traffic phase, and the occupancy of each movement, please select an appropriate action from the available actions and give an explanation.
        """
        return current_occupancy_string


class GetPreviousOccupancy:
    def __init__(self, env: Any) -> None:
        self.env = env

    @prompts(name='Get Previous Occupancy',
             description="""Useful when you want to get the congestion situation of each traffic movement at the **previous** moment. The input to this tool should be a string, `junction_id`.""")
    def inference(self, *arg, **kwargs) -> str:
        previous_occupancy = self.env.get_previous_occupancy()
        previous_occupancy_string = f"""Now you get the previous occupancy for this intersection. At the previous moment {self.env.tsc_env.sim_step-5}, the congestion situation of each movement is:\n{dict_to_str(previous_occupancy)}
        \n,where `key` is the `movement id`, and `value` represents the proportion of the queue length on this `traffic movement` to the total length. Then please use the appropriate tool to obtain current occupancy.
        """
        return previous_occupancy_string


# ########################
# Get Traditional Decision
# ########################
class GetTraditionalDecision:
    def __init__(self, env: Any) -> None:
        self.env = env
        
    @prompts(name='Get Traditional Decision',
             description="""Useful when you want to obtain the decision made by traditional methods under the current environment. The result of traditional methods is applicable for minimizing queue length. However, in long-tail problems, such as the presence of an ambulance or sudden unavailability of certain movements, traditional methods are not the best solution. The input to this tool should always be the str, 'None'.""")
    def inference(self, junction_id) -> str:
        decision = self.env.get_rl_decision()
        traditional_decision = f'The decision provided by traditional methods is to set {int(decision)} as the green signal.'
        return traditional_decision


# ###############
# Choose Actions
# ###############
class GetAvailableActions:
    def __init__(self, env: Any) -> None:
        self.env = env

    @prompts(name='Get Available Actions',
             description="""Useful before you make the decisions, this tool let you know what are your available actions in this situation step. The input to this tool should always be the str, 'None'.""")
    def inference(self, junction_id) -> str:
        outputPrefix = 'You can ONLY use one of the following actions to control the traffic light to reduce the congestion: \n'
        available_actions = self.env.get_available_actions()
        for action in available_actions:
            outputPrefix += f'- Signal Phase {action}: Make Signal Phase {action} green light and the other Signal phases red lights.\n'
            
        return outputPrefix
    

# #################
# Scenario Analysis
# #################
class GetJunctionSituation:
    def __init__(self, env: Any) -> None:
        self.env = env
    
    @prompts(name="Get Junction Situation",
        description="Useful when you want to determine whether the environment is a long-tail problem in traffic signal control. When you want to judge whether there is an ambulance in the environment, check whether each movement is passable. The input to this tool should always be the str, 'None'.")
    def inference(self, junction_id) -> str:
        # 添加救护车的信息
        rescue_movement_ids = self.env.get_rescue_movement_ids()
        if rescue_movement_ids is None:
            junction_state = """There is currently no Emergency Vehicle at this intersection.\n"""
        else:
            junction_state = f"""Currently there are Emergency Vehicles on traffic movement {rescue_movement_ids}.\n"""
        
        # 添加是否有道路被 block 的信息
        movement_state = self.env.get_movement_state()
        junction_state += """The current traffic conditions for each movement are as follows, where 'True' indicates that vehicles can pass, and 'False' indicates that there is an accident ahead and vehicles cannot pass.\n"""
        junction_state += f'{movement_state}\n'

        # 添加是否有摄像头损坏的数据
        detector_state = self.env.get_detector_state()
        junction_state += f"""The current working status of the detector is as follows. Work means it is working normally, Not Work means it is not working normally. At this time, the occupancy rate is -1, which means that the camera is damaged and we cannot obtain the congestion level of the road. At this time, we hope that the phase of the signal light can be changed in sequence.\n{detector_state}"""
        return junction_state


class GetEmergencyVehicle:
    def __init__(self, env: Any) -> None:
        self.env = env

    @prompts(name='Get Emergency Vehicle',
             description="""Useful when you want to Check if there is an Emergency Vehicle on the specific traffic movement. The input to this tool should be a string, `junction_id`.""")
    def inference(self, junction_id) -> str:
        rescue_movement_ids = self.env.get_rescue_movement()
        if len(rescue_movement_ids) == 0:
            rescue_movement_string = """There is currently no Emergency Vehicle at this intersection."""
        else:
            rescue_movement_string = f"""Currently there are Emergency Vehicles on traffic movement {rescue_movement_ids}"""
        return rescue_movement_string