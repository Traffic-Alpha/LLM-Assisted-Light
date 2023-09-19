'''
@Author: WANG Maonan
@Date: 2023-09-06 14:57:39
@Description: Agent Tools
@LastEditTime: 2023-09-19 14:44:58
'''
from typing import Any
from tshub.utils.format_dict import dict_to_str

def prompts(name, description):
    def decorator(func) -> Any:
        func.name = name
        func.description = description
        return func

    return decorator

class GetIntersectionLayout:
    def __init__(self, env) -> None:
        self.env = env
    
    @prompts(name="Get Intersection Layout",
            description="This tool provides the description of the intersection layout. The input to this tool should be `junction_id`.")
    def inference(self, junction_id):
        intersection_layout = "The description of this intersection layout"
        intersection_layout += dict_to_str(self.env.get_intersection_layout())
        return intersection_layout


class GetSignalPhaseStructure:
    def __init__(self, env) -> None:
        self.env = env
    
    @prompts(name="Get Signal Phase Structure",
            description="This tool provides the description of the signal phase structure, including the `Phase ID` in this intersection and `Movement ID` in each signal phase. The input to this tool should be `junction_id`.")
    def inference(self, junction_id):
        signal_phase_structure = "The description of this Signal Phase Structure"
        signal_phase_structure += dict_to_str(self.env.get_signal_phase_structure())
        return signal_phase_structure
    
    
class GetAvailableActions:
    def __init__(self, env: Any) -> None:
        self.env = env

    @prompts(name='Get Available Actions',
             description="""Useful before you make decisions, this tool let you know what are your available actions in this situation step. The input to this tool should be `junction_id`.""")
    def inference(self, junction_id) -> str:
        outputPrefix = 'You can ONLY use one of the following actions to control the traffic light to reduce the congestion: \n'
        available_actions = self.env.get_available_actions()
        for action in available_actions:
            outputPrefix += f'- Signal Phase {action}: Make Signal Phase {action} green light and the other Signal phases red lights.\n'
            
        outputPrefix += """\nNow you get all the available actions. To check which available action is more safety and efficiency, you should follow steps:
        Step 1: Check if there is an Emergency Vehicle near this junction. We hope the Emergency Vehicle can pass the intersection as soon as possible.
        Step 2: Get **Previous Occupancy** and **Current Occupancy** for this intersection.
        Step 3: Analyze the efficiency and safety (whether exists Emergency Vehicle) of different available actions.
        Remember to use the proper tools mentioned in the tool list ONCE a time. 
        NOTE: DONOT use the same tool repeatedly, for example `Get Available Actions`.
        """
        return outputPrefix
    

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



class GetCurrentOccupancy:
    def __init__(self, env: Any) -> None:
        self.env = env

    @prompts(name='Get Current Occupancy',
             description="""Useful when you want to get the congestion situation of each traffic movement at the **current** moment. The input to this tool should be a string, `junction_id`.""")
    def inference(self, *arg, **kwargs) -> str:
        current_occupancy = self.env.get_current_occupancy()
        current_occupancy_string = f"""Now you get the current occupancy for this intersection. At the current moment {self.env.tsc_env.sim_step}, the congestion situation of each movement is:\n{dict_to_str(current_occupancy)}
        \n,where `key` is the `movement id`, and `value` represents the proportion of the queue length on this `traffic movement` to the total length.
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
        \n,where `key` is the `movement id`, and `value` represents the proportion of the queue length on this `traffic movement` to the total length.
        """
        return previous_occupancy_string
    

class PredictQueueLength:
    def __init__(self, env: Any) -> None:
        self.env = env

    @prompts(name='Predict Queue Length',
            description="""Useful when you want to predict the queue length for the action you want to execute. The input is a str, representing the index of the signal phase that you decide to change to the green light. For example, the index for Phase 1 is 1.""")
    def inference(self, phase_id: int) -> str:
        predict_queue_length = self.env.predict_future_scene(phase_id)
        predict_state_str = f'If the Phase {phase_id} becomes green light, the queue length of each phase may become {dict_to_str(predict_queue_length)}.\n'
        predict_state_str += """You need to also check other available actions you mentioned above until all of them have beem checked."""
        return predict_state_str
    