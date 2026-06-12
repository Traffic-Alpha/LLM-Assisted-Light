'''
@Author: WANG Maonan
@Date: 2026-06-11 17:05:37
@Description: 
@LastEditTime: 2026-06-12 23:16:04
'''
"""Tool adapter exposing traffic-environment methods to the LLM agent."""

from typing import Any
from tshub.utils.format_dict import dict_to_str
from llm_tsc.tools_registry import tool


class TSCToolsAdapter:
    """Expose selected environment query methods as LLM-callable tools."""
    
    def __init__(self, env: Any):
        """Initialize adapter with environment"""
        self.env = env
    
    # ============================
    # Intersection Information
    # ============================
    
    @tool(
        name="get_intersection_layout",
        description="Get the structure and layout of the intersection. Input should be 'None'."
    )
    def get_intersection_layout(self, junction_id: str = "None") -> str:
        """
        Get intersection layout information
        
        Args:
            junction_id: Junction ID (usually 'None' for current intersection)
        
        Returns:
            Formatted description of intersection layout
        """
        try:
            layout = self.env.get_intersection_layout()
            return "The description of this intersection layout:\n" + dict_to_str(layout)
        except Exception as e:
            return f"Error getting intersection layout: {str(e)}"
    
    @tool(
        name="get_signal_phase_structure",
        description="Get the signal phase structure with Phase IDs and Movement IDs. Input should be 'None'."
    )
    def get_signal_phase_structure(self, junction_id: str = "None") -> str:
        """
        Get signal phase structure
        
        Args:
            junction_id: Junction ID (usually 'None')
        
        Returns:
            Formatted description of signal phases and movements
        """
        try:
            structure = self.env.get_signal_phase_structure()
            return "The description of Signal Phase Structure:\n" + dict_to_str(structure)
        except Exception as e:
            return f"Error getting signal phase structure: {str(e)}"
    
    # ============================
    # Current State Information
    # ============================
    @tool(
        name="get_current_occupancy",
        description="Get the current congestion status (occupancy) of each traffic movement at this moment."
    )
    def get_current_occupancy(self, junction_id: str = "None") -> str:
        """
        Get current occupancy information
        
        Args:
            junction_id: Junction ID
        
        Returns:
            Current occupancy for each movement
        """
        try:
            occupancy = self.env.get_current_occupancy()
            return "Current Occupancy:\n" + dict_to_str(occupancy)
        except Exception as e:
            return f"Error getting current occupancy: {str(e)}"
    
    @tool(
        name="get_previous_occupancy",
        description="Get the occupancy information from the previous time step."
    )
    def get_previous_occupancy(self, junction_id: str = "None") -> str:
        """
        Get previous occupancy (from last step)
        
        Args:
            junction_id: Junction ID
        
        Returns:
            Previous occupancy for each movement
        """
        try:
            occupancy = self.env.get_previous_occupancy()
            return "Previous Occupancy:\n" + dict_to_str(occupancy)
        except Exception as e:
            return f"Error getting previous occupancy: {str(e)}"
    
    # ============================
    # Actions and Decisions
    # ============================
    @tool(
        name="get_available_actions",
        description="Get all available signal phase actions you can take now."
    )
    def get_available_actions(self, junction_id: str = "None") -> str:
        """
        Get available actions
        
        Args:
            junction_id: Junction ID
        
        Returns:
            List of available phase actions
        """
        try:
            actions = self.env.get_available_actions()
            return "Available Actions:\n" + dict_to_str(actions)
        except Exception as e:
            return f"Error getting available actions: {str(e)}"
    
    @tool(
        name="get_traditional_decision",
        description="Get the phase recommended by the traditional max-pressure baseline."
    )
    def get_traditional_decision(self, junction_id: str = "None") -> str:
        """
        Get traditional decision from baseline algorithm
        
        Args:
            junction_id: Junction ID
        
        Returns:
            Recommended phase from traditional method
        """
        try:
            decision = self.env.get_traditional_decision()
            return "Traditional Decision:\n" + dict_to_str(decision)
        except Exception as e:
            return f"Error getting traditional decision: {str(e)}"
    
    # ============================
    # Edge Cases and Special Scenarios
    # ============================
    
    @tool(
        name="get_junction_situation",
        description="Get special situation information such as emergency vehicles, accident details, blocked movements, detector failures, and road access."
    )
    def get_junction_situation(self, junction_id: str = "None") -> str:
        """
        Get special situation (edge cases)
        
        Args:
            junction_id: Junction ID
        
        Returns:
            Information about special situations
        """
        try:
            situation = self.env.get_junction_situation()
            return "Junction Situation:\n" + dict_to_str(situation)
        except Exception as e:
            return f"Error getting junction situation: {str(e)}"
