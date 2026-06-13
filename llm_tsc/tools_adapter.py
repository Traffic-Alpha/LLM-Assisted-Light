'''
@Author: WANG Maonan
@Date: 2026-06-11 17:05:37
@Description: Tool adapter exposing the *dynamic* traffic-environment state to the
LLM agent. Static intersection info (layout, phase structure, available actions)
is injected into the system prompt once per episode, so it is intentionally not
exposed as a tool here.
@LastEditTime: 2026-06-13 23:16:04
'''
from typing import Any
from tshub.utils.format_dict import dict_to_str
from llm_tsc.tools_registry import tool


class TSCToolsAdapter:
    """Expose dynamic environment query methods as LLM-callable tools."""

    def __init__(self, env: Any):
        """Initialize adapter with environment"""
        self.env = env

    # ============================
    # Current State Information
    # ============================
    @tool(
        name="get_current_occupancy",
        description="Get the current congestion status (occupancy) of each traffic movement at this moment."
    )
    def get_current_occupancy(self) -> str:
        """Get current occupancy for each movement."""
        try:
            occupancy = self.env.get_current_occupancy()
            return "Current Occupancy:\n" + dict_to_str(occupancy)
        except Exception as e:
            return f"Error getting current occupancy: {str(e)}"

    @tool(
        name="get_previous_occupancy",
        description="Get the occupancy information from the previous decision step."
    )
    def get_previous_occupancy(self) -> str:
        """Get previous occupancy (from last decision step)."""
        try:
            occupancy = self.env.get_previous_occupancy()
            return "Previous Occupancy:\n" + dict_to_str(occupancy)
        except Exception as e:
            return f"Error getting previous occupancy: {str(e)}"

    # ============================
    # Decisions
    # ============================
    @tool(
        name="get_traditional_decision",
        description="Get the phase recommended by the traditional max-pressure baseline."
    )
    def get_traditional_decision(self) -> str:
        """Get traditional decision from the max-pressure baseline."""
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
    def get_junction_situation(self) -> str:
        """Get special situation (edge cases)."""
        try:
            situation = self.env.get_junction_situation()
            return "Junction Situation:\n" + dict_to_str(situation)
        except Exception as e:
            return f"Error getting junction situation: {str(e)}"
