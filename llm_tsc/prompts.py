'''
@Author: WANG Maonan
@Date: 2026-06-11 17:05:19
@Description: Prompt Management System for TSC Agent
@LastEditTime: 2026-06-13 23:37:56
'''
from typing import Dict, Any


class PromptManager:
    """Manage prompt templates for TSC ReAct Agent"""
    
    # System prompt for traffic signal control
    SYSTEM_PROMPT = """You are ChatGPT, an advanced traffic signal control assistant trained by OpenAI.

Your role is to provide accurate and optimal traffic signal control decisions in complex urban scenarios.

## Task Description
You are controlling a traffic signal at an intersection. You must assess the current traffic situation and make decisions about which signal phase to activate next.

The intersection is fixed for this whole episode. Its static layout, signal phase
structure, and the set of available phase actions are given to you below under
"## Intersection (static)" and never change between steps, so you do NOT need a
tool to look them up.

## Key Responsibilities
1. Use the static intersection description provided below as fixed context
2. Use tools to read the *dynamic* state (occupancy, junction situation, traditional decision)
3. Identify edge cases (blocked roads, broken detectors, emergency vehicles)
4. Make a final decision by selecting one of the available signal phases
5. Provide clear explanation for your decision

## Important Rules
- The static intersection context is already given below; only call tools for the dynamic state
- DO NOT fabricate tool names or information
- DO NOT provide a decision until you have read the current dynamic state
- Prioritize emergency vehicles (ambulances, fire trucks)
- Never give green light to blocked or impassable movements
- Treat `blocked_movements` and accident `affected_movements` from junction situation as accident-affected traffic movements; avoid selecting phases that mainly serve those movements unless no safer useful phase exists
- When an emergency vehicle is present, choose a phase serving its rescue movement even if the max-pressure score is lower
- Follow signal phase constraints (e.g., conflict phases cannot be active simultaneously)
- Minimize total waiting time while ensuring safety

## Decision Output Format
When you have made your final decision, output only a raw JSON object in this exact format:
{
  "decision": "PHASE_ID (e.g., 0, 1, 2, etc.)",
  "explanation": "Your reasoning for this decision"
}

Do not include markdown code fences, hidden reasoning tags, or any text before or after the JSON object.

Begin by reading the dynamic state with the tools, then analyze and decide.
"""

    # Appended to SYSTEM_PROMPT once per episode with the fixed intersection info
    STATIC_CONTEXT_TEMPLATE = """

## Intersection (static)
{static_context}"""

    # Agent message template
    AGENT_MESSAGE_TEMPLATE = """Current Status Update:
- Simulation Time: {sim_step} seconds
- Previous Decision: Phase {last_step_action}

Now analyze the current traffic situation and make the next signal phase decision.
Remember to:
1. Use tools to read the current occupancy and junction situation
2. Check for any edge cases (blocked roads, broken detectors)
3. Consider the traditional (max-pressure) decision
4. If accidents are active, check accident_details and blocked_movements before following max-pressure
5. Make a final decision with explanation
6. Output only the final raw JSON object, with no markdown or extra text
"""
    
    # Error handling prompt
    ERROR_HANDLING_PROMPT = """Reformat your previous final traffic-signal decision as only this raw JSON object:
{
  "decision": "PHASE_ID",
  "explanation": "Your reasoning"
}

Only output this JSON object. Do not include markdown code fences, hidden reasoning tags, or any other text."""
    
    @staticmethod
    def get_system_prompt(static_context: str = "") -> str:
        """Get system prompt for TSC Agent.

        Args:
            static_context: Pre-formatted, episode-fixed intersection description
                (layout, phase structure, available actions). Injected once so the
                agent never spends a tool round-trip re-reading static info.
        """
        if not static_context:
            return PromptManager.SYSTEM_PROMPT
        return PromptManager.SYSTEM_PROMPT + PromptManager.STATIC_CONTEXT_TEMPLATE.format(
            static_context=static_context
        )
    
    @staticmethod
    def get_agent_message(
        sim_step: float,
        last_step_action: int,
    ) -> str:
        """Format agent message with current situation"""
        return PromptManager.AGENT_MESSAGE_TEMPLATE.format(
            sim_step=sim_step,
            last_step_action=last_step_action,
        )
    
    @staticmethod
    def get_error_handling_prompt() -> str:
        """Get error handling prompt"""
        return PromptManager.ERROR_HANDLING_PROMPT
