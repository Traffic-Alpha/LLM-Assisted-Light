"""ReAct loop: ask the model, execute tools, and return a signal decision."""
import json
from copy import deepcopy
from time import perf_counter
from typing import Any, Callable, Dict, List, Optional

from loguru import logger
from openai import OpenAI

from la_light.llm.protocol import ConversationMessage, build_messages, build_tool_specs, execute_tool


class SimplifiedReActAgent:
    def __init__(self, llm_client: OpenAI, tools_dict: Dict[str, Callable],
                 system_prompt: str = "", max_iterations: int = 8,
                 model: str = "gpt-4", temperature: float = 0.0,
                 verbose: bool = True, max_tool_rounds: int = 2,
                 extra_body: Optional[Dict[str, Any]] = None):
        self.llm = llm_client
        self.tools = tools_dict
        self.system_prompt = system_prompt
        self.max_iterations = max_iterations
        self.model = model
        self.temperature = temperature
        self.verbose = verbose
        self.max_tool_rounds = max_tool_rounds
        self.extra_body = deepcopy(extra_body or {})
        self.conversation_history: List[ConversationMessage] = []
        self.tool_specs = build_tool_specs(self.tools)
        self.last_run_stats: Dict[str, Any] = {"api_calls": [], "tool_calls": []}

    def run(self, user_query: str, reset_history: bool = False, allow_tools: bool = True) -> str:
        self.last_run_stats = {"api_calls": [], "tool_calls": []}
        if reset_history:
            self.conversation_history = []
        self.conversation_history.append(
            ConversationMessage(role="user", content=user_query)
        )
        if self.verbose:
            logger.info(f"🚀 Starting ReAct loop (max {self.max_iterations} iterations)")
        tool_rounds = 0
        for iteration in range(self.max_iterations):
            if self.verbose:
                logger.info(f"\n📍 Iteration {iteration + 1}/{self.max_iterations}")
            messages = build_messages(self.conversation_history, self.system_prompt)
            api_started = perf_counter()
            try:
                tools_allowed = (allow_tools and tool_rounds < self.max_tool_rounds
                                 and iteration < self.max_iterations - 1)
                request = {"model": self.model, "messages": messages, "temperature": self.temperature}
                if self.extra_body:
                    request['extra_body'] = deepcopy(self.extra_body)
                if tools_allowed:
                    request.update(tools=self.tool_specs, tool_choice="auto")
                else:
                    messages.append({"role": "user", "content": "No further tool queries. Return the final phase JSON using the supplied observations."})
                response = self.llm.chat.completions.create(**request)
            except Exception as e:
                self.last_run_stats["api_calls"].append({
                    "latency_ms": (perf_counter() - api_started) * 1000,
                    "error": str(e),
                })
                logger.error(f"LLM API error: {str(e)}")
                return f"Error calling LLM: {str(e)}"
            stop_reason = response.choices[0].finish_reason
            self.last_run_stats["api_calls"].append({
                "latency_ms": (perf_counter() - api_started) * 1000,
                "finish_reason": stop_reason,
                "reasoning_content_present": bool(getattr(response.choices[0].message, "reasoning_content", None)),
            })
            if stop_reason == "tool_calls":
                if not tools_allowed:
                    return "Tool budget exhausted without a final phase decision"
                tool_rounds += 1
                assistant_message = response.choices[0].message
                tool_calls = [
                    {
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments,
                        },
                    }
                    for tool_call in assistant_message.tool_calls
                ] if getattr(assistant_message, "tool_calls", None) else []
                self.conversation_history.append(
                    ConversationMessage(
                        role="assistant",
                        content=assistant_message.content or "",
                        tool_calls=tool_calls,
                        reasoning_content=getattr(assistant_message, 'reasoning_content', None),
                    )
                )
                if self.verbose:
                    if assistant_message.content:
                        logger.info(f"💭 LLM: {assistant_message.content[:100]}...")
                if hasattr(assistant_message, 'tool_calls') and assistant_message.tool_calls:
                    for tool_call in assistant_message.tool_calls:
                        tool_name = tool_call.function.name
                        tool_call_id = tool_call.id
                        if self.verbose:
                            logger.info(f"🔧 Calling tool: {tool_name}")
                        try:
                            tool_args = json.loads(tool_call.function.arguments)
                        except json.JSONDecodeError:
                            tool_args = {}
                            logger.warning(f"Failed to parse tool arguments for {tool_name}")
                        tool_started = perf_counter()
                        observation = execute_tool(self.tools, tool_name, tool_args, self.verbose)
                        try:
                            tool_result = json.loads(observation)
                        except (ValueError, TypeError):
                            tool_result = observation
                        self.last_run_stats["tool_calls"].append({
                            "name": tool_name, "arguments": tool_args,
                            "latency_ms": (perf_counter() - tool_started) * 1000,
                            "success": not (isinstance(tool_result, dict) and "error" in tool_result),
                            "result": tool_result,
                        })
                        if self.verbose:
                            logger.info(f"📊 Observation: {observation[:100]}...")
                        self.conversation_history.append(
                            ConversationMessage(
                                role="tool",
                                content=observation,
                                tool_call_id=tool_call_id,
                                tool_name=tool_name,
                            )
                        )
                else:
                    logger.warning("No tool calls found in response despite finish_reason='tool_calls'")
                    break
            elif stop_reason == "end_turn" or stop_reason == "stop":
                final_answer = response.choices[0].message.content
                if self.verbose:
                    logger.info(f"✅ Final Answer: {(final_answer or '')[:200]}...")
                self.conversation_history.append(
                    ConversationMessage(role="assistant", content=final_answer or "",
                                        reasoning_content=getattr(response.choices[0].message, 'reasoning_content', None))
                )
                return final_answer or "No response"
            else:
                logger.warning(f"Unexpected stop reason: {stop_reason}")
                final_answer = response.choices[0].message.content
                if final_answer:
                    self.conversation_history.append(
                        ConversationMessage(role="assistant", content=final_answer,
                                            reasoning_content=getattr(response.choices[0].message, 'reasoning_content', None))
                    )
                    return final_answer
                break
        logger.warning("Max iterations reached without final answer")
        return "Max iterations reached without reaching a final decision"

    def get_history(self) -> List[ConversationMessage]:
        return self.conversation_history

    def clear_history(self) -> None:
        self.conversation_history = []
