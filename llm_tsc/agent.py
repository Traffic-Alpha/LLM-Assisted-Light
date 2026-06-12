'''
@Author: WANG Maonan
@Date: 2026-06-11 17:05:17
@Description:  Lightweight ReAct implementation for LLM-based decision making
@LastEditTime: 2026-06-12 23:14:47
'''
import json
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass
from llm_tsc.logger import logger

try:
    from openai import OpenAI
except ImportError:
    raise ImportError("Please install openai: pip install openai>=1.0.0")


@dataclass
class ConversationMessage:
    """Unified message format for conversation history"""
    role: str  # "user", "assistant", "tool"
    content: str
    tool_name: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None


class SimplifiedReActAgent:
    """
    Lightweight ReAct (Reasoning + Acting) agent using OpenAI API
    
    Flow:
    1. User query → LLM (with function calling)
    2. LLM returns thought + tool call (or final answer)
    3. Execute tool → observation
    4. Add observation to history → goto step 2 (or stop if final answer)
    
    Advantages over LangChain:
    - ~300 lines of code vs 5000+ dependencies
    - Full transparency and control
    - No API version breaking changes
    - Minimal dependencies (only OpenAI SDK)
    """
    
    def __init__(
        self,
        llm_client: OpenAI,
        tools_dict: Dict[str, Callable],
        system_prompt: str = "",
        max_iterations: int = 8,
        model: str = "gpt-4",
        temperature: float = 0.0,
        verbose: bool = True,
    ):
        """
        Initialize ReAct Agent
        
        Args:
            llm_client: OpenAI client instance
            tools_dict: Dict[tool_name] -> Callable
            system_prompt: System role description
            max_iterations: Maximum ReAct loops
            model: OpenAI model name (gpt-4, gpt-3.5-turbo, etc.)
            temperature: LLM temperature
            verbose: Print intermediate steps
        """
        self.llm = llm_client
        self.tools = tools_dict
        self.system_prompt = system_prompt
        self.max_iterations = max_iterations
        self.model = model
        self.temperature = temperature
        self.verbose = verbose
        
        self.conversation_history: List[ConversationMessage] = []
        self.tool_specs: List[Dict[str, Any]] = []
        self._build_tool_specs()
    
    def _build_tool_specs(self) -> None:
        """Build OpenAI function calling spec from tools"""
        self.tool_specs = []
        for tool_name, tool_func in self.tools.items():
            # Extract metadata from tool function
            spec = {
                "type": "function",
                "function": {
                    "name": tool_name,
                    "description": getattr(
                        tool_func,
                        "_tool_description",
                        tool_func.__doc__ or f"Tool: {tool_name}",
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": self._get_func_params(tool_func),
                        "required": self._get_required_params(tool_func),
                    }
                }
            }
            self.tool_specs.append(spec)
    
    def _get_func_params(self, func: Callable) -> Dict[str, Any]:
        """Extract parameter specs from function signature"""
        import inspect
        
        sig = inspect.signature(func)
        params = {}
        
        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue
            
            # Infer type from annotation
            param_type = "string"  # default
            if param.annotation != inspect.Parameter.empty:
                type_map = {
                    int: "integer",
                    float: "number",
                    bool: "boolean",
                    str: "string",
                }
                param_type = type_map.get(param.annotation, "string")
            
            params[param_name] = {
                "type": param_type,
                "description": f"Parameter: {param_name}"
            }
        
        return params
    
    def _get_required_params(self, func: Callable) -> List[str]:
        """Get required parameters (without defaults)"""
        import inspect
        
        sig = inspect.signature(func)
        required = []
        
        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue
            if param.default == inspect.Parameter.empty:
                required.append(param_name)
        
        return required
    
    def _build_messages(self) -> List[Dict[str, Any]]:
        """Convert conversation history to OpenAI message format"""
        messages = []
        
        if self.system_prompt:
            messages.append({
                "role": "system",
                "content": self.system_prompt
            })
        
        for msg in self.conversation_history:
            if msg.role == "tool":
                # Tool response uses tool_call_id
                messages.append({
                    "role": "tool",
                    "tool_call_id": msg.tool_call_id,
                    "content": msg.content,
                })
            else:
                openai_msg = {
                    "role": msg.role,
                    "content": msg.content,
                }
                if msg.role == "assistant" and msg.tool_calls:
                    openai_msg["tool_calls"] = msg.tool_calls
                messages.append(openai_msg)
        
        return messages
    
    def _execute_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> str:
        """Execute tool and return observation"""
        if tool_name not in self.tools:
            return f"Error: Tool '{tool_name}' not found. Available tools: {list(self.tools.keys())}"
        
        try:
            tool_func = self.tools[tool_name]
            result = tool_func(**tool_args)
            
            if self.verbose:
                logger.info(f"✓ Tool '{tool_name}' executed successfully")
            
            # Convert result to string if needed
            if isinstance(result, (dict, list)):
                return json.dumps(result, ensure_ascii=False, indent=2)
            else:
                return str(result)
        
        except Exception as e:
            error_msg = f"Tool execution error: {str(e)}"
            logger.error(error_msg)
            return error_msg
    
    def run(self, user_query: str, reset_history: bool = False) -> str:
        """
        Run ReAct loop until final answer or max iterations
        
        Args:
            user_query: User's query/task
            reset_history: Reset conversation history before run
        
        Returns:
            Final answer from LLM
        """
        if reset_history:
            self.conversation_history = []
        
        # Add user message to history
        self.conversation_history.append(
            ConversationMessage(role="user", content=user_query)
        )
        
        if self.verbose:
            logger.info(f"🚀 Starting ReAct loop (max {self.max_iterations} iterations)")
        
        for iteration in range(self.max_iterations):
            if self.verbose:
                logger.info(f"\n📍 Iteration {iteration + 1}/{self.max_iterations}")
            
            # Get LLM response with function calling
            messages = self._build_messages()
            
            try:
                response = self.llm.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=self.tool_specs,
                    temperature=self.temperature,
                    tool_choice="auto",
                )
            except Exception as e:
                logger.error(f"LLM API error: {str(e)}")
                return f"Error calling LLM: {str(e)}"
            
            # Check stop reason
            stop_reason = response.choices[0].finish_reason
            
            if stop_reason == "tool_calls":
                # LLM wants to use tools
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

                # Add assistant message to history
                self.conversation_history.append(
                    ConversationMessage(
                        role="assistant",
                        content=assistant_message.content or "",
                        tool_calls=tool_calls,
                    )
                )
                
                if self.verbose:
                    if assistant_message.content:
                        logger.info(f"💭 LLM: {assistant_message.content[:100]}...")
                
                # Process all tool calls
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
                        
                        # Execute tool
                        observation = self._execute_tool(tool_name, tool_args)
                        
                        if self.verbose:
                            logger.info(f"📊 Observation: {observation[:100]}...")
                        
                        # Add tool response to history
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
                # LLM finished with final answer
                final_answer = response.choices[0].message.content
                
                if self.verbose:
                    logger.info(f"✅ Final Answer: {final_answer[:200]}...")
                
                # Add final response to history
                self.conversation_history.append(
                    ConversationMessage(role="assistant", content=final_answer or "")
                )
                
                return final_answer or "No response"
            
            else:
                logger.warning(f"Unexpected stop reason: {stop_reason}")
                break
        
        # Max iterations reached
        logger.warning("Max iterations reached without final answer")
        return "Max iterations reached without reaching a final decision"
    
    def get_history(self) -> List[ConversationMessage]:
        """Get conversation history"""
        return self.conversation_history
    
    def clear_history(self) -> None:
        """Clear conversation history"""
        self.conversation_history = []
