"""Message serialization, tool registration and function-calling protocol."""
import json
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, Dict, List, Optional

from loguru import logger


@dataclass
class ConversationMessage:
    """Unified message format for conversation history"""
    role: str  # "user", "assistant", "tool"
    content: str
    tool_name: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    reasoning_content: Optional[str] = None


def tool(name: Optional[str] = None, description: Optional[str] = None):
    def decorator(func: Callable) -> Callable:
        func._tool_name = name or func.__name__
        func._tool_description = description or func.__doc__ or f'Tool: {func.__name__}'
        func._is_tool = True

        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        wrapper._tool_name = func._tool_name
        wrapper._tool_description = func._tool_description
        wrapper._is_tool = True
        return wrapper
    return decorator


class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, Callable] = {}

    def register(self, name: str, func: Callable) -> None:
        self._tools[name] = func

    def register_from_object(self, obj: Any) -> Dict[str, Callable]:
        tools = {}
        for attr_name in dir(obj):
            attr = getattr(obj, attr_name, None)
            if callable(attr) and hasattr(attr, '_is_tool') and attr._is_tool:
                tool_name = attr._tool_name
                self._tools[tool_name] = attr
                tools[tool_name] = attr
        return tools

    def get_tools_dict(self) -> Dict[str, Callable]:
        return dict(self._tools)

    def get_tool(self, name: str) -> Optional[Callable]:
        return self._tools.get(name)

    def list_tools(self) -> Dict[str, str]:
        return {name: getattr(func, '_tool_description', 'No description') for name, func in self._tools.items()}

    def __len__(self) -> int:
        return len(self._tools)

    def __repr__(self) -> str:
        return f'ToolRegistry({len(self._tools)} tools)'


def build_tool_specs(tools: Dict[str, Callable]) -> List[Dict[str, Any]]:
    specs = []
    for name, function in tools.items():
        specs.append({"type": "function", "function": {
            "name": name,
            "description": getattr(function, "_tool_description", function.__doc__ or f"Tool: {name}"),
            "parameters": {"type": "object", "properties": get_func_params(function),
                           "required": get_required_params(function)},
        }})
    return specs


def get_func_params(func: Callable) -> Dict[str, Any]:
    import inspect
    sig = inspect.signature(func)
    params = {}
    for param_name, param in sig.parameters.items():
        if param_name == "self":
            continue
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


def get_required_params(func: Callable) -> List[str]:
    import inspect
    sig = inspect.signature(func)
    required = []
    for param_name, param in sig.parameters.items():
        if param_name == "self":
            continue
        if param.default == inspect.Parameter.empty:
            required.append(param_name)
    return required


def build_messages(history: List[ConversationMessage], system_prompt: str) -> List[Dict[str, Any]]:
    messages = []
    if system_prompt:
        messages.append({
            "role": "system",
            "content": system_prompt
        })
    for msg in history:
        if msg.role == "tool":
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
            if msg.role == "assistant" and msg.reasoning_content is not None:
                openai_msg["reasoning_content"] = msg.reasoning_content
            messages.append(openai_msg)
    return messages


def execute_tool(tools: Dict[str, Callable], tool_name: str, tool_args: Dict[str, Any], verbose: bool) -> str:
    if tool_name not in tools:
        return json.dumps({"error": {"code": "unknown_tool", "message": tool_name,
                                      "available_tools": list(tools)}})
    try:
        tool_func = tools[tool_name]
        result = tool_func(**tool_args)
        if verbose:
            logger.info(f"✓ Tool '{tool_name}' executed successfully")
        if isinstance(result, (dict, list)):
            return json.dumps(result, ensure_ascii=False, indent=2)
        else:
            return str(result)
    except Exception as e:
        error_msg = f"Tool execution error: {str(e)}"
        logger.error(error_msg)
        return json.dumps({"error": {"code": "tool_execution_error", "message": str(e)}})
