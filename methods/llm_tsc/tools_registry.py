'''
@Author: WANG Maonan
@Date: 2026-06-11 17:05:18
@Description: 
@LastEditTime: 2026-06-11 17:05:19
'''
"""
Tools Registry and Decorator System
Provides @tool decorator for easy tool definition and registration
"""

from typing import Callable, Any, Dict, Optional
from functools import wraps


def tool(
    name: Optional[str] = None,
    description: Optional[str] = None,
):
    """
    Decorator to mark a function as a ReAct tool
    
    Usage:
        @tool(name="get_occupancy", description="Get current traffic occupancy")
        def get_occupancy(self, junction_id: str) -> str:
            ...
    
    Args:
        name: Tool name (if None, uses function name)
        description: Tool description (if None, uses function docstring)
    """
    def decorator(func: Callable) -> Callable:
        # Set metadata
        func._tool_name = name or func.__name__
        func._tool_description = description or func.__doc__ or f"Tool: {func.__name__}"
        func._is_tool = True
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        
        # Preserve metadata on wrapper
        wrapper._tool_name = func._tool_name
        wrapper._tool_description = func._tool_description
        wrapper._is_tool = True
        
        return wrapper
    
    return decorator


class ToolRegistry:
    """
    Central registry for all ReAct tools
    Provides utility methods for tool discovery and management
    """
    
    def __init__(self):
        self._tools: Dict[str, Callable] = {}
    
    def register(self, name: str, func: Callable) -> None:
        """Register a tool"""
        self._tools[name] = func
    
    def register_from_object(self, obj: Any) -> Dict[str, Callable]:
        """
        Discover and register all @tool decorated methods from an object
        Returns dict of registered tools
        
        Example:
            class TSCTools:
                @tool(name="get_occupancy", description="...")
                def get_occupancy(self, junction_id):
                    ...
            
            registry = ToolRegistry()
            tools = registry.register_from_object(TSCTools())
        """
        tools = {}
        for attr_name in dir(obj):
            attr = getattr(obj, attr_name, None)
            if callable(attr) and hasattr(attr, '_is_tool') and attr._is_tool:
                tool_name = attr._tool_name
                self._tools[tool_name] = attr
                tools[tool_name] = attr
        
        return tools
    
    def get_tools_dict(self) -> Dict[str, Callable]:
        """Get all registered tools as a dict"""
        return dict(self._tools)
    
    def get_tool(self, name: str) -> Optional[Callable]:
        """Get a specific tool"""
        return self._tools.get(name)
    
    def list_tools(self) -> Dict[str, str]:
        """List all tools with their descriptions"""
        return {
            name: getattr(func, '_tool_description', 'No description')
            for name, func in self._tools.items()
        }
    
    def __len__(self) -> int:
        return len(self._tools)
    
    def __repr__(self) -> str:
        return f"ToolRegistry({len(self._tools)} tools)"
